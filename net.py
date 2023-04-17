#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.transforms import Compose, ToTensor
from einops import rearrange
from torch.optim import Adam
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint
import pickle
import torchvision.utils as utils

class ResnetBlock(nn.Module):
    
    def __init__(self, dim_in, dim_out, groups  ):
        super().__init__()
        
        self.one=nn.Sequential(nn.Conv2d(dim_in, dim_out , 3, padding=1), 
                               nn.GroupNorm(groups, dim_out), 
                               nn.SiLU(),
                               nn.Conv2d(dim_out , dim_out, 3, padding=1), 
                               nn.GroupNorm(groups, dim_out),
                               nn.SiLU())

        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        
    def forward(self, x):
        
        h = self.one(x)
        
        return h + self.res_conv(x)
       
        
        

class LinearAttention(nn.Module):
    def __init__( self, dim, heads=4, dim_head= 16 ):
        
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        # by testing, this h and c actually can split (dim_head * heads) properly, and has nothing to do with the x,shape

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        
        return self.to_out(out)

    
    
          
class Add_attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.one=nn.Sequential(LinearAttention(dim), 
                               nn.GroupNorm(1 , dim), 
                               nn.GELU())
    def forward(self, x):

        h=self.one(x)
        
        return h + x
 


        
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)
#kernel=4, stripe=2, padding=1, keep the same size

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)



def Cross( input, target, backgroud_scale ):
    
    input = rearrange(input, "b c h w -> b c (h w)")
    target = rearrange(target, "b c h w -> b c (h w)")      
    
    cross = nn.CrossEntropyLoss(weight= torch.tensor([1,1, backgroud_scale]).to('cuda'), reduction = 'mean')
    
    return cross(input, target)



def Dice(input, target, backgroud_scale ,epsilon = 1e-5):
    
    #initial softmax
    softmax = nn.Softmax(dim=1)
    input = softmax(input)
    
    inter = 2 * (input* target).mean([-1,-2])
    sets_sum = input.mean([-1,-2]) + target.mean([-1,-2])
    dice =  (inter + epsilon) / (sets_sum + epsilon)
    diceall = (dice[:,0] + dice[:,1] + backgroud_scale*dice[:,2])/ (2 + backgroud_scale)

    diceall = diceall.mean()
    
    return  1 - diceall


class Unet03(nn.Module):
    def __init__(self, 
                 init_dim,
                 out_dim,
                 channels=3,
                 dim_mults=(1, 2, 4, 8)
                 ):
        
        super().__init__()
        
        resnet_block_groups = init_dim
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        
        self.init_conv = nn.Sequential(
                                        nn.Conv2d( channels, channels+1 , 7, padding=3 ),
                                        block_klass( channels+1 , init_dim )
                                       )

            
        dims = [*map(lambda m: init_dim * m, dim_mults)] 
        in_out = list(zip(dims[:-1], dims[1:]))
        

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
    
        for ind, (dim_in, dim_out) in enumerate(in_out):   

            self.downs.append(
                    nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out),
                        block_klass(dim_out, dim_out),    
                        Downsample(dim_out) 
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Add_attention(mid_dim)
           
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out)):
            #is_last = ind >= (num_resolutions - 1)
            
            self.ups.append(
                nn.ModuleList(
                        [   
                            Upsample(dim_in), 
                            block_klass(dim_in * 2, dim_out*2, ),
                            block_klass(dim_out*2, dim_out ),
                            Add_attention(dim_out)
                        ]
                    )
                )
       
        self.final_conv = nn.Sequential(
                                    block_klass(dim_out, dim_out), 
                                    nn.Conv2d(dim_out, out_dim, 1)
                                    )

            
    def forward(self, x ):
        
        x = self.init_conv(x)
        
        h = []
        for block1, block2 , downsample in self.downs:
    
            x = block1(x)
            x = block2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_block2(x)
        x = self.mid_attn(x)

        for upsample ,block1, block2 , attn,  in self.ups:
            
            x = upsample(x) 
            x = torch.cat((x, h.pop()), dim=1)    #(list).pop() 
            x = block1(x)
            x = block2(x)
            x = attn(x)

        return self.final_conv(x)


    
#in this case, transform, label_transform should be simoultaneously.
class LocalImaginesDataset(Dataset):
    def __init__(self, img_dir, label_img_dir, transform ):
        self.img_dir = img_dir
        self.label_img_dir = label_img_dir
        self.transform = transform
        
        self.img_names = os.listdir(img_dir)
        self.label_img_names = os.listdir(label_img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        label_img_path = os.path.join(self.label_img_dir, self.label_img_names[index])
        
        try:
            img = Image.open(img_path).convert('RGB')
            label_img = Image.open(label_img_path).convert('RGB')
            #label = Image.open(label_path).convert('L')
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return None, None
        
        img = self.transform(img)
        label_img = self.transform(label_img)
        
        return img, label_img

    
    
    
transform_train = Compose([
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            #ToTensor(),turn the data to torch.FloadTensor，and projected to [0, 1.0]        
            ])



transform_evaluate = Compose([
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            ])


def img_transform(img, label_img):
    
    #linner random crop
    random = torch.randint(20, 77, [2])

    crop_img = img[:,:, random[1]:random[1]+384, random[0]:random[0]+384]
    crop_label = label_img[:,:, random[1]:random[1]+384, random[0]:random[0]+384]
    
    #horizontal flip
    if torch.rand(1) < 0.5:
        crop_img = torch.flip(crop_img, dims=[-1])
        crop_label = torch.flip(crop_label, dims=[-1])
        
    #it is important to strength that label are conbine of [0.,0.,0.](dim=2, so we set dim=2 to background onehot),[0.502,0.,0.],[0,0.502,0.] by testing, 
    #i need to transfer it to one-hot
    crop_label = torch.where( crop_label >=  0.4 , 1. , 0. ) #. for float32
    tem = torch.ones_like(crop_label[:,2,:,:])
    crop_label[:,2,:,:] = tem - crop_label[:,0,:,:] - crop_label[:,1,:,:]
        
    return crop_img, crop_label
 

def img_transform_new(img, label_img):
        
    label_img = torch.where( label_img >=  0.4 , 1. , 0. ) #. for float32
    tem = torch.ones_like(label_img[:,2,:,:])
    label_img[:,2,:,:] = tem - label_img[:,0,:,:] - label_img[:,1,:,:]
        
    return img, label_img    



def train_loop(dataloader, dataloader_val , model, epochs  , Dice_scale = 0.8 , backgroud_scale = 0.6 , learning_rate = 1e-5  ,device = "cuda"):
    #we set model as an imput to impove flexibility
    
    model.train()
    
    optimizer = Adam(model.parameters(),  lr = learning_rate)
    #scaler = torch.cuda.amp.GradScaler()
    
    train_losses = []
    accurates = []
    
    with tqdm(total= epochs , desc=f'training', unit='Epoch') as pbar:
        for epoch in range(1, epochs + 1): 

            running_loss = 0.
            running_loss1 = 0.
            running_loss2 = 0.
            
            for batch, (img, label_img) in enumerate(dataloader):
            
                with torch.no_grad():
                    img, label_img = img_transform(img, label_img)
                    
                img = img.to(device=device)  #, memory_format=torch.channels_last
                label_img = label_img.to(device=device)  #, memory_format=torch.channels_last

                with torch.autocast(device): #save space 
                    
                    pred_img = model( img )
                    loss1 = Cross( pred_img, label_img, backgroud_scale )
                    loss2 =  Dice( pred_img , label_img , backgroud_scale )                    
                    loss = loss1 + Dice_scale*loss2
    

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                running_loss += loss.item()
                
            epoch_loss = (running_loss1 , running_loss2  , running_loss)
            train_losses.append(epoch_loss)
            
            
            if epoch != 0  and epoch % 5 == 0 :
                print('Epoch {}/{}: epoch loss={}'.format(epoch, epochs, running_loss))
                
            if epoch != 0 and epoch % 10 == 0:
                with torch.no_grad():
                    accurate = evaluate_small (model, dataloader_val , epoch )
                    accurates.append(accurate)
            
            pbar.update( 1 )

    with open("training_loss.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    
    with open("accurates.pkl", "wb") as f:
        pickle.dump(accurates, f)
        


def evaluate_small(model, dataloader_val , epoch ,device= 'cuda' ,  save_dir='images'):
    
    to_pil = transforms.ToPILImage()
    size = 23

    pred_onehots = []
    pred_softs= []
    
    accurate_10 = 0
    accurate_5 = 0
    accurate_20 = 0
            
    
    for batch, (img, label_img) in enumerate(dataloader_val):
            
            
            label_img = torch.where( label_img >= 0.4 , 1. , 0. ) #. for float32
            tem = torch.ones_like(label_img[:,2,:,:])
            label_img[:,2,:,:] = tem - label_img[:,0,:,:] - label_img[:,1,:,:]
            
            label_img = torch.sum(label_img, (-1,-2))
            
            img = img.to(device=device)
            model.eval()
            pred_img = model( img )
            model.train()
            

            softmax = nn.Softmax(dim=1)
            pred_soft = softmax(pred_img)
            
            pred_idx = torch.argmax(pred_soft, dim=1) 
            pred_onehot = F.one_hot(pred_idx, 3).permute(0, 3, 1, 2).type(torch.float)
            #pred_onehot = pred_onehot.permute(0, 3, 1, 2).type(torch.float)
                        
            pred_sum = torch.sum(pred_onehot, (-1,-2))
            
            
            pred_onehots.append(pred_onehot)
            pred_softs.append(pred_soft)
            
            
            j = pred_sum[0,1]/(pred_sum[0,0] + pred_sum[0,1])  
            i = label_img[0,1]/(label_img[0,0] + label_img[0,1])
            
            if (j <= 0.1+i and j >= i - 0.1 ):
                accurate_10 += 1
            if (j <= 0.05+i and j >= i - 0.05 ):
                accurate_5 += 1
            if (j <= 0.2+i and j >= i - 0.2 ):
                accurate_20 += 1   
            
            
    print('[ epoch={}|20%范围内的准确率：'.format(epoch) , 100*accurate_20/size,'%  ]\n',
          '[ epoch={}|10%范围内的准确率：'.format(epoch), 100*accurate_10/size , '%  ]\n' ,
          '[ epoch={}|5%范围内的准确率：'.format(epoch) , 100*accurate_5/size ,'%  ]\n' )

    
    # images save 
    
    if epoch != 0 and epoch % 20 == 0:
        pred_soft_dir = os.path.join(save_dir, 'pred_soft{}'.format(epoch))
        os.makedirs(pred_soft_dir, exist_ok=True)
        for i in range(size):
            filename = f'epoch_{epoch}_image_{i}.png'
            filepath = os.path.join(pred_soft_dir, filename)

            img_tensor = pred_softs[i].squeeze(0)
            img_pil = to_pil(img_tensor)
            img_pil.save( filepath)


        pred_onehot_dir = os.path.join(save_dir, 'pred_onehot{}'.format(epoch))
        os.makedirs(pred_onehot_dir, exist_ok=True)
        for i in range(size):
            filename = f'epoch_{epoch}_image_{i}.png'
            filepath = os.path.join(pred_onehot_dir, filename)

            img_tensor = pred_onehots[i].squeeze(0)
            img_pil = to_pil(img_tensor)
            img_pil.save(filepath)
    
            
    return  accurate_10/size , accurate_5/size , accurate_20/size





def plot_training():
    
    with open("training_loss.pkl", "rb") as f:
        train_losses = pickle.load(f)

    with open("accurates.pkl", "rb") as f:
        accurates = pickle.load(f)

    accurates.insert(0, [0, 0, 0])

    train_losses1 = [item[0] for item in train_losses]
    train_losses2 = [item[1] for item in train_losses]
    train_losses = [item[2] for item in train_losses]

    accurate_10 = [item[0] for item in accurates]
    accurate_5 = [item[1] for item in accurates]
    accurate_20 = [item[2] for item in accurates]


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,24) ) #figsize=(10,5)

    ax1.plot(train_losses1, label=' CrossEntropy Loss')
    ax1.plot(train_losses2, label=' Dice Loss')
    ax1.plot(train_losses, label='Total Loss', linewidth= 2 )
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses',fontsize=25)

    ax2.plot(accurate_20, label='Accuracy within 20%')
    ax2.plot(accurate_10, label='Accuracy within 10%')
    ax2.plot(accurate_5, label='Accuracy within 5%')
    ax2.legend()
    ax2.set_xlabel(' 5*Epoch')
    ax2.set_ylabel('accurate')
    ax2.set_title('Test-data Accuates',fontsize=25)

    for i, acc in enumerate(accurate_10):
        ax2.text(i, acc, f'{acc*100:.0f}%', ha='center', fontsize=10)

    for i, acc in enumerate(accurate_5):
        ax2.text(i, acc, f'{acc*100:.0f}%', ha='center', fontsize=10)

    for i, acc in enumerate(accurate_20):
        ax2.text(i, acc, f'{acc*100:.0f}%', ha='center', fontsize=10)

    plt.show()
    

def evaluate_result(model, dataloader_val  ,device= 'cuda'):
    
    to_pil = transforms.ToPILImage()
    size = len(dataloader_val)

    pred_onehots = []
    pred_labels= []
    original = []
    
    accurate_10 = 0
    accurate_5 = 0
    accurate_20 = 0
    
    label_ratio = []
    pre_ratio = []
    
    
    for batch, (img, label_img) in enumerate(dataloader_val):
            
            
            original.append(img)
            img = img.unsqueeze(0)
            label_img = label_img.unsqueeze(0)
            
            label_img = torch.where( label_img >= 0.4 , 1. , 0. ) #. for float32
            
            tem = torch.ones_like(label_img[:,2,:,:])
            label_img[:,2,:,:] = tem - label_img[:,0,:,:] - label_img[:,1,:,:]
            pred_label  = label_img
            
            label_img = torch.sum(label_img, (-1,-2))
            
            
            img = img.to(device=device)
            model.eval()
            pred_img = model( img )
            model.train()
            

            softmax = nn.Softmax(dim=1)
            pred_soft = softmax(pred_img)
            
            pred_idx = torch.argmax(pred_soft, dim=1) 
            pred_onehot = F.one_hot(pred_idx, 3).permute(0, 3, 1, 2).type(torch.float)
            #pred_onehot = pred_onehot.permute(0, 3, 1, 2).type(torch.float)
                        
            pred_sum = torch.sum(pred_onehot, (-1,-2))
            
            
            pred_onehots.append(pred_onehot.squeeze(0))
            pred_labels.append(pred_label.squeeze(0))
            
            
            j = pred_sum[0,1]/(pred_sum[0,0] + pred_sum[0,1])  
            i = label_img[0,1]/(label_img[0,0] + label_img[0,1])
            
            pre_ratio.append(j)
            label_ratio.append(i)
            
            if (j <= 0.1+i and j >= i - 0.1 ):
                accurate_10 += 1
            if (j <= 0.05+i and j >= i - 0.05 ):
                accurate_5 += 1
            if (j <= 0.2+i and j >= i - 0.2 ):
                accurate_20 += 1   
            
            
    # print('[ epoch={}|20%范围内的准确率：'.format(epoch) , 100*accurate_20/size,'%  ]\n',
    #       '[ epoch={}|10%范围内的准确率：'.format(epoch), 100*accurate_10/size , '%  ]\n' ,
    #       '[ epoch={}|5%范围内的准确率：'.format(epoch) , 100*accurate_5/size ,'%  ]\n' )


    return  accurate_10/size , accurate_5/size , accurate_20/size , pred_onehots , pred_labels ,original ,pre_ratio ,label_ratio




def plot_train(model,dataset_trainplot):
    
    a10 , a5 ,  a20 , pred_onehots , pred_labels ,original ,pre_ratio ,label_ratio = evaluate_result(model, dataset_trainplot )
    diff = [abs(x-y) for x, y in zip(pre_ratio, label_ratio)]
    top5_idx = sorted(range(len(diff)), key=lambda i: diff[i])[:30] #reverse=True

    to_pil = transforms.ToPILImage()
    fig, axs = plt.subplots(30, 4 ,figsize=(20, 150) )#,figsize=(20, 80)

    for i, ax_row in enumerate(axs):
        ax_row[0].imshow(to_pil(original[top5_idx[i]]))
        ax_row[0].axis('off')
        ax_row[1].imshow(to_pil(pred_labels[top5_idx[i]]))
        ax_row[1].axis('off')
        ax_row[2].imshow(to_pil(pred_onehots[top5_idx[i]]))
        ax_row[2].axis('off')
        ax_row[3].text(0.4, 0.4, f'Prediction={100*pre_ratio[top5_idx[i]]:.2f}%\nLabel ={100*label_ratio[top5_idx[i]]:.2f}%', ha='center', va='center', fontsize=15)
        ax_row[3].axis('off')

    # fig.set_extent([0, 10, 0, 10]) # set the size of the image in data coordinates
    # fig.set_aspect('equal') 


    #fig.set_size_inches(5, 5)
    fig.suptitle("Traindata Accurate", fontsize=25, fontweight="bold")
    plt.show()


def plot_test(model,dataset_val):

    a10 , a5 ,  a20 , pred_onehots , pred_labels ,original ,pre_ratio ,label_ratio = evaluate_result(model, dataset_val )
    diff = [abs(x-y) for x, y in zip(pre_ratio, label_ratio)]
    top5_idx = sorted(range(len(diff)), key=lambda i: diff[i])[:23] #reverse=True

    to_pil = transforms.ToPILImage()
    fig, axs = plt.subplots(23, 4 ,figsize=(20, 115 ))#,figsize=(20, 80)

    for i, ax_row in enumerate(axs):
        ax_row[0].imshow(to_pil(original[top5_idx[i]]))
        ax_row[0].axis('off')
        ax_row[1].imshow(to_pil(pred_labels[top5_idx[i]]))
        ax_row[1].axis('off')
        ax_row[2].imshow(to_pil(pred_onehots[top5_idx[i]]))
        ax_row[2].axis('off')
        ax_row[3].text(0.4, 0.4, f'Prediction={100*pre_ratio[top5_idx[i]]:.2f}%\nLabel ={100*label_ratio[top5_idx[i]]:.2f}%', ha='center', va='center', fontsize=15)
        ax_row[3].axis('off')


    fig.suptitle("Testdata Accurate", fontsize=25, fontweight="bold")
    plt.show()