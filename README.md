# UCAS-CV-homework

## Install

 在文件目录下打开控制台，开启虚拟环境,输入
```bash
$ pip install -r requirements.txt
```


## Train

直接用juyter run all“训练”文件即可

定义model的格子：可修改网络参数，或直接替换为自己的神经网络。

*数据读入部分对原图和特征图同时做了随机裁剪（到384）和随机水平翻转防止过拟合，将特征图处理为3分类的one_hot tensor*

*因此自选模型需要满足可输入384 channel=3, 输出384 channel=3*


<img src=".\README\01.png" width="600px"></img>



## Evaluate

直接用juyter run all“画结果”文件即可

修改网络参数，或直接替换神经网络同上

二选一用自带的权重画图，或者刚刚训练好的权重画图

结果按照训练集和测试集的比例的正确率排序

<img src=".\README\02.png" width="600px"></img>