# Intermediate_Use_Flask_To_Predict_Mnist
## 一、事先声明
+ 在您进入到本项目前，作者先对本项目难度做一个解释，作者本人所有项目难度划分运用以下规则：
  + Simple:入门级别，不包含模型部署只负责训练，只提供少量的可视化功能，模型大多为简单模型(参考ResNet与U-Net难度)。
  + Intermediate:中等级别，大部分项目不包含前端部署，模型大多为2016年之后提出的。
  + Advanced:困难级别，基本上包含模型部署模块，模型多为2020年后提出的新时代模型架构(例如Mamba)。
  + Competition/Thesis:个人参与学术竞赛与发表论文的项目，出于部分原因可能项目会缺失数据集等。

## 二、项目介绍
+ 本项目源代码来自https://github.com/ghplvh/PytorchMinst 仓库，但原始代码存在一些逻辑问题和较为过时的torch语法，在本仓库中已修正。
+ 本项目适合希望实现简单的模型部署的初学者，对于工业应用以学习ONNX和TensorRT为主。
+ 优化内容：①训练模型代码；②Web前端显示效果；③增加了版本检查，cuda检查并灵活应用；④修复了一些随着包版本不兼容导致的问题。

## 三、内容介绍(只介绍关键部分)
+ 本项目包含：
+ requirements.txt：包的版本，运行下面命令即可下载到虚拟环境中，pytorch请前往官网下载
 ```txt
pip install -r requirements.txt
```
+ cnn.model.py：模型训练主程序
+ app.py：flask生成本地网页端主程序

## 四、运行展示
![Web端效果展示](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/610062f16c08e609ab9b47da1dc872768e3bf002/Intermediate_Use_Flask_To_Predict_Mnist/%E8%BF%90%E8%A1%8C%E7%A4%BA%E6%84%8F%E5%9B%BE.png)

## 五、部分资源下载地址
+ pytorch官网下载带cuda的pytorch：https://pytorch.org
+ Anaconda官网下载地址：https://anaconda.org/anaconda/conda
# 最后，再次感谢https://github.com/ghplvh/PytorchMinst 仓库提供的原始代码
