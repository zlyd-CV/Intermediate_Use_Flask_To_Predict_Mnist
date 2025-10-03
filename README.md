# Use_Flask_To_Predict_Mnist

## 一、项目介绍
+ 本项目源代码来自https://github.com/ghplvh/PytorchMinst 仓库，但原始代码存在一些逻辑问题和较为过时的torch语法，在本仓库中已修正。
+ 本项目适合希望实现简单的模型部署的初学者，对于工业应用以学习ONNX和TensorRT为主。
+ 优化内容：①训练模型代码；②Web前端显示效果；③增加了版本检查，cuda检查并灵活应用；④修复了一些随着包版本不兼容导致的问题。
+ 项目流程图

![项目流程图](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/cbf5e9bf3463cad30ebbd27819774a8bbe5f86b5/Intermediate_Use_Flask_To_Predict_Mnist/Flask%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84.drawio.svg)

## 二、内容介绍(只介绍关键部分)
+ 本项目包含：
+ requirements.txt：包的版本，运行下面命令即可下载到虚拟环境中，pytorch请前往官网下载
 ```txt
pip install -r requirements.txt
```
+ cnn.model.py：模型训练主程序
+ app.py：flask生成本地网页端主程序

## 三、运行展示
![Web端效果展示](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/610062f16c08e609ab9b47da1dc872768e3bf002/Intermediate_Use_Flask_To_Predict_Mnist/%E8%BF%90%E8%A1%8C%E7%A4%BA%E6%84%8F%E5%9B%BE.png)

## 四、部分资源下载地址
+ pytorch官网下载带cuda的pytorch：https://pytorch.org
+ Anaconda官网下载地址：https://anaconda.org/anaconda/conda
# 最后，再次感谢https://github.com/ghplvh/PytorchMinst 仓库提供的原始代码
