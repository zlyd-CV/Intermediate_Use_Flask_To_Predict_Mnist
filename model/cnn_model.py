"""
@file:cnn_model.py
@time:2020/2/3-14:48
"""
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 可自定义超参数
batch_size = 256
learning_rate = 0.0001
num_epoches = 5


class MNISTCnn(nn.Module):
    def __init__(self):
        super(MNISTCnn, self).__init__()
        # 4层卷积，2层最大池化，卷积后用使用批标准化加快收敛速度，用Relu增加非线性，最后用全连接层得出分类得分
        # 卷积层使用torch.nn.Conv2d来构建，激活层使用torch.nn.ReLU来构建，池化层使用torch.nn.MaxPool2d来构建，全连接层使用torch.nn.Linear来构建
        # BatchNorm2d批标准化用于加快收敛速度

        # 2、一张图片的大小为[1,28,28]，即 深度为 1，高度 28， 宽度 28。
        # 3、nn.Conv2d(1,16,5,1,2) 函数中，我们定义输出深度定义为 16，卷积核大小定义为 3X3，滑动步长定义为 1，使输出空间与输入空间相同尺寸，那么可以通过输出空间公式 (n−m+2p)/s+1，得到填充 0 的数量为 2。
        # 4、卷积层的输出为 [16，28，28]，经过激活层 nn.ReLU()，使用 2X2 的 两层池化层池化，得到池化层输出空间大小为 [16,14,14]。那么全连接层的输入空间大小就为 16X14X14。
        # 	        self.layer1 = nn.Sequential(  # 1, 28, 28
        # 	            nn.Conv2d(1, 16, 5, 1, 2),   # 卷积层,输入深度为1,输出深度16,卷积核5*5,步长1,padding=(kernel_size-1)/2如果stride=1
        # 	            nn.ReLU(),  # 激活层
        # 	            nn.MaxPool2d(kernel_size=2)  # 池化层
        # 	        )  # 输出: 16, 14, 14

        # [1,28,28]
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3),  # [16,26,26]
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3),  # [32,24,24]
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))  # [32,12,12]

        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),  # [64,10,10]
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3),  # [128,8,8]
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))  # [128,4,4]
        #
        self.fc = nn.Sequential(nn.Linear(128 * 4 * 4, 1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def train(net, train_data, num_epoches, optimizer, criterion):
    """
    训练网络
    :return:
    """
    batch_num = len(train_data)  # 统计DataLoader对象的批次数
    length = len(train_data.dataset)  # DataLoader对象里的Dataset原始数据个数
    print(f"length:{length}")

    for epoch in range(num_epoches):
        train_loss = 0.0
        total_correct = 0

        # 启用 BatchNormalization 和 Dropout
        net.train()
        for index, data in enumerate(train_data):
            img, label = data
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            # 得到网络前向传播的结果
            output = net(img)
            # 计算预测结果out和实际结果label的误差损失，（out为每个预测分类的概率）
            loss = criterion(output, label)  # 返回的是一个batch的平均损失
            # 将梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 通过梯度做一步参数更新
            optimizer.step()
            # torch.max返回指定维度（1）中的最大值和相应序号，所以pred为预测的分类
            pred = torch.argmax(output, dim=1)

            train_loss += loss.item() * batch_size
            batch_correct = torch.sum(pred == label).item()
            total_correct += batch_correct
            batch_acc = batch_correct / batch_size
            if index % 100 == 0:
                print('Epoch {}/{},Iter {}/{} Loss: {:.4f},batch_acc:{}'.format(epoch + 1, num_epoches,
                                                                                index,
                                                                                batch_num,
                                                                                loss.item(),
                                                                                batch_acc))
        train_acc = total_correct / length
        total_train_loss = train_loss / length  # 该epoches下训练集平均损失
        print("-"*50)
        print(f"Epoch {epoch + 1}/{num_epoches}, total_train_loss:{total_train_loss:.4f},train_acc:{train_acc:.4f}")
        print("-"*50)

def test(model, test_loader, criterion):
    """
    测试网络
    :param model:
    :param test_loader:
    :return:
    """
    model.eval()  # 将模型改为预测模式，自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    eval_loss = 0.0
    eval_correct = 0
    for data in test_loader:
        img, label = data
        with torch.no_grad():  # 在评估模式或不需要梯度时使用
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            out = model(img)
            pred = torch.argmax(out, dim=1)
            loss = criterion(out, label)
            eval_loss += loss.item() * batch_size
            eval_correct += torch.sum(pred == label).item()

    eval_loss = eval_loss / len(test_loader.dataset)
    eval_acc = eval_correct / len(test_loader.dataset)
    print('Test Loss:{:.4f},ACC:{:.4f}'.format(eval_loss, eval_acc))


if __name__ == '__main__':
    # 1、将各种预处理操作组合到一起 torch.FloadTensor，并归一化到[0, 1.0] 0-1减均值除方差 -0.5-0.5 -1-1
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # 2、下载训练集MNIST手写数字训练集 root指定了数据集存放的路径，transform指定导入数据集时需要进行何种变换操作，train设置为True说明导入的是训练集合，否则为测试集合。
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    # 3、batch_size设置了每批装载的数据图片为64个，shuffle设置为True在装载过程中为随机乱序
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)
    # 打印MNIST数据集的训练集及测试集的尺寸
    print(train_dataset.data.size())
    # 4、解析数据集，查看图片
    # plt.imshow(train_dataset.data[0].numpy(), cmap='gray')
    # plt.title('%i' % train_dataset.targets[0])
    # plt.show()
    # 5、建立一个数据迭代器 该接口的目的：将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MNISTCnn()
    # 查看网络层参数
    print(model.parameters)
    if torch.cuda.is_available():
        model = model.cuda()
    # 6、交叉熵、优化
    criterion = nn.CrossEntropyLoss()
    # 将模型的参数作为需要更新的参数传入优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 7、训练网络
    train(model, train_loader, num_epoches, optimizer, criterion)
    # 8、保存网络
    # torch.save(model.state_dict(), 'model.pth')

    # 9、实例化网络，加载字典参数
    model = MNISTCnn()
    model.load_state_dict(torch.load('model.pth', map_location='cuda'))  # or "cuda"
    model.eval()
    model.to('cuda')
    print("*************开始测试*********************")
    test(model, test_loader, criterion)
