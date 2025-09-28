from model.cnn_model import MNISTCnn
from flask import Flask, render_template, request
import numpy as np
import torch
import re
import base64
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

model_file = r"D:\GitHub Project\Deep Learning\PytorchMinst-master\PytorchMinst-master\model\model.pth"
# 在这里加载模型，但要确保 MNISTCnn 先被导入
# 添加 map_location='cpu' 有时可以解决不同环境下的反序列化问题。
try:
    model = MNISTCnn()
    model.load_state_dict(torch.load(model_file, map_location='cuda', weights_only=False))  # or "cuda" if using GPU
    model.eval()  # 将模型设置为评估模式
    model.to('cuda')
except Exception as e:
    print(f"加载模型时出错: {e}")
    # 适当处理错误，例如退出或显示维护页面
    model = None  # 或者一个空模型/占位模型


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['Get', 'POST'])
def predict():
    parseImage(request.get_data())
    img = to_transform()
    img = img.to("cuda")
    response = predict(img)

    return response


def parseImage(imgData):
    """
    生成图片
    :param imgData:
    :return:
    """
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    with open('./output.png', 'wb') as output:
        output.write(base64.decodebytes(imgStr))


def to_transform():
    """
    将图片转换为28*28尺寸的灰度图并归一化
    # 增加一个维度 1 28 28
    :return:
    """
    img = Image.open('output.png')
    df_transforms = transforms.Compose(
        [transforms.Resize((28, 28),interpolation=transforms.InterpolationMode.LANCZOS),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor()])
    copy_img = df_transforms(img)
    copy_img = copy_img.unsqueeze(0)
    return copy_img


def predict(img):
    """
    预测数字
    :return:
    """
    out = model(img)

    _, pred = torch.max(out, 1)
    # 转换张量tensor([4])为numpy数组得到一般数字
    resp = pred[0].cpu().numpy()  # <-- move to CPU before numpy()

    return str(resp)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888)
