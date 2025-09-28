# 忽视flask的警告
import sys
import torch
import numpy as np
import matplotlib
import flask
from importlib.metadata import version  # 导入获取版本的工具
import platform
from packaging import version

if __name__ == '__main__':
    # 检查各库版本
    print("python 版本：",sys.version)
    print("pytorch 版本：", torch.__version__)
    print("NumPy 版本：", np.__version__)
    print("matplotlib版本：", matplotlib.__version__)
    print("flask版本：", flask.__version__)
    """
    # 用于生成.txt文件
    print("numpy>=", np.__version__)
    print("matplotlib>=", matplotlib.__version__)
    print("flask>=", flask.__version__)
    """
    # 以下检查不强行需求
    if version.parse(platform.python_version()) <= version.parse("3.9"):
        raise ValueError("python 版本不匹配")
    else:
        print("python 版本符合要求")

    # 检查 pytorch 版本
    if version.parse(torch.__version__) != version.parse("2.7.1+cu118"):
        raise ValueError("torch 版本不匹配")
    else:
        print("torch 版本符合要求")

    # 检查 numpy 版本
    if version.parse(np.__version__) != version.parse("1.23.5"):
        raise ValueError("numpy 版本不匹配")
    else:
        print("numpy 版本符合要求")

    # 检查 matplotlib 版本
    if version.parse(matplotlib.__version__) != version.parse("3.7.2"):
        raise ValueError("matplotlib 版本不匹配")
    else:
        print("matplotlib 版本符合要求")

    # 检查 flask 版本
    if version.parse(flask.__version__) != version.parse("3.1.2"):
        raise ValueError("flask 版本不匹配")
    else:
        print("flask 版本符合要求")