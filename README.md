# 一、运行环境

- Windows
- Python 3.8
- Anaconda3
- pytorch 1.8.1
- PyCharm Community Edition 2020.2.3 x64
- CPU

# 二、数据集地址

* MoNuSeg Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)
* GLAS Dataset - [Link (Original)](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)

### 准备以下格式的数据集，以便于代码使用：

```angular2html
├── datasets
    ├── GlaS
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── MoNuSeg
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol
```

# 三、开始

### 1、Conda用户创建好虚拟环境后在pycharm中导入Anaconda3中配置好的环境

### 2、安装 ```requirements.txt```

```angular2html
pip install -r requirements.txt
```

### 3、训练模型

```angular2html
python train_model.py
```

### 4、修改config.py

 把代码修改成模型训练后产生的文件名，例子

```angular2html
test_session = "Test_session_05.19_21h10"
```

### 5、测试模型获得可视化结果

```angular2html
python test_model.py
```

### 6、补充

模型训练的结果在MoNuSeg_visualize_test目录下

由于训练结果过大（765M），上传GitHub时受到大小限制，故已移除，使用时需要重新训练

