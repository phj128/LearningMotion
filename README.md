# Learning Motion

## Motivation of this repository

1. 面向实验室本科生的科研训练。通过复现[SAMP](https://samp.is.tue.mpg.de/)的MotionNet来学习基于自回归模型的Motion Synthesis算法、PyTorch编程。
2. 这个框架主要由我个人使用，如果有一些design上的建议请直接提issue。

## 如何使用这个代码框架 

1. 这个代码只包括使用pytorch训练网络模型的代码，并实现了torch模型转ONNX的功能，把ONNX模型放到Unity中来进行实际的使用。
2. 目前所要求的任务并不需要写Unity的代码，但需要在[Unity](https://github.com/mohamedhassanmus/SAMP_Training)中运行来查看模型效果。

## Prerequisites

### Unity

1. 安装好[Unity Editor](https://unity.com/releases/editor/whats-new/2019.4.22)，推荐使用2019.4.22f1的版本。

2. 下载好[Unity的代码](https://github.com/mohamedhassanmus/SAMP)。

3. 按照SAMP的流程能够成功运行SAMP的Demo并能够使用WSAD来操作人物和执行和物体的交互。

### Python
1. 确保你已经熟悉使用python, 尤其是debug工具: ipdb。

2. 计算机科学非常讲究自学能力和自我解决问题的能力，如果有一些内容没有介绍的十分详细，请先自己尝试探索代码框架。如果遇到代码问题，请先搜索网上的资料，或者查看仓库的Issues里有没有相似的已归档的问题。

3. 如果有问题，也欢迎直接在这个仓库的Issue里提问。

## Data preparation

Download [SAMP](https://samp.is.tue.mpg.de/) dataset and add a link to the datasets directory. After preparation, you should have the following directory structure: 
```
datasets/samp
|-- MotionNet
|   |-- test
|   |-- train
```


## 从训练MLP来学习这个框架

### 任务定义

训练一个MLP，将一帧的状态（包括pose，trajetory等）作为输入, 输出下一帧的状态。

### Training

```
python train.py --config configs/samp/mlp.yaml
```

### 查看loss曲线

```
tensorboard --logdir=./work_dirs --bind_all
```

### Torch2onnx

```
python torch2onnx.py --config configs/samp/mlp.yaml,configs/samp/onnx/regression.yaml --epoch 100
```

转换好的ONNX模型会储存在对应的文件夹下，此处则为```work_dirs/mlp/onnx/100/mlp.onnx```

### Use ONNX in Unity

把ONNX模型放到Unity代码的文件夹```Assets/OnnxModels/```下。

打开Unity代码文件夹中的```Assets/Demo/Main_Demo.unity```，选中Unity界面左侧的Hierarchy里的03301_red，在右侧的Inspector栏中的SAMPNN(Script)里的Model Asset属性来选择对应的ONNX模型，来查看你训练的效果。

你会发现训练出来的MLP模型仅仅能实现基本的站立，不能完成走路和与物体交互。

## 开始复现SAMP

实现SAMP主要包括实现Scheduled Sampling，MoE和VAE，我们一步一步完成。

### Scheduled Sampling

我们已经在 ```configs/samp/``` 中创建好了一个配置文件， ```configs/samp/mlp_ss.yaml``` 。

实现Scheduled Sampling主要是需要实现连续两帧的feature转换。

完成这一任务需要你修改 ```motion/utils/samp.py``` 里的代码，实现 ```transform_data``` 所需要的各个函数。

你可以发现这个模型能基本的进行前进走路，但其他动作以及和物体交互的效果较差。

### MoE

我们已经在 ```configs/samp/``` 中创建好了一个配置文件， ```configs/samp/moe.yaml``` 。其中包含了使用MoE所需要的参数。

我们已经为你实现了基本的逻辑，需要你来完成具体的网络搭建。

请在 ```motion/modeling/modules/MoE/MoE.py``` 中实现MoE。

在此基础上，在 ```motion/modeling/modules/decoder/moe.py``` 中实现MoEDecoder。

为了能够转换成ONNX，你需要参考 ```motion/modeling/modules/decoder/mlp.py``` 实现MoE对应的ONNX模型。

你可以发现这个模型效果更好，但有的时候躺下动作无法执行。

### VAE

我们已经在 ```configs/samp/``` 中创建好了一个配置文件， ```configs/samp/samp.yaml``` 。其中包含了使用VAE所需要的参数。

我们已经为你实现了基本的逻辑，需要你来完成具体的网络搭建。

完成这一任务需要在 ```motion/modeling/modules/encoder/samp.py``` 中实现VAE的encoder， 在 ```motion/modeling/modules/decoder/samp.py``` 中实现VAE的decoder。请注意，SAMP的MotionNet是在MoE的基础上增加的VAE。

为了能够转换成ONNX，你需要参考 ```motion/modeling/modules/decoder/mlp.py``` 实现VAE对应的ONNX模型。

因为VAE有额外的输入z，你需要使用 ```configs/samp/onnx/vae.yaml``` 来匹配。

至此，你已经完成了复现SAMP，你应该可以得到一个效果和SAMP所提供的模型差不多的模型。

## 框架模块设计的解释
### Engine模块
参考 ```motion/engine/trainer.py```

定义了整个框架的运行逻辑和流程。

### Pipeline模块
参考 ```motion/pipeline/regression.py```

定义了数据在网络里如何传输和计算loss。

### dataset模块

参考 ```motion/dataset/samp.py```

核心函数包括：init, getitem, len.

init函数负责从磁盘中load指定格式的文件，计算并存储为特定形式。

getitem函数负责在运行时提供给网络一次训练需要的输入，以及groundtruth的输出。

len函数是训练或者测试的数量。getitem函数获得的index值通常是[0, len-1]。


### module模块和model模块:

module参考 ```motion/modeling/modules/decoder/mlp.py```

model参考 ```motion/modeling/models/regression.py```

核心函数包括：init, forward.

init函数负责定义网络所必需的模块，forward函数负责接收dataset的输出，利用定义好的模块，计算输出。

我们的框架里Model表示整个模型，Module表示一个模型里的多个小模块，例如SAMP的MotioNet是一整个模型，Encoder和Decoder是这个模型小的模块。

需要在model的定义前使用```@MODELS.register_module()```来注册module。

需要在module的定义前使用```@MODULES.register_module()```来注册module。

### loss模块和criterion模块

loss参考 ```motion/loss/loss.py```

criterion参考 ```motion/criterion/regression.py```

loss模块定义了我们所需要使用的loss

criterion模块定义了测试时如何计算metric或后处理。

### Hook模块

hook参考 ```motion/hook/hook.py```

我们使用Hook来实现了模型的存储，读取，测试和log的记录。

## References
这个代码参考了大量别的代码框架，包括但不限于：

[detectron2](https://github.com/facebookresearch/detectron2)

[mmdetection](https://github.com/open-mmlab/mmdetection)

[DI-engine](https://github.com/opendilab/DI-engine)

[SAMP_Training](https://github.com/mohamedhassanmus/SAMP_Training)