# Learning Motion

## Motivation of this repository

1. 面向实验室本科生的科研训练。通过复现[SAMP](https://samp.is.tue.mpg.de/)的 MotionNet 来学习基于自回归模型的 Motion Synthesis 算法、PyTorch 编程。
2. 这个框架主要由我个人使用，如果有一些 design 上的建议请直接提 issue。

## 如何使用这个代码框架 

1. 这个代码只包括使用 pytorch 训练网络模型的代码，并实现了 torch 模型转 ONNX 的功能，把 ONNX 模型放到 Unity 中来进行实际的使用。
2. 目前所要求的任务并不需要写 Unity 的代码，但需要在[Unity](https://github.com/mohamedhassanmus/SAMP_Training)中运行来查看模型效果。

## Prerequisites

### Unity

1. 安装好 [Unity Editor](https://unity.com/releases/editor/whats-new/2019.4.22)，推荐使用 2019.4.22f1 的版本。

2. 下载好 [Unity 的代码](https://github.com/mohamedhassanmus/SAMP)。

3. 按照SAMP的流程能够成功运行 SAMP 的 Demo 并能够使用 WSAD 来操作人物和执行和物体的交互。

### Python
1. 确保你已经熟悉使用 python, 尤其是 debug 工具: ipdb。

2. 计算机科学非常讲究自学能力和自我解决问题的能力，如果有一些内容没有介绍的十分详细，请先自己尝试探索代码框架。如果遇到代码问题，请先搜索网上的资料，或者查看仓库的 Issues 里有没有相似的已归档的问题。

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

训练一个 MLP，将一帧的状态（包括 pose，trajetory 等）作为输入, 输出下一帧的状态。

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

转换好的 ONNX 模型会储存在对应的文件夹下，此处则为```work_dirs/mlp/onnx/100/mlp.onnx```

### Use ONNX in Unity

把 ONNX 模型放到 Unity 代码的文件夹 ```Assets/OnnxModels/``` 下。

打开 Unity 代码文件夹中的 ```Assets/Demo/Main_Demo.unity```，选中 Unity 界面左侧的 Hierarchy 里的 03301_red，在右侧的 Inspector 栏中的 SAMPNN(Script) 里的 Model Asset 属性来选择对应的ONNX模型，来查看你训练的效果。

你会发现训练出来的MLP模型仅仅能实现基本的站立，不能完成走路和与物体交互。

## 开始复现SAMP

实现 SAMP 主要包括实现 Scheduled Sampling，MoE 和 VAE，我们一步一步完成。

### Scheduled Sampling

我们已经在 ```configs/samp/``` 中创建好了一个配置文件， ```configs/samp/mlp_ss.yaml``` 。

实现 Scheduled Sampling 主要是需要实现连续两帧的feature转换。

完成这一任务需要你修改 ```motion/utils/samp.py``` 里的代码，实现 ```transform_data``` 所需要的各个函数。

你可以发现这个模型能基本的进行前进走路，但其他动作以及和物体交互的效果较差。

### MoE

我们已经在 ```configs/samp/``` 中创建好了一个配置文件， ```configs/samp/moe.yaml``` 。其中包含了使用 MoE 所需要的参数。

我们已经为你实现了基本的逻辑，需要你来完成具体的网络搭建。

请在 ```motion/modeling/modules/MoE/MoE.py``` 中实现 MoE。

在此基础上，在 ```motion/modeling/modules/decoder/moe.py``` 中实现 MoEDecoder。

为了能够转换成ONNX，你需要参考 ```motion/modeling/modules/decoder/mlp.py``` 实现 MoE 对应的 ONNX 模型。

你可以发现这个模型效果更好，但有的时候躺下动作无法执行。

### VAE

我们已经在 ```configs/samp/``` 中创建好了一个配置文件， ```configs/samp/samp.yaml``` 。其中包含了使用 VAE 所需要的参数。

我们已经为你实现了基本的逻辑，需要你来完成具体的网络搭建。

完成这一任务需要在 ```motion/modeling/modules/encoder/samp.py``` 中实现 VAE 的 encoder， 在 ```motion/modeling/modules/decoder/samp.py``` 中实现 VAE 的 decoder。请注意，SAMP 的 MotionNet 是在 MoE 的基础上增加的 VAE。

为了能够转换成 ONNX，你需要参考 ```motion/modeling/modules/decoder/mlp.py``` 实现 VAE 对应的 ONNX 模型。

因为 VAE 有额外的输入 z，你需要使用 ```configs/samp/onnx/vae.yaml``` 来匹配。

至此，你已经完成了复现 SAMP，你应该可以得到一个效果和 SAMP 所提供的模型差不多的模型。

## 框架模块设计的解释

### Engine模块

参考 ```motion/engine/trainer.py```

定义了整个框架的运行逻辑和流程。

### Pipeline模块

参考 ```motion/pipeline/regression.py```

定义了数据在网络里如何传输和计算 loss。

### dataset模块

参考 ```motion/dataset/samp.py```

核心函数包括：init, getitem, len.

init 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式。

getitem 函数负责在运行时提供给网络一次训练需要的输入，以及 groundtruth 的输出。

len函数是训练或者测试的数量。getitem 函数获得的 index 值通常是[ 0, len-1]。


### module模块和model模块:

module参考 ```motion/modeling/modules/decoder/mlp.py```

model参考 ```motion/modeling/models/regression.py```

核心函数包括：init, forward.

init 函数负责定义网络所必需的模块，forward 函数负责接收 dataset 的输出，利用定义好的模块，计算输出。

我们的框架里 Model 表示整个模型，Module 表示一个模型里的多个小模块，例如 SAMP 的 MotionNet 是一整个模型，Encoder 和 Decoder 是这个模型小的模块。

需要在 model 的定义前使用 ```@MODELS.register_module()``` 来注册 module。

需要在 module 的定义前使用 ```@MODULES.register_module()``` 来注册 module。

### loss模块和criterion模块

loss 参考 ```motion/loss/loss.py```

criterion 参考 ```motion/criterion/regression.py```

loss 模块定义了我们所需要使用的 loss

criterion 模块定义了测试时如何计算 metric 或后处理。

### Hook模块

hook 参考 ```motion/hook/hook.py```

我们使用 hook 来实现了模型的存储，读取，测试和 log 的记录。

## References
这个代码参考了大量别的代码框架，包括但不限于：

[detectron2](https://github.com/facebookresearch/detectron2)

[mmdetection](https://github.com/open-mmlab/mmdetection)

[DI-engine](https://github.com/opendilab/DI-engine)

[SAMP_Training](https://github.com/mohamedhassanmus/SAMP_Training)
