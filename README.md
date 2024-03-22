# MuseNet: 基于CNN和注意力机制的音腔识别系统

## 数据准备

## 训练

修改 `configs/musenet_train.yaml`，之后执行

```shell
python train.py
```

## 推理

修改 `configs/musenet_inference.yaml`，之后执行

```shell
python inference.py -a 音频路径 [-g]
```

* `-g` ： 是否使用 CUDA 推理，如果是则指定该参数，否则不指定

推理结果以 `json` 文件的形式保存，保存在 `result/` 目录下

