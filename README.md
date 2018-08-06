# ChinesePlateRecognition
### 项目介绍：
此项目是在faster rcnn的基础上修改而来，主要有如下两处修改：
<li>fast rcnn网络中bbox回归的全连接层不变，而分类的全连接层改为conv1-128和 conv1-32</li>
<li>fast rcnn网络中的分类层改成lstm结构，并且使用ctc作为其loss函数</li>

### 项目运行步骤:
1, 在setup脚本中设置和gpu匹配的结构
```
cd ChinesePlateRecognition/lib
vim setup.py
```
2, 编译Cpython模块
```
make clean
make
cd ..
```
3, 训练模型
```
./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
# 例如:
./experiments/scripts/train_faster_rcnn_ctc.sh 0 standard_car_plate vgg16
```

更多深度学习、机器学习、统计学习的内容可以观看我的博客
[段逍遥的博客](https://blog.csdn.net/u011070767)
