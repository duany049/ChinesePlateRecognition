# ChinesePlateRecognition
### 项目介绍：
此项目是在faster rcnn的基础上修改而来，主要有如下两处修改：
<li>fast rcnn网络中bbox回归的全连接层不变，而分类的全连接层改为conv1-128和 conv1-32</li>
<li>fast rcnn网络中的分类层改成lstm结构，并且使用ctc作为其loss函数</li>

