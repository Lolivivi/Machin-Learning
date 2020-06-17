---


---

<h1 id="introduction-to-cnn-for-visual-recognition">Introduction to CNN for visual recognition</h1>
<h2 id="ch01-cnn简介">CH01 CNN简介</h2>
<h3 id="计算机视觉从20世纪60年代末到2017年的简史">1. 计算机视觉从20世纪60年代末到2017年的简史</h3>
<h3 id="计算机视觉问题包括图像分类、目标定位、目标检测和场景理解">2. 计算机视觉问题包括图像分类、目标定位、目标检测和场景理解</h3>
<ul>
<li>图像分类的常见场景，例如微博发布图片时的不同推送标签、bilibili的视频分区，通过计算机识别图像中的内容，并将其分类至所属的类型当中。</li>
<li>目标定位，识别图像中的指定目标。</li>
<li>目标检测
<ol>
<li>目标分割，确定图像中每个像素点的位置，对像素点进行规范化削减 (池化)</li>
<li>基于特征的目标识别，根据对象的有表现性和不变性的关键特征进行识别</li>
<li>识别整幅图</li>
</ol>
</li>
</ul>
<h3 id="imagenet是目前图像分类领域最大的数据集之一">3. Imagenet是目前图像分类领域最大的数据集之一</h3>
<p>计算机视觉要能够实现上述功能需要极大的数据集支持，通过数据集中的数据训练才能使计算机视觉的性能能够更大程度上接近人类，甚至超越人类。<br>
如果数据集过小则会出现过拟合化的问题。<br>
从2012年Imagenet的比赛开始，CNN(卷积神经网络)一直是赢家。<br>
CNN实际上是在1997年由Yann Lecun发明的。</p>
