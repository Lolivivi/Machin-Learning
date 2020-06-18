---


---

<h1 id="ch02-image-classification-图像分类">CH02 Image classification 图像分类</h1>
<h3 id="一、-图像分类、数据驱动方法和流程">一、 图像分类、数据驱动方法和流程</h3>
<ol>
<li><strong>图像分类</strong>
<ul>
<li><strong>目标：</strong> 所谓图像分类问题，就是已有固定的分类标签集合，然后对于输入的图像，从分类标签集合中找出一个分类标签，最后把分类标签分配给该输入图像。</li>
<li>计算机中提取到的是每个像素点的RGB数据。</li>
<li><strong>面对的困难：</strong> 视角变化、大小变化、形变、遮挡、光照条件、背景干扰和类内差异等会造成同一物体的不同表现。</li>
<li><strong>计算边缘：</strong> 针对上述问题提出一种解决方案，让计算机能够提取出输入的图像的边缘，但这种方案的缺陷在于不可推演。</li>
</ul>
</li>
<li><strong>数据驱动 Data-Driven Approach</strong><br>
比深度学习更广义。<br>
为了解决计算边缘不可推演的问题，我们通过模拟人类学习识别图像的方式来训练计算机，给计算机很多数据，然后实现学习算法，让计算机学习到每个类的外形。<br>
想要实现数据驱动，则需要较大的数据集的支持。<br>
训练 ——核心要素——&gt; 模型<br>
数据驱动的算法分为训练函数和预测函数
<ul>
<li><strong>训练函数：</strong> 记录训练数据</li>
<li><strong>预测函数：</strong> 输入新的图片源，检测关键特征，输出lable</li>
</ul>
</li>
<li><strong>Nearest Neighbor分类器（最临近算法）</strong><br>
<img src="https://cs231n.github.io/assets/nn.jpg" alt="图像分类数据集：CIFAR-10" title="图像分类数据集：CIFAR-10"><br>
<strong>左边</strong>：从CIFAR-10数据库来的样本图像。<br>
<strong>右边</strong>：第一列是测试图像，然后第一列的每个测试图像右边是使用Nearest Neighbor算法，根据像素差异，从训练集中选出的10张最类似的图片。<br>
比较图片的方法就是比较32x32x3的像素块。最简单的方法就是逐个像素比较，最后将差异值全部加起来。即将两张图片先转化为两个向量 I1 和 I2 ，然后计算他们的<strong>L1距离</strong> (向量差求和)。<br>
如果两张图片一模一样，那么L1距离为0，但是如果两张图片很是不同，那L1值将会非常大。<br>
<strong>Q&amp;A</strong>
<ul>
<li>训练中有N个实例，那么训练与测试的速度有多快？<br>
训练的速度会非常快，因为训练时连续的，仅需要通过指针保存数据；而测试则需要耗费大量时间，因为测试样例需要与N个实例进行比较。</li>
<li>实际应用中最邻近算法表现如何？<br>
<img src="https://cs231n.github.io/assets/knn.jpeg" alt="knn"><br>
并不准确，会产生误差（由于噪声、失真等产生的像素点数据偏差或错误）</li>
</ul>
</li>
<li><strong>k-Nearest Neighbor分类器（K-近邻算法）</strong><br>
针对上述KNN算法的不足，与其只找最相近的那1个图片的标签，我们找最相似的k个图片的标签，然后让他们针对测试图片进行投票，最后把票数最高的标签作为对测试图片的预测。所以当k=1的时候，k-Nearest Neighbor分类器就是Nearest Neighbor分类器。从直观感受上就可以看到，更高的k值可以让分类的效果更平滑，使得分类器对于异常值更有抵抗力。</li>
<li><strong>距离选择</strong><br>
不同的距离度量的选择会对底层拓扑结构的预测产生影响。<br>
<strong>L1距离：</strong> 取决于坐标系的选取，对目标任务有特定意义时采用<br>
<strong>L2距离：</strong> 不受坐标系的影响，更加自然</li>
</ol>
<h3 id="二、验证集、交叉验证集和超参数调优">二、验证集、交叉验证集和超参数调优</h3>

