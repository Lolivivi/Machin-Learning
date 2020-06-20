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
<p>卷积神经网络只关注图像，循环神经网络只关注语言。</p>
<ol>
<li>
<p><strong>用于超参数调优的验证集</strong><br>
在机器学习的上下文中，超参数是在开始学习过程之前设置值的参数。<br>
<strong>决不能使用测试集来进行调优</strong><br>
如果使用测试集调优，算法实际部署后，性能可能会远低于预期，即出现算法对测试集<strong>过拟合</strong>。最终测试的时候再使用测试集，可以很好地近似度量你所设计的分类器的泛化性能。</p>
<blockquote>
<p>测试数据集只使用一次，即在训练完成后评价最终的模型时使用。</p>
</blockquote>
<p>因此为了不使用测试集进行调优，我们从训练集中抽取出一部分作为验证集，用于超参数调优，从而确定最优的超参数，运行在测试集上，对算法进行评价。</p>
<blockquote>
<p>把训练集分成训练集和验证集。使用验证集来对所有超参数调优。最后只在测试集上跑一次并报告结果。</p>
</blockquote>
</li>
<li>
<p><strong>交叉验证</strong><br>
有时候，训练集数量较小（因此验证集的数量更小），则采用<strong>交叉验证</strong>的方法，这种方法更加复杂些。将训练集平均分成n份，其中n-1份用来训练，1份用来验证。然后我们循环着取其中n-1份来训练，其中1份来验证，最后取所有n次验证结果的平均值作为算法验证结果。</p>
</li>
<li>
<p><strong>实际应用</strong><br>
在实际情况下，人们不是很喜欢用交叉验证，主要是因为它会耗费较多的计算资源。一般直接把训练集按照50%-90%的比例分成训练集和验证集。但这也是根据具体情况来定的：如果超参数数量多，需要更大的验证集，而验证集的数量不够，则采用交叉验证吧。至于分成几份比较好，一般都是分成3、5和10份。<br>
<img src="https://cs231n.github.io/assets/crossval.jpeg" alt="crossval"><br>
上图示例常用的数据分割模式。给出训练集和测试集后，训练集一般会被均分。这里是分成5份。前面4份用来训练，黄色那份用作验证集调优。如果采取交叉验证，那就各份轮流作为验证集。最后模型训练完毕，超参数都定好了，让模型跑一次（而且只跑一次）测试集，以此测试结果评价算法。</p>
</li>
<li>
<p><strong>Nearest Neighbor分类器的优劣</strong></p>
<ul>
<li>易于理解，实现简单</li>
<li>算法的训练不需要花时间，但测试要花费大量时间计算<br>
Nearest Neighbor分类器在某些特定情况（比如数据维度较低）下，可能是不错的选择。但是在实际的图像分类工作中，很少使用。因为图像都是高维度数据（他们通常包含很多像素），而高维度向量之间的距离通常是反直觉的。下面的图片展示了基于像素的相似和基于感官的相似是有很大不同的：<br>
<img src="https://cs231n.github.io/assets/samenorm.png" alt="samenorm"><br>
在高维度数据上，基于像素的的距离和感官上的非常不同。<br>
再举一个例子，使用t-SNE的可视化技术，将CIFAR-10中的图片按照二维方式排布，这样能很好展示图片之间的像素差异值。在这张图片中，排列相邻的图片L2距离就小，但可以明显看出，图片的排列是被背景主导而不是图片语义内容本身主导。<br>
<img src="https://cs231n.github.io/assets/pixels_embed_cifar10.jpg" alt="pixels_embed_cifar10"></li>
</ul>
</li>
</ol>
<hr>
<h3 id="summary">Summary</h3>
<ul>
<li>介绍了<strong>图像分类</strong>问题。在该问题中，给出一个由被标注了分类标签的图像组成的集合，要求算法能预测没有标签的图像的分类标签，并根据算法预测准确率进行评价。</li>
<li>介绍了一个简单的图像分类器：<strong>最近邻分类器(Nearest Neighbor classifier)</strong>。分类器中存在不同的超参数(比如k值或距离类型的选取)，要想选取好的超参数不是一件轻而易举的事。</li>
<li>选取超参数的正确方法是：将原始训练集分为训练集和<strong>验证集</strong>，我们在验证集上尝试不同的超参数，最后保留表现最好那个。</li>
<li>如果训练数据量不够，使用<strong>交叉验证</strong>方法，它能帮助我们在选取最优超参数的时候减少噪音。</li>
<li>一旦找到最优的超参数，就让算法以该参数在测试集跑且只跑一次，并根据测试结果评价算法。</li>
<li>最近邻分类器能够在CIFAR-10上得到将近40%的准确率。该算法简单易实现，但需要存储所有训练数据，并且在测试的时候过于耗费计算能力。</li>
<li>最后，我们知道了仅仅使用L1和L2范数来进行像素比较是不够的，图像更多的是按照背景和颜色被分类，而不是语义主体分身。</li>
</ul>
<p><strong>实际应用k-NN</strong><br>
如果你希望将k-NN分类器用到实处（最好别用到图像上，若是仅仅作为练手还可以接受），那么可以按照以下流程：</p>
<ol>
<li>预处理你的数据：对你数据中的特征进行归一化（normalize），让其具有零平均值（zero mean）和单位方差（unit variance）。在后面的小节我们会讨论这些细节。本小节不讨论，是因为图像中的像素都是同质的，不会表现出较大的差异分布，也就不需要标准化处理了。</li>
<li>如果数据是高维数据，考虑使用降维方法，比如PCA(wiki ref,  CS229ref,  blog ref)或随机投影。</li>
<li>将数据随机分入训练集和验证集。按照一般规律，70%-90% 数据作为训练集。这个比例根据算法中有多少超参数，以及这些超参数对于算法的预期影响来决定。如果需要预测的超参数很多，那么就应该使用更大的验证集来有效地估计它们。如果担心验证集数量不够，那么就尝试交叉验证方法。如果计算资源足够，使用交叉验证总是更加安全的（份数越多，效果越好，也更耗费计算资源）。</li>
<li>在验证集上调优，尝试足够多的k值，尝试L1和L2两种范数计算方式。</li>
<li>如果分类器跑得太慢，尝试使用Approximate Nearest Neighbor库（比如FLANN）来加速这个过程，其代价是降低一些准确率。</li>
<li>对最优的超参数做记录。记录最优参数后，是否应该让使用最优参数的算法在完整的训练集上运行并再次训练呢？因为如果把验证集重新放回到训练集中（自然训练集的数据量就又变大了），有可能最优参数又会有所变化。在实践中，<strong>不要这样做</strong>。千万不要在最终的分类器中使用验证集数据，这样做会破坏对于最优参数的估计。<strong>直接使用测试集来测试用最优参数设置好的最优模型</strong>，得到测试集数据的分类准确率，并以此作为你的kNN分类器在该数据上的性能表现。</li>
</ol>
<hr>
<h3 id="further-reading">Further Reading</h3>
<p>Here are some (optional) links you may find interesting for further reading:</p>
<ul>
<li>
<p><a href="https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf">A Few Useful Things to Know about Machine Learning</a>, where especially section 6 is related but the whole paper is a warmly recommended reading.</p>
</li>
<li>
<p><a href="https://people.csail.mit.edu/torralba/shortCourseRLOC/index.html">Recognizing and Learning Object Categories</a>, a short course of object categorization at ICCV 2005.</p>
</li>
</ul>
<hr>
<p>参考课件 <a href="http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture02.pdf">cs231n_2018_lecture02</a></p>

