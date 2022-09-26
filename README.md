# 基于生成对抗网络的 AI 人脸卡通化实现

## 1.简述

本代码在已有研究的基础上（[SystemErrorWang/White-box-Cartoonization](https://github.com/SystemErrorWang/White-box-Cartoonization)），以生成对抗网络（GAN）为核心技术构建人脸卡通化模型，并进一步引入了人脸关键点的辅助定位方式，由此优化了将人脸图像转化为卡通人像的风格变换技术。

建模的主要思路为：选取非成对数据进行训练，模型的主体为两组对抗生成网络，分别用于迁移卡通图像的质地和表层特征。通过调整损失函数的权重和使用不同风格的卡通数据集，该模型能够生成各类风格和样式的卡通人脸像。

## 2.如何使用代码

本仓库依据上述研究的建模思路，将该论文的Tensorflow1.0工程框架重新改写为Pytorch框架，并进行了若干细节优化。

### 1）获取数据集

将卡通数据集和人脸数据集放置于`datasets/{yourdatasets}`，并修改`pretrain.py`、`train.py`、`test.py`代码中读取数据集的路径。

### 2）模型预训练

运行`pretrain.py`文件，即可生成预训练模型，该模型存放位置为`./checkpoints/saved_models/pre_gen_batch_{iters}.pth`。

```sh
python pretrain.py --total_iter 50000
```

参数`total_iter`用于调节预训练epoch数。

### 3）训练模型

```sh
python train.py --total_iter 30000 --model_version 1 --w0 1e5 --w1 2e-1 --w2 5 --w3 2e1 --w4 6e-1
```

参数`total_iter`调节训练epoch数，`model_version`为模型编号（用于区分不同模型），其余参数用于调节模型各部分loss权重（详见下文模型简介）。

### 4）测试模型精度

```sh
python test.py --model_version 49 --batch 500
```

`model_version`为模型编号，`batch`为测试模型所需的batch数。运行代码后会输出卡通画前后的人脸图像，用于进行对比。

## 3.模型简介

模型主体是由一个生成器和两个判别器构成的对抗生成网络， 其中生成器 *G* 的作用接收原始的人脸照，并通过生成器内部的网络结构，产出卡通化的人脸照。在训练过程中，生成器 *G* 产出的图像随即经过四种图像处理方法，生成四幅特征图，分别展现生成 图像的结构（Structure）、质地（Texture）、表层（Surface）和关键点（Landmark）。

![image-20220926131544686](github.com/ArthurinRUC/FacialCartoonization/tree/main/image/model2)

![image-20220926131645696](github.com/ArthurinRUC/FacialCartoonization/tree/main/image/model1)

其中生成器（及其子结构）和判别器架构如下图所示。

![image-20220926131729998](/Users/arthur/Library/Application Support/typora-user-images/image-20220926131729998.png)

![image-20220926131735204](/Users/arthur/Library/Application Support/typora-user-images/image-20220926131735204.png)

#### 损失函数

除了我们关心的生成图像的四类特征之外，我们还希望生成图像和原图像之间的内容差异不至于太大，并且内容上的约束也不应该阻碍卡通画特征的学习。因此我们采用预训练的VGG19网络前四层（最后一层去除了激活函数和池化函数），直接提取原始图像和生成图像的高维特征，并采用$l_1$范数进行约束。因此，我们构建的内容损失函数为：

$$
\begin{equation}
\mathcal{L}_{content}(G,D) = \mathbb{E}_{x_i \sim S_{data}(x)}  ||VGG_{19}(G(x_i)) - VGG_{19}(x_i)||
\end{equation}
$$
此外，为去除生成图像的椒盐噪声，我们还约束了生成图像的相邻像素值之差，使得相邻像素点具有一定的连续性，避免出现异常像素点：
$$
\begin{equation}
\mathcal{L}_{variance}(G,D) = \mathbb{E}_{x_i \sim S_{data}(x)}  \dfrac{\nabla_x G(x_i) + \nabla_y G(x_i)}{C * H * W}
\end{equation}
$$
模型整体的损失函数由以上六种损失函数的加权平均组成。
$$
\begin{equation}
\mathcal{L} = \lambda_1 \cdot \mathcal{L}_{content} +\lambda_2 \cdot \mathcal{L}_{variance} +\lambda_3 \cdot \mathcal{L}_{landmark} +\lambda_4 \cdot \mathcal{L}_{structure} +\lambda_5 \cdot \mathcal{L}_{texture} +\lambda_6 \cdot \mathcal{L}_{surface}
\end{equation}
$$
通过调节不同部分损失函数的权重，我们可以灵活控制生成卡通人脸像特征。在实验部分，我们将调整部分权重并进行消融实验，进一步展现各类特征是如何体现在图像上的。

#### 实验设计

训练模型需要两类数据集：人脸数据集和卡通图像数据集。

* 本文的人脸数据集来源于FFHQ高清人脸数据集（Flickr-Faces-High-Quality）。FFHQ是一个高质量的人脸数据集，包含$1024 \times 1024$分辨率的70000张PNG格式高清人脸图像，本文使用了其中的5000张人脸图片，并将分辨率统一转化为 $128 \times 128$。
* 本文的卡通数据集来源于AnimeGAN论文提供的卡通图像，其中包含Hayao、Paprika、Shinkai、SummerWar等多种风格的卡通片图像，每类图片约2000张。

#### 模型结果

![image-20220926132308501](/Users/arthur/Library/Application Support/typora-user-images/image-20220926132308501.png)

![image-20220926132315851](/Users/arthur/Library/Application Support/typora-user-images/image-20220926132315851.png)

![image-20220926132323790](/Users/arthur/Library/Application Support/typora-user-images/image-20220926132323790.png)

## 4.参考文献

[1]      B. Amberg and T. Vetter, “Optimal landmark detection using shape models and branch and bound,” in *2011 International Conference on Computer Vision*, IEEE, 2011, pp. 455–462.

[2]      P. N. Belhumeur, D. W. Jacobs, D. J. Kriegman, and N. Kumar, “Localizing parts of faces using a consensus of exemplars,” *IEEE transactions on pattern analysis and machine intelligence*, vol. 35, no. 12, pp. 2930–2940, 2013.

[3]      J. Chen, G. Liu, and X. Chen, “Animegan: A novel lightweight gan for photo animation,” in *Interna- tional Symposium on Intelligence Computation and Applications*, Springer, 2019, pp. 242–256.

[4]      Y. Chen, Y.-K. Lai, and Y.-J. Liu, “Cartoongan: Generative adversarial networks for photo cartooniza- tion,” in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2018, pp. 9465–9474.

[5]      A. Deshpande, J. Lu, M.-C. Yeh, M. Jin Chong, and D. Forsyth, “Learning diverse image coloriza- tion,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2017, pp. 6837–6845.

[6]      L. A. Gatys, A. S. Ecker, and M. Bethge, “Image style transfer using convolutional neural networks,” in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, pp. 2414– 2423.

[7]      X. Glorot and Y. Bengio, “Understanding the difﬁculty of training deep feedforward neural net- works,” in *Proceedings of the thirteenth international conference on artiﬁcial intelligence and statis- tics*, JMLR Workshop and Conference Proceedings, 2010, pp. 249–256.

[8]      R. Gomez-Ojeda, Z. Zhang, J. Gonzalez-Jimenez, and D. Scaramuzza, “Learning-based image en- hancement for visual odometry in challenging hdr environments,” in *2018 IEEE International Con- ference on Robotics and Automation (ICRA)*, IEEE, 2018, pp. 805–811.

[9]      B. Gooch, G. Coombe, and P. Shirley, “Artistic vision: Painterly rendering using computer vision techniques,” in *Proceedings of the 2nd international symposium on Non-photorealistic animation and rendering*, 2002, 83–ff.

[10]      B. Gooch, E. Reinhard, and A. Gooch, “Human facial illustrations: Creation and psychophysical evaluation,” *ACM Transactions on Graphics (TOG)*, vol. 23, no. 1, pp. 27–44, 2004.

[11]      I. Goodfellow, J. Pouget-Abadie, M. Mirza, *et al.*, “Generative adversarial nets,” *Advances in neural information processing systems*, vol. 27, 2014.

[12]      K. He, J. Sun, and X. Tang, “Guided image ﬁltering,” *IEEE transactions on pattern analysis and machine intelligence*, vol. 35, no. 6, pp. 1397–1409, 2012.

[13]      A. Hertzmann, “Painterly rendering with curved brush strokes of multiple sizes,” in *Proceedings of the 25th annual conference on Computer graphics and interactive techniques*, 1998, pp. 453–460.

[14]      X. Huang and S. Belongie, “Arbitrary style transfer in real-time with adaptive instance normaliza- tion,” in *Proceedings of the IEEE international conference on computer vision*, 2017, pp. 1501–1510.

[15]      A. Ignatov, N. Kobyshev, R. Timofte, K. Vanhoey, and L. Van Gool, “Wespe: Weakly supervised photo enhancer for digital cameras,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*, 2018, pp. 691–700.

[16]      P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2017, pp. 1125–1134.

[17]      Y. Jing, Y. Yang, Z. Feng, J. Ye, Y. Yu, and M. Song, “Neural style transfer: A review,” *IEEE trans-* *actions on visualization and computer graphics*, vol. 26, no. 11, pp. 3365–3385, 2019.

[18]      J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in *European conference on computer vision*, Springer, 2016, pp. 694–711.

[19]      A. Kolliopoulos, *Image segmentation for stylized non-photorealistic rendering and animation*. Cite- seer, 2005.

[20]      L. Liang, R. Xiao, F. Wen, and J. Sun, “Face alignment via component-based discriminative search,” in *European conference on computer vision*, Springer, 2008, pp. 72–85.

[21]      T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, “Spectral normalization for generative adversar- ial networks,” *arXiv preprint arXiv:1802.05957*, 2018.

[22]      M. Rashid, X. Gu, and Y. Jae Lee, “Interspecies knowledge transfer for facial keypoint detection,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2017, pp. 6894– 6903.

[23]      Y. Shang, K. Miao, B. Chen, and Z. Wen, “Transfer photo to anime with dual discriminators gan,” in *2022* *2nd* *International* *Conference on Consumer Electronics and Computer Engineering (ICCECE)*, IEEE, 2022, pp. 265–270.

[24]      Y.-Z. Song, P. L. Rosin, P. M. Hall, and J. P. Collomosse, “Arty shapes.,” in *CAe*, Citeseer, 2008, pp. 65–72.

[25]      Y. Sun, X. Wang, and X. Tang, “Deep convolutional network cascade for facial point detection,” in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2013, pp. 3476– 3483.

[26]      C. Tomasi and R. Manduchi, “Bilateral ﬁltering for gray and color images,” in *Sixth international conference on computer vision (IEEE Cat. No. 98CH36271)*, IEEE, 1998, pp. 839–846.

[27]      D. Ulyanov, V. Lebedev, A. Vedaldi, and V. S. Lempitsky, “Texture networks: Feed-forward synthe- sis of textures and stylized images.,” in *ICML*, vol. 1, 2016, p. 4.

[28]      X. Wang and J. Yu, “Learning to cartoonize using white-box cartoon representations,” in *Proceedings* *of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2020, pp. 8090–8099.

[29]      H. Wu, S. Zheng, J. Zhang, and K. Huang, “Fast end-to-end trainable guided ﬁlter,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2018, pp. 1838–1847.

[30]      R. Wu, X. Gu, X. Tao, X. Shen, Y.-W. Tai, *et al.*, “Landmark assisted cyclegan for cartoon face generation,” *arXiv preprint arXiv:1907.01424*, 2019.

[31]      H. Yin, A. Mallya, A. Vahdat, J. M. Alvarez, J. Kautz, and P. Molchanov, “See through gradients: Image batch recovery via gradinversion,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2021, pp. 16 337–16 346.

[32]      J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle- consistent adversarial networks,” in *Proceedings of the IEEE international conference on computer vision*, 2017, pp. 2223–2232.

[33]      X. Zhu and D. Ramanan, “Face detection, pose estimation, and landmark localization in the wild,” in

*2012 IEEE conference on computer vision and pattern recognition*, IEEE, 2012, pp. 2879–2886.
