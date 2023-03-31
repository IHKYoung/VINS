# VINS 升级手记

> 这一部分是基于VINS-Mono进行的；在Mono的基础上，再会考虑加入RGBD、Stereo的支持。
>
> 主要涉及的升级：
>
> 1. 代码的工程管理：使用include、src对头文件和源文件作区分
> 2. ROS的包管理：使用一个包（vins）管理多个功能节点node
> 3. **代码的逐行重写（重构）+注解，同时涉及一些库的升级和改写**

## 补充知识点

### 单位球误差

单位球误差（unit sphere error）是用于评估相机位姿估计精度的一种指标。它基于相机的重投影误差（reprojection error），并考虑了相机的旋转和平移部分之间的耦合关系。

具体来说，假设有 $N$ 个三维空间点 $\mathbf{X}_i=(X_i,Y_i,Z_i)^T$ 和对应的图像点 $\mathbf{x}_i=(u_i,v_i)^T$，以及相机的位姿 $\mathbf{T}=(\mathbf{R}|\mathbf{t})$，其中 $\mathbf{R}$ 是旋转矩阵，$\mathbf{t}$ 是平移向量。对于每个点 $\mathbf{X}_i$，其在相机坐标系下的投影 $\mathbf{x}_i'=\mathbf{K}\mathbf{T}\mathbf{X}_i$，其中 $\mathbf{K}$ 是相机内参数矩阵。投影误差为其真实图像坐标 $\mathbf{x}_i$ 与其投影坐标 $\mathbf{x}_i'$ 之间的欧氏距离：
$$
e_i = \Vert x_i -x_i^\prime \Vert
$$
将投影误差 $e_i$ 除以相机焦距 $f$，得到其在归一化平面上的单位化误差 $e_i' = e_i / f$。对于所有的 $N$ 个点，定义单位球误差为它们在归一化平面上单位化误差的均值：
$$
unit\ sphere\ error = \frac{1}{N}\sum_{i=1}^{N}e_i^\prime
$$

单位球误差越小，说明相机位姿估计的精度越高。它是相机位姿估计算法中常用的性能指标之一。

