# FCN Implementation with Pytorch


## 0. Develop Environment


## 1. Explain about Implementation


## 2. Brief Summary of *'Fully Convolutional Networks for Semantic Segmentation'*

### 2.1. Goal
- Make network with end-to-end, pixels-to-pixels for semantic segmentation

### 2.2. Intuition
- Convolutionalization : classification network can get arbitrary image input
- Deconvolution : learnable upsampling
- Skip : fuse local information and global information

### 2.3. Evaluation Metric
- Mean pixel intersection over union with the mean taken over all classes, including background

### 2.4. Network Architecture
![Architecture](./Figures/Architecture.png)

- Initialization
  * final deconvolution layers : fixed to bilinear interpolation
  * intermediate upsampling layer : bilinear interpolation
  * class scoring convolution layer : zero-initialized
  * FCN-16s : FCN-32s

### 2.5. Train and Inference on PASCAL VOC 2011
- Objective : per-pixel multinomial logistic loss (no class balancing)
- Train Details
  * minibatch SGD with momentum
    * batch size : 20
    * learning rate : 0.0001
    * momentum : 0.9
    * weight decay : 0.0002 or 0.0005


## 3. Reference Paper
- 2014 Fully Convolutional Networks for Semantic Segmentation [[paper]](https://arxiv.org/pdf/1411.4038.pdf)
- 2016 Fully Convolutional Networks for Semantic Segmentation [[paper]](https://arxiv.org/pdf/1605.06211.pdf)
