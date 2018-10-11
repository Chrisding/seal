## Simultaneous Edge Alignment and Learning (SEAL)

By Zhiding Yu, Weiyang Liu, Yang Zou, Chen Feng, Srikumar Ramalingam, B. V. K. Vijayakumar and Jan Kautz

### License

SEAL is released under the MIT License (refer to the LICENSE file for details).

### Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Installation](#installation)
0. [Usage](#usage)
0. [Models](#models)
0. [Results](#results)
0. [Video Demo](#video-demo)
0. [Note](#note)


### Introduction

The repository contains the entire pipeline (including data preprocessing, training, testing, evaluation, visualization, and demo generation, etc) for **`SEAL`**.

SEAL is a recently proposed learning framework towards edge learning under noisy labels. The framework seeks to directly generate high quality thin/crisp object semantic boundaries without any post-processing, by jointly performing edge alignment with edge learning. In particular, edge alignment is formulated as latent variable optimization and learned end-to-end during network training. For more details, please refer to the [arXiv technical report](https://arxiv.org/abs/1808.01992) and the [ECCV18 paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhiding_Yu_SEAL_A_Framework_ECCV_2018_paper.pdf). We highly recommend the readers refer to [arXiv](https://arxiv.org/abs/1808.01992) for latest updates in detailed description and experiments.

We use CASENet as the backbone network for SEAL since it is the state-of-the-art deep network for category-aware semantic edge detection. CASENet adopts a modified ResNet-101 architecture with dilated convolution. More details about CASENet can be found in the [paper](https://arxiv.org/abs/1705.09759) and the [prototxt file](https://github.com/Chrisding/seal/blob/master/exper/cityscapes/config/model_reweight.prototxt).

SEAL currently achieves the state-of-the-art category-aware semantic edge detection performance on the [Semantic Boundaries Dataset](http://home.bharathh.info/pubs/codes/SBD/download.html) and the [Cityscapes Dataset](https://www.cityscapes-dataset.com/).


### Citation

If you find **SEAL** useful in your research, please consider to cite:

	@inproceedings{yu2018seal, 
	  title={Simultaneous Edge Alignment and Learning}, 
	  author={Yu, Zhiding and Liu, Weiyang and Zou, Yang and Feng, Chen and Ramalingam, Srikumar and Vijaya Kumar, BVK and Kautz, Jan},
	  booktitle={European Conference on Computer Vision (ECCV)}, 
	  year={2018}
	}

Our another closely-related previous work in ICML'16 ([more](https://github.com/wy1iu/LargeMargin_Softmax_Loss)):

	@inproceedings{yu2017casenet,
	 title={uppercase{CASEN}et: Deep Category-Aware Semantic Edge Detection},
	 author={Z. Yu and C. Feng and M. Y. Liu and S. Ramalingam},
	 booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	 year={2017}
	}


### Requirements
1. Requirements for `Matlab`
2. Requirements for `Caffe` and `matcaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html) and [SEAL Caffe distribution readme](https://github.com/Chrisding/seal/tree/master/caffe))


### Video Demo
[![SEAL Demo](https://img.youtube.com/vi/gpy20uGnlY4/maxres3.jpg)](https://www.youtube.com/watch?v=gpy20uGnlY4)


### Contact

  [Zhiding Yu](https://chrisding.github.io/)

  Questions can also be left as issues in the repository. We will be happy to answer them.

