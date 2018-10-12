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
0. [Video Demo](#video-demo)
0. [Contact](#contact)


### Introduction

The repository contains the entire pipeline (including data preprocessing, training, testing, evaluation, visualization, and demo generation, etc) for **`SEAL`**.

SEAL is a recently proposed learning framework towards edge learning under noisy labels. The framework seeks to directly generate high quality thin/crisp object semantic boundaries without any post-processing, by jointly performing edge alignment with edge learning. In particular, edge alignment is formulated as latent variable optimization and learned end-to-end during network training. For more details, please refer to the [arXiv technical report](https://arxiv.org/abs/1808.01992) and the [ECCV18 paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhiding_Yu_SEAL_A_Framework_ECCV_2018_paper.pdf). We highly recommend the readers refer to [arXiv](https://arxiv.org/abs/1808.01992) for latest updates in detailed description and experiments.

We use CASENet as the backbone network for SEAL since it is the state-of-the-art deep network for category-aware semantic edge detection. CASENet adopts a modified ResNet-101 architecture with dilated convolution. More details about CASENet can be found in the [paper](https://arxiv.org/abs/1705.09759) and the [prototxt file](https://github.com/Chrisding/seal/blob/master/exper/cityscapes/config/model_reweight.prototxt). Note that this code has been designed to fully support training/testing of CASENet, simply by changing a few input parameters. The original implementation of CASENet is available [here](http://www.merl.com/research/?research=license-request&sw=CASENet).

SEAL currently achieves the state-of-the-art category-aware semantic edge detection performance on the [Semantic Boundaries Dataset](http://home.bharathh.info/pubs/codes/SBD/download.html) and the [Cityscapes Dataset](https://www.cityscapes-dataset.com/).


### Citation

If you find **SEAL** useful in your research, please consider to cite the following papers:

	@inproceedings{yu2018seal, 
	  title={Simultaneous Edge Alignment and Learning}, 
	  author={Yu, Zhiding and Liu, Weiyang and Zou, Yang and Feng, Chen and Ramalingam, Srikumar and Vijaya Kumar, BVK and Kautz, Jan},
	  booktitle={European Conference on Computer Vision (ECCV)}, 
	  year={2018}
	}

	@inproceedings{yu2017casenet,
	 title={uppercase{CASEN}et: Deep Category-Aware Semantic Edge Detection},
	 author={Z. Yu and C. Feng and M. Y. Liu and S. Ramalingam},
	 booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	 year={2017}
	}


### Requirements
1. Requirements for `Matlab`
2. Requirements for `Caffe` and `matcaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html) and [SEAL Caffe distribution readme](https://github.com/Chrisding/seal/tree/master/caffe))


### Installation
1. Clone the SEAL repository. We'll call the directory that you cloned SEAL as **`SEAL_ROOT`**.

    ```Shell
    git clone --recursive https://github.com/Chrisding/seal.git
    ```

2. Build Caffe and matcaffe

    ```Shell
    cd $SEAL_ROOT/caffe
    # Follow the Caffe installation instructions to install all required packages:
    # http://caffe.berkeleyvision.org/installation.html
    # Follow the instructions to install matio:
    # https://sourceforge.net/projects/matio/files/matio/1.5.2/
    make all -j8 && make matcaffe


### Usage

Upon successfully compiling the SEAL Caffe distribution, you can run the following experiments.

#### Part 1: Preprocessing
**SBD Data:** In this part, we assume you are in the directory **`$SEAL_ROOT/data/sbd-preprocess/`**
1. Download the SBD dataset (with both original and CRF-preprocessed SBD ground truths) from https://drive.google.com/open?id=17UrG33UI6VdHe8d1nknMw4U-ZqFZKiFo, and place the tarball "sbd.tar.gz" in **`data_orig/`**. Run the following command:

	```Shell
	tar -xvzf data_orig/sbd.tar.gz -C data_orig && rm data_orig/sbd.tar.gz
	```
    
2. Perform data augmentation and generate training edge labels by running the following code:

	```Matlab
	# In Matlab Command Window
	run code/demoPreproc.m
	```
    This will create augmented images and instance-sensitive(inst)/non-instance-sensitive(cls) edge labels for network training in **`data_proc/`**.

3. Generate edge ground truths for evaluation.

	```Matlab
	# In Matlab Command Window
  	run code/demoGenGT.m
  	```
    This will create two folders (**`gt_orig_thin/`** and **`gt_orig_raw/`**) in the directory of **`gt_eval/`**, containing the thinned and unthinned evaluation ground truths from the original SBD data.

    We do not provide the code to compute evaluation ground truths from the re-annotated SBD test set. You can download the tarball containing the precomputed ground truths from https://drive.google.com/open?id=1cOTz1wqOky2XQW1AMlLTjABRaD-53Q1S, and place the tarball "gt_reanno.tar.gz" in **`gt_eval/`**. Run the following command:

	```Shell
	tar -xvzf gt_eval/gt_reanno.tar.gz -C gt_eval && rm gt_eval/gt_reanno.tar.gz
	```

**Cityscapes Data:** In this part, we assume you are in the directory **`$SEAL_ROOT/data/cityscapes-preprocess/`**
1. Download the files "gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip" and "leftImg8bit_demoVideo.zip" from the [Cityscapes website](https://www.cityscapes-dataset.com/) to **`data_orig/`**, and unzip them:

	```Shell
	unzip data_orig/gtFine_trainvaltest.zip -d data_orig && rm data_orig/gtFine_trainvaltest.zip
	unzip data_orig/leftImg8bit_trainvaltest.zip -d data_orig && rm data_orig/leftImg8bit_trainvaltest.zip
	unzip data_orig/leftImg8bit_demoVideo.zip -d data_orig && rm data_orig/leftImg8bit_demoVideo.zip
	```

2. Generate training edge labels by running the following code:

	```Matlab
	# In Matlab Command Window
	run code/demoPreproc.m
	```
	This will create instance-sensitive(inst)/non-instance-sensitive(cls) edge labels for network training in **`data_proc/`**.

3. Generate edge ground truths for evaluation.

	```Matlab
	# In Matlab Command Window
  	run code/demoGenGT.m
  	```
    This will create two folders (**`gt_thin/`** and **`gt_raw/`**) in the directory of **`gt_eval/`**, containing the thinned and unthinned evaluation ground truths.

#### Part 2: Train
**Train on SBD:** In this part, we assume you are in the directory **`$SEAL_ROOT/exper/sbd/`**



### Video Demo
[![SEAL Demo](https://img.youtube.com/vi/gpy20uGnlY4/hq3.jpg)](https://www.youtube.com/watch?v=gpy20uGnlY4)


### Contact

  [Zhiding Yu](https://chrisding.github.io/)

  Questions can also be left as issues in the repository. We will be happy to answer them.

