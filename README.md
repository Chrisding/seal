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
	 author={Yu, Zhiding and Feng, Chen and Liu, Ming-Yu and Ramalingam, Srikumar},
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
    ```


### Usage

Upon successfully compiling the SEAL Caffe distribution, you can run the following experiments.

#### Part 1: Preprocessing
**SBD Data:** In this part, we assume you are in the directory **`$SEAL_ROOT/data/sbd-preprocess/`**.
1. Download the SBD dataset (with both original and CRF-preprocessed SBD ground truths) from [Google Drive](https://drive.google.com/open?id=17UrG33UI6VdHe8d1nknMw4U-ZqFZKiFo), and place the tarball **`sbd.tar.gz`** in **`data_orig/`**. Run the following command:

	```Shell
	tar -xvzf data_orig/sbd.tar.gz -C data_orig && rm data_orig/sbd.tar.gz
	```
    
2. Perform data augmentation and generate training edge labels by running the following command:

	```Matlab
	# In Matlab Command Window
	run code/demoPreproc.m
	```
    This will create augmented images and their **instance-sensitive(inst)**/**non-instance-sensitive(cls)** edge labels for network training in **`data_proc/`**.

3. Generate edge ground truths for evaluation by running the following command:

	```Matlab
	# In Matlab Command Window
  	run code/demoGenGT.m
  	```
    This will create two folders (**`gt_orig_thin/`** and **`gt_orig_raw/`**) in the directory of **`gt_eval/`**, containing the thinned and unthinned evaluation ground truths from the original SBD data.

    We do not provide the code to compute evaluation ground truths from the re-annotated SBD test set. You can download the tarball containing the precomputed ground truths from [Google Drive](https://drive.google.com/open?id=1cOTz1wqOky2XQW1AMlLTjABRaD-53Q1S), and place the tarball **`gt_reanno.tar.gz`** in **`gt_eval/`**. Run the following command:

	```Shell
	tar -xvzf gt_eval/gt_reanno.tar.gz -C gt_eval && rm gt_eval/gt_reanno.tar.gz
	```

**Cityscapes Data:** In this part, we assume you are in the directory **`$SEAL_ROOT/data/cityscapes-preprocess/`**. Note that in this repository, all Cityscapes pipelines are **instance-sensitive** only.
1. Download the files **`gtFine_trainvaltest.zip`**, **`leftImg8bit_trainvaltest.zip`** and **`leftImg8bit_demoVideo.zip`** from the [Cityscapes website](https://www.cityscapes-dataset.com/) to **`data_orig/`**, and unzip them:

	```Shell
	unzip data_orig/gtFine_trainvaltest.zip -d data_orig && rm data_orig/gtFine_trainvaltest.zip
	unzip data_orig/leftImg8bit_trainvaltest.zip -d data_orig && rm data_orig/leftImg8bit_trainvaltest.zip
	unzip data_orig/leftImg8bit_demoVideo.zip -d data_orig && rm data_orig/leftImg8bit_demoVideo.zip
	```

2. Generate training edge labels by running the following command:

	```Matlab
	# In Matlab Command Window
	run code/demoPreproc.m
	```
	This will create instance-sensitive edge labels for network training in **`data_proc/`**.

3. Generate edge ground truths for evaluation by running the following command:

	```Matlab
	# In Matlab Command Window
  	run code/demoGenGT.m
  	```
    This will create two folders (**`gt_thin/`** and **`gt_raw/`**) in the directory of **`gt_eval/`**, containing the thinned and unthinned evaluation ground truths.

#### Part 2: Training
**Train on SBD:** In this part, we assume you are in the directory **`$SEAL_ROOT/exper/sbd/`**.

1. Download the init model (for CASENet) and warm-up init models (for CASENet-S/CASENet-C/SEAL) from [Google Drive](https://drive.google.com/open?id=10ZNGT3Sc6jdNJa6b2U9g_4A-9srAtN1i) and put the zip file **`model_init.zip`** in **`model/`**. Run the following command:

    ```Shell
    unzip model/model_init.zip -d model
    ```

    The models **`model_init_inst_warm`** and **`model_init_cls_warm`** are warm-up init models that are obtained by additionally training a few iterations from **`model_init`** with unweighted sigmoid cross-entropy loss and relatively small learning rates. It is used to initialize models that involve unweighted sigmoid cross-entropy loss and stabilize their early training, since the loss is harder to learn than the reweighted one. Alternatively, you can obtain your own warm-up models using the code.

2. To train the network models, call the **`solve`** function with the following input argument format:

    ```Shell
    solve(<data_root>, <file_list_path>, <init_model_path>, <snapshot_prefix>, <iter_num>, <lr>, <gpu_id>, <loss_type>, <sigma_x>, <sigma_y>, <lambda>)
    ```

    By choosing the last four input, one could train the models of SEAL and all baselines (CASENet, CASENet-S and CASENet-C) reported in the paper. For example, to train SEAL with instance-sensitive(**`IS`**)/non-instance-sensitive(**`non-IS`**) edge labels from the original SBD data, run the following commands:

    ```Shell
    matlab -nodisplay -r "solve('../../data/sbd-preprocess/data_proc', '../../data/sbd-preprocess/data_proc/trainvalaug_inst_orig.mat', './model/model_init_inst_warm.caffemodel', 'model_inst_seal', 22000, 5.0*10^-8, <gpu_id>, 'unweight', 1, 4, 0.02)" 2>&1 | tee ./log/seal_inst.txt
    matlab -nodisplay -r "solve('../../data/sbd-preprocess/data_proc', '../../data/sbd-preprocess/data_proc/trainvalaug_cls_orig.mat', './model/model_init_cls_warm.caffemodel', 'model_cls_seal', 22000, 5.0*10^-8, <gpu_id>, 'unweight', 1, 4, 0.02)" 2>&1 | tee ./log/seal_cls.txt
    ```
    
    This will output model snapshots in the **`model/`** folder and log files in the **`log/`** folder. You may leave **`snapshot_prefix`** as empty (**`[]`**), which will automatically give a prefix composed by file list name and param configs. We assume training on 12G memory GPUs (such as TitanX/XP) without any other occupation. If you happen to have smaller GPU memories, consider decreasing the default 472x472 training crop size in **`solve`**. The new size must be dividable by 8.

    You can also remove **`lambda`** or set it to 0 to train vanilla SEAL without Markov smoothness. If you set either/both **`sigma_x`** and **`sigma_y`** to 0, or simply remove them and **`lambda`**, the code will train without any alignment. For example, CASENet/CASENet-S/CASENet-C with IS labels can be obtained by running the following commands, respectively:

    ```Shell
    matlab -nodisplay -r "solve('../../data/sbd-preprocess/data_proc', '../../data/sbd-preprocess/data_proc/trainvalaug_inst_orig.mat', './model/model_init.caffemodel', 'model_inst_casenet', 22000, 1.0*10^-7, <gpu_id>, 'reweight')" 2>&1 | tee ./log/casenet_inst.txt
    matlab -nodisplay -r "solve('../../data/sbd-preprocess/data_proc', '../../data/sbd-preprocess/data_proc/trainvalaug_inst_orig.mat', './model/model_init_inst_warm.caffemodel', 'model_inst_casenet-s', 22000, 5.0*10^-8, <gpu_id>, 'unweight')" 2>&1 | tee ./log/casenet-s_inst.txt
    matlab -nodisplay -r "solve('../../data/sbd-preprocess/data_proc', '../../data/sbd-preprocess/data_proc/trainvalaug_inst_crf.mat', './model/model_init_inst_warm.caffemodel', 'model_inst_casenet-c', 22000, 5.0*10^-8, <gpu_id>, 'unweight')" 2>&1 | tee ./log/casenet-c_inst.txt
    ```

    The command to obtain CASENet/CASENet-S/CASENet-C with non-IS edge labels can be similarly derived, simply by changing all **`inst`** suffixes to **`cls`**. You can also download all pretrained models from [Google Drive](https://drive.google.com/open?id=16WNjpYVdXPzBNuU9zcVPQNogPm9rp5xd).

**Train on Cityscapes:** In this part, we assume you are in the directory **`$SEAL_ROOT/exper/cityscapes/`**.

1. Download the init model and warm-up init models from [Google Drive](https://drive.google.com/open?id=1vLfjeSeX2jiJZjAlI7yXg2JcdW-Gtf87) and put the zip file **`model_init.zip`** in **`model/`**. Run the following command:

    ```Shell
    unzip model/model_init.zip -d model
    ```

2. Train CASENet/CASENet-S/SEAL models by running the following commands, respectively:

    ```Shell
    matlab -nodisplay -r "solve('../../data/cityscapes-preprocess/data_proc', '../../data/cityscapes-preprocess/data_proc/train.mat', './model/model_init.caffemodel', 'model_casenet', 28000, 5.0*10^-8, <gpu_id>, 'reweight')" 2>&1 | tee ./log/model_casenet.txt
    matlab -nodisplay -r "solve('../../data/cityscapes-preprocess/data_proc', '../../data/cityscapes-preprocess/data_proc/train.mat', './model/model_init_warm.caffemodel', 'model_casenet-s', 28000, 2.5*10^-8, <gpu_id>, 'unweight')" 2>&1 | tee ./log/model_casenet-s.txt
    matlab -nodisplay -r "solve('../../data/cityscapes-preprocess/data_proc', '../../data/cityscapes-preprocess/data_proc/train.mat', './model/model_init_warm.caffemodel', 'model_seal', 28000, 2.5*10^-8, <gpu_id>, 'unweight', 1, 3, 0.02)" 2>&1 | tee ./log/model_seal.txt
    ```

    You can download all pretrained models from [Google Drive](https://drive.google.com/open?id=1v14HXltyr3ajxd9gFSMFkJGIibn1PFXb).

    ```Shell
    matlab -nodisplay -r "solve('../../data/cityscapes-preprocess/data_proc', '../../data/cityscapes-preprocess/data_proc/train.mat', './model/model_init.caffemodel', 'model_casenet', 28000, 5.0*10^-8, <gpu_id>, 'reweight')" 2>&1 | tee ./log/model_casenet.txt
    matlab -nodisplay -r "solve('../../data/cityscapes-preprocess/data_proc', '../../data/cityscapes-preprocess/data_proc/train.mat', './model/model_init_warm.caffemodel', 'model_casenet-s', 28000, 2.5*10^-8, <gpu_id>, 'unweight')" 2>&1 | tee ./log/model_casenet-s.txt
    matlab -nodisplay -r "solve('../../data/cityscapes-preprocess/data_proc', '../../data/cityscapes-preprocess/data_proc/train.mat', './model/model_init_warm.caffemodel', 'model_seal', 28000, 2.5*10^-8, <gpu_id>, 'unweight', 1, 3, 0.02)" 2>&1 | tee ./log/model_seal.txt
    ```

#### Part 3: Testing
**Test on SBD:** In this part, we assume you are in the directory **`$SEAL_ROOT/exper/sbd/`**. To test the network models, call the **`deploy`** function with the following input argument format:
Try Try Try

	```Shell
	matlab -nodisplay -r "solve('../../data/cityscapes-preprocess/data_proc', '../../data/cityscapes-preprocess/data_proc/train.mat', './model/model_init.caffemodel', 'model_casenet', 28000, 5.0*10^-8, <gpu_id>, 'reweight')" 2>&1 | tee ./log/model_casenet.txt
	matlab -nodisplay -r "solve('../../data/cityscapes-preprocess/data_proc', '../../data/cityscapes-preprocess/data_proc/train.mat', './model/model_init_warm.caffemodel', 'model_casenet-s', 28000, 2.5*10^-8, <gpu_id>, 'unweight')" 2>&1 | tee ./log/model_casenet-s.txt
	matlab -nodisplay -r "solve('../../data/cityscapes-preprocess/data_proc', '../../data/cityscapes-preprocess/data_proc/train.mat', './model/model_init_warm.caffemodel', 'model_seal', 28000, 2.5*10^-8, <gpu_id>, 'unweight', 1, 3, 0.02)" 2>&1 | tee ./log/model_seal.txt
	```

Try Try Try

    ```Shell
    solve(<data_root>, <file_list_path>, <model_path>, <result_directory>, <gpu_id>)
    ```

### Video Demo
[![SEAL Demo](https://img.youtube.com/vi/gpy20uGnlY4/hq3.jpg)](https://www.youtube.com/watch?v=gpy20uGnlY4)


### Contact

  [Zhiding Yu](https://chrisding.github.io/)

  Questions can also be left as issues in the repository. We will be happy to answer them.

