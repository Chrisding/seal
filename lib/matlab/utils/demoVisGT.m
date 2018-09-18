%% Visualize SBD GTs
visualize_gt('../../../data/sbd-preprocess/data_proc', '../../../data/sbd-preprocess/data_proc/train.mat', './colors_sbd.mat', '../../../exper/sbd/result/vis/train')
visualize_gt('../../../data/sbd-preprocess/data_proc', '../../../data/sbd-preprocess/data_proc/val.mat', './colors_sbd.mat', '../../../exper/sbd/result/vis/val')
visualize_gt('../../../data/sbd-preprocess/data_proc', '../../../data/sbd-preprocess/data_proc/test.mat', './colors_sbd.mat', '../../../exper/sbd/result/vis/test')

%% Visualize Cityscapes GTs
visualize_gt('../../../data/cityscapes-preprocess/data_proc', '../../../data/cityscapes-preprocess/data_proc/train.mat', './colors_cityscapes.mat', '../../../exper/cityscapes/result/vis/train')
visualize_gt('../../../data/cityscapes-preprocess/data_proc', '../../../data/cityscapes-preprocess/data_proc/val.mat', './colors_cityscapes.mat', '../../../exper/cityscapes/result/vis/val')