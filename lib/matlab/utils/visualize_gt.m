clc; clear; close all;
path = genpath('./function');
addpath(path);

delete(gcp('nocreate'));
parpool('local', 36);

%% Set directories
root_gt = '/home/datasets/cityscapes_aug/edge/label_inst/data_gen';
vis_dir = './result/visualization/gt_color/val';
mkdir(vis_dir);

%% Load inputs
s = load([root_gt '/val_edgeCat.mat']);
names = fieldnames(s);
list_gt = s.(names{1});
load('./function/colors.mat');

%% Main loop
num_gt = size(list_gt, 1);
for idx_gt = 1:num_gt
    pos = max(strfind(list_gt{idx_gt, 2}, '/'));
    fileName = list_gt{idx_gt, 2}(pos+1:end-4);
    fileNameOut = [fileName(1:end-15) '_leftImg8bit.png'];
    disp(['Visualizing file id: ' num2str(idx_gt) ' file name: ' fileName])
    s = load([root_gt list_gt{idx_gt, 2}]);
    names = fieldnames(s);
    gt = s.(names{1});
    img = imread([root_gt list_gt{idx_gt, 1}]);
    [height, width, chn] = size(img);
    img_vis = double(reshape(img, [height*width, chn]))./255;
    bdry_vis = zeros(height*width, 3);
    bdry_sum = zeros(height*width, 1);
    num_cls = size(gt, 1);
    for idx_cls = 1:num_cls
        bdry_cls = reshape(full(gt{idx_cls, 1}), [height*width, 1]);
        bdry_vis = bdry_vis + double(bdry_cls)*(colors(idx_cls,:)./255);
        bdry_sum = bdry_sum + bdry_cls(:);
    end
    idx_bdry = bdry_sum > 0;
    bdry_vis(idx_bdry, :) = bdry_vis(idx_bdry, :)./repmat(bdry_sum(idx_bdry), [1, 3]);
    bdry_vis(~idx_bdry, :) = 1;
    % bdry_vis(~idx_bdry, :) = img_vis(~idx_bdry, :);
    bdry_vis = reshape(bdry_vis, [height, width, 3]);
    imwrite(bdry_vis, [vis_dir '/' fileNameOut], 'png');
end