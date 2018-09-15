clc; clear; close all;
path = genpath('./function');
addpath(path);

delete(gcp('nocreate'));
parpool('local', 36);

%% Set directories
% config = 'res_case_train_reweight_iter_26000';
config = 'res_case_train_uniweight_iter_26000';
% config = 'res_case_train_uniweight_sx1_sy3_mrf0.04_iter_26000';
root_pred = ['./result/test_phase/pred_val/' config];
vis_dir = ['./result/visualization/pred_color/val/' config];
mkdir(vis_dir);

%% Load inputs
list_folder = dir(root_pred);
num_cls = size(list_folder, 1)-2;
list_pred = dir([root_pred '/' list_folder(3).name '/*.png']);
load('./function/colors.mat');
colors = 255-colors;

%% Main loop
num_pred = size(list_pred, 1);
parfor idx_pred = 1:num_pred
    fileName = list_pred(idx_pred).name;
    disp(['Visualizing file id: ' num2str(idx_pred) ' file name: ' fileName])
    pred = imread([root_pred '/' list_folder(3).name '/' fileName]);
    [height, width, chn] = size(pred);
    bdry_vis = zeros(height*width, 3);
    bdry_sum = zeros(height*width, 1);
    bdry_max = zeros(height*width, 1);
    for idx_cls = 1:num_cls
        bdry_cls = reshape(double(imread([root_pred '/' list_folder(idx_cls+2).name '/' fileName]))./255, [height*width, 1]);
        bdry_vis = bdry_vis + double(bdry_cls)*(colors(idx_cls,:)./255);
        bdry_sum = bdry_sum + bdry_cls(:);
        bdry_max = max(bdry_max, bdry_cls);
    end
    idx_bdry = bdry_sum > 0;
    bdry_vis(idx_bdry, :) = bdry_vis(idx_bdry, :)./repmat(bdry_sum(idx_bdry), [1, 3]);
    bdry_vis = bdry_vis.*repmat(bdry_max, [1 3]);
    bdry_vis = 1 - bdry_vis;
    bdry_vis = reshape(bdry_vis, [height, width, 3]);
    imwrite(bdry_vis, [vis_dir '/' fileName], 'png');
end