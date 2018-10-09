% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% This script is used to:
% 1. Serve as the top-level function for model deploy
% 2. Set test directories and list
%
% Input arguments:
% data_dir: root directory of source data
% list_dir: source data list
% model_load: directory of model to be loaded
% result_dir: directory to put test results
% gpu_id: which gpu to use
% --------------------------------------------------------

function deploy(data_dir, list_dir, model_load, result_dir, gpu_id)
addpath('../../caffe/matlab');
path = genpath('../../lib/matlab');
addpath(path);

%% Input sanity check
assert(ischar(data_dir) && exist(data_dir, 'file')==7,...
	'Wrong input argument: data_dir must be a valid source data directory.');
assert(ischar(list_dir) && exist(list_dir, 'file')==2,...
	'Wrong input argument: list_dir must be a valid path of source data list.');
assert(ischar(model_load) && exist(model_load, 'file')==2 &&...
    strcmp(model_load(max(strfind(model_load, '.'))+1:end), 'caffemodel'),...
    'Wrong input argument: model_load must be a valid path with caffemodel extension.');
assert(isreal(gpu_id) && (gpu_id==round(gpu_id)),...
    'Wrong input argument: gpu id must be an integer.');

%% Pre-set parameters
param.num_cls = 19; % Important! Need to be the same with GT!
param.mean = [122.675 116.669 104.008];
param.crop_size_test = 632;
param.colormap = '../../lib/matlab/utils/dataset/colors_city.mat';

%% Main body
deploy_dir = './config/deploy.prototxt';
if(isempty(result_dir))
	result_dir = ['./result/deploy/' list_dir(max(strfind(list_dir, '/'))+1:max(strfind(list_dir, '.')-1))...
				  '/' model_load(max(strfind(model_load, '/'))+1:max(strfind(model_load, '.'))-1)];
end
net = net_init(gpu_id, deploy_dir, model_load);
do_test(net, param, data_dir, list_dir, result_dir);

%% Clear everything
caffe.reset_all();
delete(gcp('nocreate'));
exit;
end