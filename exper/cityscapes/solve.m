% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% This script is used to:
% 1. Serve as the top-level function for model training
% 2. Set hyperparameters and directories
%
% Input arguments:
% data_dir: root directory of source data
% list_dir: source data list
% model_init: init model directory
% config: string to encode param config and model/result names
% iter_num: maximum number of iterations
% base_lr: initial learning rate
% gpu_id: which gpu to use
% varargin: {loss_type, (sigma_x), (sigma_y), (lambda)}
% --------------------------------------------------------

function solve(data_dir, list_dir, model_init, config, iter_num, base_lr, gpu_id, varargin)
%% Add library paths
addpath('../../caffe/matlab');
path = genpath('../../lib/matlab');
addpath(path);

%% Input sanity check
assert(ischar(data_dir) && exist(data_dir, 'file')==7,...
	'Wrong input argument: data_dir must be a valid source data directory.');
assert(ischar(list_dir) && exist(list_dir, 'file')==2,...
	'Wrong input argument: list_dir must be a valid path of source data list.');
assert(ischar(model_init) && exist(model_init, 'file')==2 &&...
    (strcmp(model_init(max(strfind(model_init, '.'))+1:end), 'caffemodel') ||...
     strcmp(model_init(max(strfind(model_init, '.'))+1:end), 'solverstate')),...
    'Wrong input argument: model_init must be a valid path with caffemodel/solverstate extension.');
assert((ischar(config)||isempty(config)),...
    'Wrong input argument: config must be either empty or a string.');
assert(isreal(iter_num) && (iter_num==round(iter_num)) && (iter_num>0),...
    'Wrong input argument: iter_number must be a positive integer.');
assert(isreal(base_lr) && (base_lr>0),...
    'Wrong input argument: base learning rate must be a real postive number.');
assert(isreal(gpu_id) && (gpu_id==round(gpu_id)),...
    'Wrong input argument: gpu id must be an integer.');

%% Set hyperparameters
% Set training hyperparameters
param.num_cls = 19; % Important! Need to be the same with GT!
param.mean = [122.675 116.669 104.008];
param.crop_size_train = 472;
param.mirror = true;
param.iter_num = iter_num; % Maximum number of iterations
if(strcmp(model_init(max(strfind(model_init, '.'))+1:end), 'caffemodel'))
    param.resume = false;
else
    param.resume = true;
end
% Parse and set input seal hyperparameters
param = arg_parser(param, varargin); % Determine loss_type, sigma_x, sigma_y, lambda
param.neigh_size = 16; % Neighborhood size for Markov smoothness prior
param.max_spatial_cost = 2; % Controls the search range when doing alignment
param.par_size = 400; % Number of images to solve assignment in parallel
param.vis_align = true;
% Set sovler hyperparameters
if(isempty(config))
    set_name = list_dir(max(strfind(list_dir, '/'))+1:end-4);
	config = param2config(set_name, param);
end
if(strcmp(param.loss_type, 'unweight'))
    param.solver.train_net = '"config/model_unweight.prototxt"';
else
    param.solver.train_net = '"config/model_reweight.prototxt"';
end
param.solver.lr_policy = '"step"';
param.solver.base_lr = base_lr;
param.solver.gamma = 0.2;
param.solver.iter_size = 10;
param.solver.stepsize = 20000;
param.solver.average_loss = 20;
param.solver.display = 1;
param.solver.momentum = 0.9;
param.solver.weight_decay = 0.0005;
param.solver.snapshot = 1000;
param.solver.snapshot_prefix = ['"model/' config '"'];
param.solver.solver_mode = 'GPU';

%% Setup Parallel Pool
num_worker = 12; % Number of matlab workers for parallel computing
matlabVer = version('-release');
if( str2double(matlabVer(1:4)) > 2013 || (str2double(matlabVer(1:4)) == 2013 && strcmp(matlabVer(5), 'b')) )
    delete(gcp('nocreate'));
    parpool('local', num_worker);
else
    if(matlabpool('size')>0) %#ok<*DPOOL>
        matlabpool close
    end
    matlabpool open 8
end

%% Main program
solver_dir = ['./config/solver_' config '.prototxt'];
gen_solver(solver_dir, param);
solver = solver_init(gpu_id, solver_dir, model_init, param.resume);
result_dir = ['./result/solve/' config];
if(param.resume)
    s = load([model_init(1:max(strfind(model_init, '.'))-1) '.mat']);
    param.state = s.state;
else
    net_surgery(solver);
end
do_train(solver, param, data_dir, list_dir, result_dir);

%% Clear everything
caffe.reset_all();
delete(gcp('nocreate'));
exit;
end