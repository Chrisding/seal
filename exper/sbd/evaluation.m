% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% This script is used to:
% 1. Serve as the top-level function for result evaluation
% 2. Set evaluation parameters and directories
%
% Input arguments:
%   file_list   : Name list of the files to be evaluated
%   gt_dir      : Directory containing ground truths
%   eval_dir    : Directory containing predictions to be evaluated
%   result_dir  : Directory to put evaluation results
%   idx_cls     : Current category
%   [margin]    : Size of margin to be ignored
%   [nthresh]   : Number of points in PR curve
%   [MaxDist]   : Edge misalignment tolerance threshold
%   [thinpb]    : Option to apply morphological thinning on evaluated boundaries
% --------------------------------------------------------

function evaluation(file_list, gt_dir, eval_dir, result_dir, margin, nthresh, thinpb, maxDist)

if(nargin<5), margin = 0; end;
if(nargin<6), nthresh = 99; end;
if(nargin<7), thinpb = true; end;
if(nargin<8), maxDist = 0.0075; end;

%% Set directories
path = genpath('../../lib/matlab');
addpath(path);

%% Set parameters
num_cls = 20;

%% Setup Parallel Pool
delete(gcp('nocreate'));
parpool('local', 12);

%% Load the evaluation file list
s = load(file_list);
names = fieldnames(s);
list_eval = s.(names{1});

%% Perform evaluation
if(exist(result_dir, 'file')==0)
    mkdir(result_dir);
end
categories = categories_sbd();
for idx_cls = 1:num_cls
    fprintf('Benchmarking boundaries for category %d: %s\n', idx_cls, categories{idx_cls});
    result_cls = benchmark_category(list_eval, eval_dir, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist);
    save([result_dir '/class_' num2str(idx_cls) '.mat'], 'result_cls');
end

%% Summarize evaluation results
result_list = dir([result_dir '/class_*.mat']);
num_result = size(result_list, 1);
result_f = zeros(num_cls, 1);
for idx_result = 1:num_result
    result_name = result_list(idx_result, 1).name;
    idx = find(result_name=='_', 1, 'last');
    idx_cls = str2double(result_name(idx+1:end-4));
    load([result_dir '/class_' num2str(idx_cls)])
    result_f(idx_cls) = result_cls{2, 1}(4);
end
fprintf('====================== Summary MF-ODS ======================\n\n');
for idx_cls = 1:num_cls
    fprintf('%2d %14s:  %.2f\n', idx_cls, categories{idx_cls}, result_f(idx_cls));
end
fprintf('\n      Mean MF-ODS:  %.2f\n', mean(result_f));