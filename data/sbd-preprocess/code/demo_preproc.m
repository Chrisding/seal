% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to:
% 1. Perform data augmentation on the SBD dataset with multiple scales
% 2. Generate instance-sensitive multi-label semantic edges on the augmented data
% 3. Create filelists for the augmented data and labels
% --------------------------------------------------------

function demo_preproc()

clc; clear; close all;

%% Add library paths
path = genpath('../../../lib/matlab');
addpath(path);

%% Setup Directories
dataRoot = '../data_orig';
genDataRoot = '../data_proc';

%% Setup Parameters
scaleSet = [0.5 0.75 1 1.25 1.5];
numScale = length(scaleSet);
numCls = 20;
radius = 2;
edgeType = 'regular';
numVal = 1000; % param not effective when flagSeed is true & seed exists
flagSeed = true;
flagBinFile = false;

%% Setup Parallel Pool
numWorker = 12; % Number of matlab workers for parallel computing
matlabVer = version('-release');
if( str2double(matlabVer(1:4)) > 2013 || (str2double(matlabVer(1:4)) == 2013 && strcmp(matlabVer(5), 'b')) )
    delete(gcp('nocreate'));
    parpool('local', numWorker);
else
    if(matlabpool('size')>0) %#ok<*DPOOL>
        matlabpool close
    end
    matlabpool open 8
end

%% Generate Preprocessed Dataset
setList = {'val'};
for setID = 1:length(setList)
    setName = setList{1, setID};
    % Create output directories

    if(strcmp(setName, 'train'))
        for scale = scaleSet
            % Train
            if(exist([genDataRoot '/image/train/scale_' num2str(scale)], 'file')==0)
                mkdir([genDataRoot '/image/train/scale_' num2str(scale)]);
            end
            if(exist([genDataRoot '/label/train/scale_' num2str(scale)], 'file')==0)
                mkdir([genDataRoot '/label/train/scale_' num2str(scale)]);
            end
            % Validation
            if(exist([genDataRoot '/image/val/scale_' num2str(scale)], 'file')==0)
                mkdir([genDataRoot '/image/val/scale_' num2str(scale)]);
            end
            if(exist([genDataRoot '/label/val/scale_' num2str(scale)], 'file')==0)
                mkdir([genDataRoot '/label/val/scale_' num2str(scale)]);
            end
        end
    else
        % Test
        if(exist([genDataRoot '/image/test'], 'file')==0)
            mkdir([genDataRoot '/image/test']);
        end
        if(exist([genDataRoot '/label/test'], 'file')==0)
            mkdir([genDataRoot '/label/test']);
        end
    end
    
    fidIn = fopen([dataRoot '/' setName '.txt']);
    fileName = fgetl(fidIn);
    fileList = cell(1,1);
    countFile = 0;
    while ischar(fileName)
        countFile = countFile + 1;
        fileList{countFile} = fileName;
        fileName = fgetl(fidIn);
    end
    fclose(fidIn);
    
    if(strcmp(setName, 'train'))
        if(exist('./seed.mat', 'file') && flagSeed)
            s = load('./seed.mat');
            fields = fieldnames(s);
            valSet = s.(fields{1});
        else
            valSet = false(countFile, 1);
            idxRand = randperm(countFile);
            valSet(idxRand(1:numVal), 1) = true;
            save('./seed.mat', 'valSet');
        end
    end

    % Compute boundaries and write generated data
    disp(['Computing ' setName ' set boundaries'])
    parfor_progress(countFile);
    parfor idxFile = 1:countFile
        fileName = fileList{idxFile};
        if(strcmp(setName, 'train'))
            scaleSetRun = scaleSet;
        else
            scaleSetRun = 1;
        end
        for idx_scale = 1:length(scaleSetRun)
            scale = scaleSetRun(idx_scale);
            s_inst = load([dataRoot '/inst/' fileList{idxFile} '.mat']);
            seg_inst = s_inst.GTinst.Segmentation;
            seg_inst = imresize(seg_inst, scale, 'nearest');
            set_cls = unique(s_inst.GTinst.Categories)';
            [height, width] = size(seg_inst);
            
            labelEdge = cell(numCls, 1);
            for cls = 1:numCls
                if(ismember(cls, set_cls))
                    set_inst = find(s_inst.GTinst.Categories == cls)';
                    seg_map_inst = zeros(height, width);
                    for inst = set_inst
                        seg_map_inst(seg_inst==inst) = inst;
                    end
                    labelEdge{cls, 1} = sparse(seg2edge(seg_map_inst, radius, [], edgeType));
                else
                    labelEdge{cls, 1} = sparse(false(height, width));
                end
            end
            labelEdge2 = zeros(height, width, 'uint32');
            if(flagBinFile)
                for idx_cls = 1:numCls
                    idxEdge = full(labelEdge{idx_cls});
                    if(sum(idxEdge(:))~=0)
                        labelEdge2(idxEdge) = labelEdge2(idxEdge) + 2^(idx_cls-1);
                    end
                end
            end
            
            img = imread([dataRoot '/img/' fileName '.jpg']);
            imgScale = imresize(img, scale, 'bicubic');
            if(strcmp(setName, 'train'))
                if(~valSet(idxFile, 1))
                    % Write image file
                    imwrite(imgScale, [genDataRoot '/image/train/scale_' num2str(scale) '/' fileName '.png'], 'png')
                    % Write label file
                    savelabeledge([genDataRoot '/label/train/scale_' num2str(scale) '/' fileName '.mat'], labelEdge);
                    % Write label file
                    if(flagBinFile)
                        fidLabel = fopen([genDataRoot '/label/train/scale_' num2str(scale) '/' fileName '.bin'], 'w');
                        fwrite(fidLabel, labelEdge2', 'uint32'); % Important! Transpose input matrix to become row major.
                        fclose(fidLabel);
                    end
                else
                    % Write image file
                    imwrite(imgScale, [genDataRoot '/image/val/scale_' num2str(scale) '/' fileName '.png'], 'png')
                    % Write label file
                    savelabeledge([genDataRoot '/label/val/scale_' num2str(scale) '/' fileName '.mat'], labelEdge);
                    % Write label file
                    if(flagBinFile)
                        fidLabel = fopen([genDataRoot '/label/val/scale_' num2str(scale) '/' fileName '.bin'], 'w');
                        fwrite(fidLabel, labelEdge2', 'uint32'); % Important! Transpose input matrix to become row major.
                        fclose(fidLabel);
                    end
                end
            else
                % Write image file
                imwrite(imgScale, [genDataRoot '/image/test/' fileName '.png'], 'png')
                % Write label file
                savelabeledge([genDataRoot '/label/test/' fileName '.mat'], labelEdge);
                % Write label file
                if(flagBinFile)
                    fidLabel = fopen([genDataRoot '/label/test/' fileName '.bin'], 'w');
                    fwrite(fidLabel, labelEdge2', 'uint32'); % Important! Transpose input matrix to become row major.
                    fclose(fidLabel);
                end
            end
        end
        parfor_progress();
    end
    parfor_progress(0);
    
    % Create file lists
    disp(['Creating ' setName ' set file lists'])
    if(strcmp(setName, 'train'))
        if(flagBinFile)
            fidListTrainValAug = fopen([genDataRoot '/trainval_aug.txt'], 'w');
            fidListTrainVal = fopen([genDataRoot '/trainval.txt'], 'w');
            fidListTrainAug = fopen([genDataRoot '/train_aug.txt'], 'w');
            fidListTrain = fopen([genDataRoot '/train.txt'], 'w');
            fidListValAug = fopen([genDataRoot '/val_aug.txt'], 'w');
            fidListVal = fopen([genDataRoot '/val.txt'], 'w');
        end
        listTrainValAug = cell(countFile*numScale, 2);
        listTrainVal = cell(countFile, 2);
        listTrainAug = cell((countFile-numVal)*numScale, 2);
        listTrain = cell(countFile-numVal, 2);
        listValAug = cell(numVal*numScale, 2);
        listVal = cell(numVal, 2);
        countTrainValAug = 0;
        countTrainVal = 0;
        countTrainAug = 0;
        countTrain = 0;
        countValAug = 0;
        countVal = 0;
    else
        if(flagBinFile)
            fidListTest = fopen([genDataRoot '/test.txt'], 'w');
        end
        listTest = cell(countFile, 2);
        countTest = 0;
    end
    parfor_progress(countFile);
    for idxFile = 1:countFile
        fileName = fileList{idxFile};
        if(strcmp(setName, 'train'))
            scaleSetRun = scaleSet;
        else
            scaleSetRun = 1;
        end
        for idx_scale = 1:length(scaleSetRun)
            scale = scaleSetRun(idx_scale);
            if(strcmp(setName, 'train'))
                if(~valSet(idxFile, 1))
                    % Add to train_aug and trainval_aug
                    countTrainAug = countTrainAug + 1;
                    listTrainAug{countTrainAug, 1} = ['/image/train/scale_' num2str(scale) '/' fileName '.png'];
                    listTrainAug{countTrainAug, 2} = ['/label/train/scale_' num2str(scale) '/' fileName '.mat'];
                    countTrainValAug = countTrainValAug + 1;
                    listTrainValAug{countTrainValAug, 1} = ['/image/train/scale_' num2str(scale) '/' fileName '.png'];
                    listTrainValAug{countTrainValAug, 2} = ['/label/train/scale_' num2str(scale) '/' fileName '.mat'];
                    if(scale == 1)
                        % Add to train and trainval
                        countTrain = countTrain + 1;
                        listTrain{countTrain, 1} = ['/image/train/scale_' num2str(scale) '/' fileName '.png'];
                        listTrain{countTrain, 2} = ['/label/train/scale_' num2str(scale) '/' fileName '.mat'];
                        countTrainVal = countTrainVal + 1;
                        listTrainVal{countTrainVal, 1} = ['/image/train/scale_' num2str(scale) '/' fileName '.png'];
                        listTrainVal{countTrainVal, 2} = ['/label/train/scale_' num2str(scale) '/' fileName '.mat'];
                    end
                else
                    % Add to val_aug and trainval_aug
                    countValAug = countValAug + 1;
                    listValAug{countValAug, 1} = ['/image/val/scale_' num2str(scale) '/' fileName '.png'];
                    listValAug{countValAug, 2} = ['/label/val/scale_' num2str(scale) '/' fileName '.mat'];
                    countTrainValAug = countTrainValAug + 1;
                    listTrainValAug{countTrainValAug, 1} = ['/image/val/scale_' num2str(scale) '/' fileName '.png'];
                    listTrainValAug{countTrainValAug, 2} = ['/label/val/scale_' num2str(scale) '/' fileName '.mat'];
                    if(scale == 1)
                        % Add to val and trainval
                        countVal = countVal + 1;
                        listVal{countVal, 1} = ['/image/val/scale_' num2str(scale) '/' fileName '.png'];
                        listVal{countVal, 2} = ['/label/val/scale_' num2str(scale) '/' fileName '.mat'];
                        countTrainVal = countTrainVal + 1;
                        listTrainVal{countTrainVal, 1} = ['/image/val/scale_' num2str(scale) '/' fileName '.png'];
                        listTrainVal{countTrainVal, 2} = ['/label/val/scale_' num2str(scale) '/' fileName '.mat'];
                    end
                end
            else
                % Add to test
                countTest = countTest + 1;
                listTest{countTest, 1} = ['/image/test/' fileName '.png'];
                listTest{countTest, 2} = ['/label/test/' fileName '.mat'];
            end

            if(flagBinFile)
                if(strcmp(setName, 'train'))
                    if(~valSet(idxFile, 1))
                        % Add to train_aug and trainval_aug
                        fprintf(fidListTrainAug, ['/image/train/scale_' num2str(scale) '/' fileName '.png /label/train/scale_' num2str(scale) '/' fileName '.bin\n']);
                        fprintf(fidListTrainValAug, ['/image/train/scale_' num2str(scale) '/' fileName '.png /label/train/scale_' num2str(scale) '/' fileName '.bin\n']);
                        if(scale == 1)
                            % Add to train and trainval
                            fprintf(fidListTrain, ['/image/train/scale_' num2str(scale) '/' fileName '.png /label/train/scale_' num2str(scale) '/' fileName '.bin\n']);
                            fprintf(fidListTrainVal, ['/image/train/scale_' num2str(scale) '/' fileName '.png /label/train/scale_' num2str(scale) '/' fileName '.bin\n']);
                        end
                    else
                        % Add to val_aug and trainval_aug
                        fprintf(fidListValAug, ['/image/val/scale_' num2str(scale) '/' fileName '.png /label/val/scale_' num2str(scale) '/' fileName '.bin\n']);
                        fprintf(fidListTrainValAug, ['/image/val/scale_' num2str(scale) '/' fileName '.png /label/val/scale_' num2str(scale) '/' fileName '.bin\n']);
                        if(scale == 1)
                            % Add to val and trainval
                            fprintf(fidListVal, ['/image/val/scale_' num2str(scale) '/' fileName '.png /label/val/scale_' num2str(scale) '/' fileName '.bin\n']);
                            fprintf(fidListTrainVal, ['/image/val/scale_' num2str(scale) '/' fileName '.png /label/val/scale_' num2str(scale) '/' fileName '.bin\n']);
                        end
                    end
                else
                    % Add to test
                    fprintf(fidListTest, ['/image/test/' fileName '.png /label/test/' fileName '.bin\n']);
                end
            end
        end
        parfor_progress();
    end
    parfor_progress(0);
    
    % Write file lists
    if(strcmp(setName, 'train'))
        save([genDataRoot '/trainval_aug.mat'], 'listTrainValAug');
        save([genDataRoot '/trainval.mat'], 'listTrainVal');
        save([genDataRoot '/train_aug.mat'], 'listTrainAug');
        save([genDataRoot '/train.mat'], 'listTrain');
        save([genDataRoot '/val_aug.mat'], 'listValAug');
        save([genDataRoot '/val.mat'], 'listVal');
        if(flagBinFile)
            fclose(fidListTrainValAug);
            fclose(fidListTrainVal);
            fclose(fidListTrainAug);
            fclose(fidListTrain);
            fclose(fidListValAug);
            fclose(fidListVal);
        end
    else
        save([genDataRoot '/test.mat'], 'listTest');
        if(flagBinFile)
            fclose(fidListTest);
        end
    end
end