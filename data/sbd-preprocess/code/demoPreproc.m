% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to:
% 1. Perform data augmentation on original/crf-preprocessed SBD dataset with multiple scales
% 2. Generate instance-sensitive(IS)/non-IS multi-label semantic edges on the augmented data
% 3. Create filelists for the augmented data and labels
% --------------------------------------------------------

function demoPreproc()

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
numVal = 1000; % Param not effective when flagSeed is true & seed exists
flagSeed = true; % Use pre-stored seed to sample val set
flagOrig = true; % Flag to compute edge labels from orignal SBD data or not
flagCRF = true; % Flag to compute edge labels from CRF preprocessed SBD data or not (For CASENet-C)
flagInst = true; % Flag to compute instance-sensitive (IS) edge labels or not
flagCls = true; % Flag to compute non-instance-sensitive (non-IS) edge labels or not
flagBinFile = false; % Flag to dump .bin labels or not (For pycaffe CASENet/SEAL)

%% Sanity Check
if(flagOrig)
    if(flagInst)
        assert(exist([dataRoot '/inst_orig'], 'file')==7, 'Missing original SBD inst GTs!')
    end
    if(flagCls)
        assert(exist([dataRoot '/cls_orig'], 'file')==7, 'Missing original SBD cls GTs!')
    end
end
if(flagCRF)
    if(flagInst)
        assert(exist([dataRoot '/inst_crf'], 'file')==7, 'Missing CRF-preprocessed SBD inst GTs!')
    end
    if(flagCls)
        assert(exist([dataRoot '/cls_crf'], 'file')==7, 'Missing CRF-preprocessed SBD cls GTs!')
    end
end
flagProc = (flagOrig||flagCRF) && (flagInst||flagCls);
if(flagProc)
    assert(exist([dataRoot '/img'], 'file')==7, 'Missing original SBD images!')
end

%% Setup Parallel Pool
numWorker = 12; % Number of matlab workers for parallel computing
delete(gcp('nocreate'));
parpool('local', numWorker);

%% Generate Preprocessed Dataset
setList = {'train', 'val'};
srcList = {};
if(flagInst||flagCls)
    if(flagOrig)
        srcList = [srcList; 'orig'];
    end
    if(flagCRF)
        srcList = [srcList; 'crf'];
    end
end

for idxSrc = 1:length(srcList)
    srcName = srcList{idxSrc};
    if(strcmp(srcName, 'orig'))
        if(flagInst)
            if(flagBinFile)
                fidListTrainValTestInst = fopen([genDataRoot '/trainvaltest_inst_' srcName '.txt'], 'w');
            end
            listTrainValTestInst = cell(1, 2);
        end
        countTrainValTest = 0;
    end
    
    for setID = 1:length(setList)
        setName = setList{1, setID};
        if(strcmp(srcName, 'orig') || (strcmp(srcName, 'crf') && strcmp(setName, 'train')))
            % Create output directories
            if(strcmp(setName, 'train'))
                for scale = scaleSet
                    % Train
                    if(exist([genDataRoot '/image/train/scale_' num2str(scale)], 'file')==0)
                        mkdir([genDataRoot '/image/train/scale_' num2str(scale)]);
                    end
                    if(exist([genDataRoot '/image/val/scale_' num2str(scale)], 'file')==0)
                        mkdir([genDataRoot '/image/val/scale_' num2str(scale)]);
                    end
                    if(flagInst)
                        if(exist([genDataRoot '/label_inst_' srcName '/train/scale_' num2str(scale)], 'file')==0)
                            mkdir([genDataRoot '/label_inst_' srcName '/train/scale_' num2str(scale)]);
                        end
                        if(exist([genDataRoot '/label_inst_' srcName '/val/scale_' num2str(scale)], 'file')==0)
                            mkdir([genDataRoot '/label_inst_' srcName '/val/scale_' num2str(scale)]);
                        end
                    end
                    if(flagCls)
                        if(exist([genDataRoot '/label_cls_' srcName '/train/scale_' num2str(scale)], 'file')==0)
                            mkdir([genDataRoot '/label_cls_' srcName '/train/scale_' num2str(scale)]);
                        end
                        if(exist([genDataRoot '/label_cls_' srcName '/val/scale_' num2str(scale)], 'file')==0)
                            mkdir([genDataRoot '/label_cls_' srcName '/val/scale_' num2str(scale)]);
                        end
                    end
                end
            else
                % Test
                if(exist([genDataRoot '/image/test'], 'file')==0)
                    mkdir([genDataRoot '/image/test']);
                end
                if(flagInst)
                    if(exist([genDataRoot '/label_inst_' srcName '/test'], 'file')==0)
                        mkdir([genDataRoot '/label_inst_' srcName '/test']);
                    end
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
                    numVal = sum(valSet(1:countFile)); % Overwrite input numVal
                else
                    valSet = false(countFile, 1);
                    idxRand = randperm(countFile);
                    valSet(idxRand(1:numVal), 1) = true;
                    save('./seed.mat', 'valSet');
                end
            end
            
            % Compute boundaries and write generated data
            disp(['data src: ' srcName ', computing ' setName ' set boundaries'])
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
                    labelEdgeInst = [];
                    labelEdgeInst2 = [];
                    labelEdgeCls = [];
                    labelEdgeCls2 = [];
                    if(flagInst)
                        sInst = load([dataRoot '/inst_' srcName '/' fileList{idxFile} '.mat']);
                        segInst = sInst.GTinst.Segmentation;
                        segInst = imresize(segInst, scale, 'nearest');
                        setCls = unique(sInst.GTinst.Categories)';
                        [height, width] = size(segInst);
                        labelEdgeInst = cell(numCls, 1);
                        for idxCls = 1:numCls
                            if(ismember(idxCls, setCls))
                                set_inst = find(sInst.GTinst.Categories == idxCls)';
                                seg_map_inst = zeros(height, width);
                                for inst = set_inst
                                    seg_map_inst(segInst==inst) = inst;
                                end
                                labelEdgeInst{idxCls, 1} = sparse(seg2edge(seg_map_inst, radius, [], edgeType));
                            else
                                labelEdgeInst{idxCls, 1} = sparse(false(height, width));
                            end
                        end
                        labelEdgeInst2 = zeros(height, width, 'uint32');
                        if(flagBinFile)
                            for idxCls = 1:numCls
                                idxEdge = full(labelEdgeInst{idxCls});
                                if(sum(idxEdge(:))~=0)
                                    labelEdgeInst2(idxEdge) = labelEdgeInst2(idxEdge) + 2^(idxCls-1);
                                end
                            end
                        end
                    end
                    if(flagCls)
                        sCls = load([dataRoot '/cls_' srcName '/' fileName '.mat']);
                        segCls = sCls.GTcls.Segmentation;
                        segCls = imresize(segCls, scale, 'nearest');
                        [height, width] = size(segCls);
                        labelEdgeCls = cell(numCls, 1);
                        labelEdgeCls2 = zeros(height, width, 'uint32');
                        for idxCls = 1:numCls
                            idxSeg = segCls == idxCls;
                            if(sum(idxSeg(:))~=0)
                                idxEdge = seg2edge(idxSeg, radius, [], edgeType);
                                labelEdgeCls{idxCls, 1} = sparse(idxEdge)
                                labelEdgeCls2(idxEdge) = labelEdgeCls2(idxEdge) + 2^(idxCls-1);
                            else
                                labelEdgeCls{idxCls, 1} = sparse(false(height, width));
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
                            if(flagInst)
                                savelabeledge([genDataRoot '/label_inst_' srcName '/train/scale_' num2str(scale) '/' fileName '.mat'], labelEdgeInst);
                                if(flagBinFile)
                                    fidLabel = fopen([genDataRoot '/label_inst_' srcName '/train/scale_' num2str(scale) '/' fileName '.bin'], 'w');
                                    fwrite(fidLabel, labelEdgeInst2', 'uint32'); % Important! Transpose input matrix to become row major.
                                    fclose(fidLabel);
                                end
                            end
                            if(flagCls)
                                savelabeledge([genDataRoot '/label_cls_' srcName '/train/scale_' num2str(scale) '/' fileName '.mat'], labelEdgeCls);
                                if(flagBinFile)
                                    fidLabel = fopen([genDataRoot '/label_cls_' srcName '/train/scale_' num2str(scale) '/' fileName '.bin'], 'w');
                                    fwrite(fidLabel, labelEdgeCls2', 'uint32'); % Important! Transpose input matrix to become row major.
                                    fclose(fidLabel);
                                end
                            end
                        else
                            % Write image file
                            imwrite(imgScale, [genDataRoot '/image/val/scale_' num2str(scale) '/' fileName '.png'], 'png')
                            % Write label file
                            if(flagInst)
                                savelabeledge([genDataRoot '/label_inst_' srcName '/val/scale_' num2str(scale) '/' fileName '.mat'], labelEdgeInst);
                                if(flagBinFile)
                                    fidLabel = fopen([genDataRoot '/label_inst_' srcName '/val/scale_' num2str(scale) '/' fileName '.bin'], 'w');
                                    fwrite(fidLabel, labelEdgeInst2', 'uint32'); % Important! Transpose input matrix to become row major.
                                    fclose(fidLabel);
                                end
                            end
                            if(flagCls)
                                savelabeledge([genDataRoot '/label_cls_' srcName '/val/scale_' num2str(scale) '/' fileName '.mat'], labelEdgeCls);
                                if(flagBinFile)
                                    fidLabel = fopen([genDataRoot '/label_cls_' srcName '/val/scale_' num2str(scale) '/' fileName '.bin'], 'w');
                                    fwrite(fidLabel, labelEdgeCls2', 'uint32'); % Important! Transpose input matrix to become row major.
                                    fclose(fidLabel);
                                end
                            end
                        end
                    else
                        % Write image file
                        imwrite(imgScale, [genDataRoot '/image/test/' fileName '.png'], 'png')
                        % Write label file
                        if(flagInst)
                            savelabeledge([genDataRoot '/label_inst_' srcName '/test/' fileName '.mat'], labelEdgeInst);
                            if(flagBinFile)
                                fidLabel = fopen([genDataRoot '/label_inst_' srcName '/test/' fileName '.bin'], 'w');
                                fwrite(fidLabel, labelEdgeInst2', 'uint32'); % Important! Transpose input matrix to become row major.
                                fclose(fidLabel);
                            end
                        end
                    end
                end
                parfor_progress();
            end
            parfor_progress(0);
            
            % Create file lists
            disp(['data src: ' srcName ', creating ' setName ' set file lists'])
            if(strcmp(setName, 'train'))
                listVal = cell(numVal, 1);
                if(flagBinFile)
                    fidListVal = fopen([genDataRoot '/val.txt'], 'w');
                end
                if(flagInst)
                    if(flagBinFile)
                        fidListTrainValAugInst = fopen([genDataRoot '/trainvalaug_inst_' srcName '.txt'], 'w');
                        fidListTrainAugInst = fopen([genDataRoot '/trainaug_inst_' srcName '.txt'], 'w');
                    end
                    listTrainValAugInst = cell(countFile*numScale, 2);
                    listTrainAugInst = cell((countFile-numVal)*numScale, 2);
                end
                if(flagCls)
                    if(flagBinFile)
                        fidListTrainValAugCls = fopen([genDataRoot '/trainvalaug_cls_' srcName '.txt'], 'w');
                        fidListTrainAugCls = fopen([genDataRoot '/trainaug_cls_' srcName '.txt'], 'w');
                    end
                    listTrainValAugCls = cell(countFile*numScale, 2);
                    listTrainAugCls = cell((countFile-numVal)*numScale, 2);
                end
                countTrainValAug = 0;
                countTrainAug = 0;
                countVal = 0;
            else
                listTest = cell(countFile, 1);
                if(flagBinFile)
                    fidListTest = fopen([genDataRoot '/test.txt'], 'w');
                end
                if(flagInst)
                    if(flagBinFile)
                        fidListTestInst = fopen([genDataRoot '/test_inst_' srcList{idxSrc} '.txt'], 'w');
                    end
                    listTestInst = cell(countFile, 2);
                end
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
                            countTrainValAug = countTrainValAug + 1;
                            if(scale==1)
                                countTrainValTest = countTrainValTest + 1;
                            end
                            if(flagInst)
                                listTrainAugInst{countTrainAug, 1} = ['/image/train/scale_' num2str(scale) '/' fileName '.png'];
                                listTrainAugInst{countTrainAug, 2} = ['/label_inst_' srcName '/train/scale_' num2str(scale) '/' fileName '.mat'];
                                listTrainValAugInst{countTrainValAug, 1} = ['/image/train/scale_' num2str(scale) '/' fileName '.png'];
                                listTrainValAugInst{countTrainValAug, 2} = ['/label_inst_' srcName '/train/scale_' num2str(scale) '/' fileName '.mat'];
                                if(scale==1)
                                    listTrainValTestInst{countTrainValTest, 1} = ['/image/train/scale_' num2str(scale) '/' fileName '.png'];
                                    listTrainValTestInst{countTrainValTest, 2} = ['/label_inst_' srcName '/train/scale_' num2str(scale) '/' fileName '.mat'];
                                end
                            end
                            if(flagCls)
                                listTrainAugCls{countTrainAug, 1} = ['/image/train/scale_' num2str(scale) '/' fileName '.png'];
                                listTrainAugCls{countTrainAug, 2} = ['/label_cls_' srcName '/train/scale_' num2str(scale) '/' fileName '.mat'];
                                listTrainValAugCls{countTrainValAug, 1} = ['/image/train/scale_' num2str(scale) '/' fileName '.png'];
                                listTrainValAugCls{countTrainValAug, 2} = ['/label_cls_' srcName '/train/scale_' num2str(scale) '/' fileName '.mat'];
                            end
                        else
                            % Add to val and trainval_aug
                            countTrainValAug = countTrainValAug + 1;
                            if(scale==1)
                                countVal = countVal + 1;
                                countTrainValTest = countTrainValTest + 1;
                                listVal{countVal, 1} = ['/image/val/scale_' num2str(scale) '/' fileName '.png'];
                            end
                            if(flagInst)
                                listTrainValAugInst{countTrainValAug, 1} = ['/image/val/scale_' num2str(scale) '/' fileName '.png'];
                                listTrainValAugInst{countTrainValAug, 2} = ['/label_inst_' srcName '/val/scale_' num2str(scale) '/' fileName '.mat'];
                                if(scale==1)
                                    listTrainValTestInst{countTrainValTest, 1} = ['/image/val/scale_' num2str(scale) '/' fileName '.png'];
                                    listTrainValTestInst{countTrainValTest, 2} = ['/label_inst_' srcName '/val/scale_' num2str(scale) '/' fileName '.mat'];
                                end
                            end
                            if(flagCls)
                                listTrainValAugCls{countTrainValAug, 1} = ['/image/val/scale_' num2str(scale) '/' fileName '.png'];
                                listTrainValAugCls{countTrainValAug, 2} = ['/label_cls_' srcName '/val/scale_' num2str(scale) '/' fileName '.mat'];
                            end
                        end
                    else
                        % Add to test
                        countTest = countTest + 1;
                        countTrainValTest = countTrainValTest + 1;
                        listTest{countTest, 1} = ['/image/test/' fileName '.png'];
                        if(flagInst)
                            listTestInst{countTest, 1} = ['/image/test/' fileName '.png'];
                            listTestInst{countTest, 2} = ['/label_inst_' srcName '/test/' fileName '.mat'];
                            listTrainValTestInst{countTrainValTest, 1} = ['/image/test/' fileName '.png'];
                            listTrainValTestInst{countTrainValTest, 2} = ['/label_inst_' srcName '/test/' fileName '.mat'];
                        end
                    end
                    
                    if(flagBinFile)
                        if(strcmp(setName, 'train'))
                            if(~valSet(idxFile, 1))
                                % Add to train_aug and trainval_aug
                                if(flagInst)
                                    fprintf(fidListTrainAugInst, ['/image/train/scale_' num2str(scale) '/' fileName '.png /label_inst_' srcName '/train/scale_' num2str(scale) '/' fileName '.bin\n']);
                                    fprintf(fidListTrainValAugInst, ['/image/train/scale_' num2str(scale) '/' fileName '.png /label_inst_' srcName '/train/scale_' num2str(scale) '/' fileName '.bin\n']);
                                    if(scale==1)
                                        fprintf(fidListTrainValTestInst, ['/image/train/scale_' num2str(scale) '/' fileName '.png /label_inst_' srcName '/train/scale_' num2str(scale) '/' fileName '.bin\n']);
                                    end
                                end
                                if(flagCls)
                                    fprintf(fidListTrainAugCls, ['/image/train/scale_' num2str(scale) '/' fileName '.png /label_cls_' srcName '/train/scale_' num2str(scale) '/' fileName '.bin\n']);
                                    fprintf(fidListTrainValAugCls, ['/image/train/scale_' num2str(scale) '/' fileName '.png /label_cls_' srcName '/train/scale_' num2str(scale) '/' fileName '.bin\n']);
                                end
                            else
                                % Add to val and trainval_aug
                                if(scale == 1)
                                    fprintf(fidListVal, ['/image/val/scale_' num2str(scale) '/' fileName '.png\n']);
                                end
                                if(flagInst)
                                    fprintf(fidListTrainValAugInst, ['/image/val/scale_' num2str(scale) '/' fileName '.png /label_inst_' srcName '/val/scale_' num2str(scale) '/' fileName '.bin\n']);
                                    if(scale == 1)
                                        fprintf(fidListTrainValTestInst, ['/image/val/scale_' num2str(scale) '/' fileName '.png /label_inst_' srcName '/val/scale_' num2str(scale) '/' fileName '.bin\n']);
                                    end
                                end
                                if(flagCls)
                                    fprintf(fidListTrainValAugCls, ['/image/val/scale_' num2str(scale) '/' fileName '.png /label_cls_' srcName '/val/scale_' num2str(scale) '/' fileName '.bin\n']);
                                end
                            end
                        else
                            % Add to test
                            fprintf(fidListTest, ['/image/test/' fileName '.png\n']);
                            if(flagInst)
                                fprintf(fidListTestInst, ['/image/test/' fileName '.png /label_inst_' srcName '/test/' fileName '.bin\n']);
                                fprintf(fidListTrainValTestInst, ['/image/test/' fileName '.png /label_inst_' srcName '/test/' fileName '.bin\n']);
                            end
                        end
                    end
                end
                parfor_progress();
            end
            parfor_progress(0);
            
            % Write file lists
            if(strcmp(setName, 'train'))
                save([genDataRoot '/val.mat'], 'listVal');
                if(flagBinFile)
                    fclose(fidListVal);
                end
                if(flagInst)
                    save([genDataRoot '/trainvalaug_inst_' srcName '.mat'], 'listTrainValAugInst');
                    save([genDataRoot '/trainaug_inst_' srcName '.mat'], 'listTrainAugInst');
                    if(flagBinFile)
                        fclose(fidListTrainValAugInst);
                        fclose(fidListTrainAugInst);
                    end
                end
                if(flagCls)
                    save([genDataRoot '/trainvalaug_cls_' srcName '.mat'], 'listTrainValAugCls');
                    save([genDataRoot '/trainaug_cls_' srcName '.mat'], 'listTrainAugCls');
                    if(flagBinFile)
                        fclose(fidListTrainValAugCls);
                        fclose(fidListTrainAugCls);
                    end
                end
            else
                save([genDataRoot '/test.mat'], 'listTest');
                if(flagBinFile)
                    fclose(fidListTest);
                end
                if(flagInst)
                    save([genDataRoot '/test_inst_' srcName '.mat'], 'listTestInst');
                    if(flagBinFile)
                        fclose(fidListTestInst);
                    end
                end
            end
        end
    end
    if(strcmp(srcName, 'orig'))
        if(flagInst)
            save([genDataRoot '/trainvaltest_inst_' srcName '.mat'], 'listTrainValTestInst');
            if(flagBinFile)
                fclose(fidListTrainValTestInst);
            end
        end
    end
end