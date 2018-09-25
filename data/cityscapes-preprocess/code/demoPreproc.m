% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to:
% 1. Generate instance-sensitive multi-label semantic edges on the Cityscapes dataset
% 2. Create filelists for the generated data and labels
% --------------------------------------------------------

function demoPreproc()
clc; clear; close all;

%% Add library paths
path = genpath('../../../lib/matlab');
addpath(path);

%% Setup Directories and Suffixes
dataRoot = '../data_orig';
genDataRoot = '../data_proc';
suffixImage = '_leftImg8bit.png';
suffixColor = '_gtFine_color.png';
suffixLabelIds = '_gtFine_labelIds.png';
suffixInstIds = '_gtFine_instanceIds.png';
suffixTrainIds = '_gtFine_trainIds.png';
suffixPolygons = '_gtFine_polygons.json';
suffixEdge1 = '_gtFine_edge.bin';
suffixEdge2 = '_gtFine_edge.mat';

%% Setup Parameters
numCls = 19;
radius = 2;
flagBinFile = false; % Optional: Output .bin label files that can be read by PyCaffe CASENet/SEAL

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

%% Generate Output Directory
if(exist(genDataRoot, 'file')==0)
    mkdir(genDataRoot);
end

%% Preprocess Training Data and Labels
setList = {'train', 'val', 'test'};
for idxSet = 1:length(setList)
    setName = setList{idxSet};
    if(flagBinFile)
        fidList = fopen([genDataRoot '/' setName '.txt'], 'w');
    end
    dataList = cell(1, 1);
    countFile = 0;
    cityList = dir([dataRoot '/leftImg8bit/' setName]);
    for idxCity = 3:length(cityList)
        cityName = cityList(idxCity).name;
        if(exist([genDataRoot '/leftImg8bit/' setName '/' cityName], 'file')==0)
            mkdir([genDataRoot '/leftImg8bit/' setName '/' cityName]);
        end
        if(exist([genDataRoot '/gtFine/' setName '/' cityName], 'file')==0)
            mkdir([genDataRoot '/gtFine/' setName '/' cityName]);
        end
        fileList = dir([dataRoot '/leftImg8bit/' setName '/' cityName '/*.png']);
        
        % Generate and write data
        display(['Set: ' setName ', City: ' cityName])
        parfor_progress(length(fileList));
        parfor idxFile = 1:length(fileList)
        	assert(strcmp(fileList(idxFile).name(end-length(suffixImage)+1:end), suffixImage), 'suffixImage mismatch!')
            fileName = fileList(idxFile).name(1:end-length(suffixImage));
            % Copy image
            copyfile([dataRoot '/leftImg8bit/' setName '/' cityName '/' fileName suffixImage], [genDataRoot '/leftImg8bit/' setName '/' cityName '/' fileName suffixImage]);
            % Copy gt files
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixColor], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixColor]);
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds]);
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds]);
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixPolygons], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixPolygons]);
            if(~strcmp(setName, 'test'))
                % Transform label id map to train id map and write
                labelIdMap = imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds]);
                instIdMap = imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds]);
                trainIdMap = labelid2trainid(labelIdMap);
                imwrite(trainIdMap, [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixTrainIds], 'png');
                % Transform color map to edge map and write
                edgeMapBin = seg2edge(instIdMap, radius, [2 3]', 'regular'); % Avoid generating edges on "rectification border" (labelId==2) and "out of roi" (labelId==3)
                [height, width, ~] = size(trainIdMap);
                labelEdge = cell(numCls, 1);
                labelEdge2 = zeros(height, width, 'uint32');
                for idxCls = 1:numCls
                    idxSeg = trainIdMap == idxCls-1;
                    if(sum(idxSeg(:))~=0)
                        segMap = zeros(size(instIdMap));
                        segMap(idxSeg) = instIdMap(idxSeg);
                        idxEdge = seg2edge_fast(segMap, edgeMapBin, radius, [], 'regular');
                        labelEdge{idxCls, 1} = sparse(idxEdge);
                        labelEdge2(idxEdge) = labelEdge2(idxEdge) + 2^(idxCls-1);
                    else
                        labelEdge{idxCls, 1} = sparse(false(height, width));
                    end
                end
                if(flagBinFile)
                    fidEdge = fopen([genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixEdge1], 'w');
                    fwrite(fidEdge, labelEdge2', 'uint32'); % Important! Transpose input matrix to become row major
                    fclose(fidEdge);
                end
                savelabeledge([genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixEdge2], labelEdge); % parfor does not support directly using save.
            end
            parfor_progress();
        end
        parfor_progress(0);
        
        % Create file lists
        for idxFile = 1:length(fileList)
            countFile = countFile + 1;
            fileName = fileList(idxFile).name(1:end-length(suffixImage));
            if(ismember(setName, {'train', 'val'}))
                if(flagBinFile)
                    fprintf(fidList, ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage ' /gtFine/' setName '/' cityName '/' fileName suffixEdge1 '\n']);
                end
                dataList{countFile, 1} = ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage];
                dataList{countFile, 2} = ['/gtFine/' setName '/' cityName '/' fileName suffixEdge2];
            else
                if(flagBinFile)
                    fprintf(fidList, ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage '\n']);
                end
                dataList{countFile, 1} = ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage];
            end
        end
    end
    if(flagBinFile)
        fclose(fidList); %#ok<*UNRCH>
    end
    save([genDataRoot '/' setName '.mat'], 'dataList');
end

%% Preprocess Demo Video Sequences
seqList = dir([dataRoot '/leftImg8bit/demoVideo']);
for idxCity = 3:length(seqList)
    seqName = seqList(idxCity).name;
    if(exist([genDataRoot '/leftImg8bit/demoVideo/' seqName], 'file')==0)
        mkdir([genDataRoot '/leftImg8bit/demoVideo/' seqName]);
    end
    fileList = dir([dataRoot '/leftImg8bit/demoVideo/' seqName '/*.png']);
    
    % Generate and write data
    display(['Set: demoVideo, City: ' seqName])
    parfor_progress(length(fileList));
    parfor idxFile = 1:length(fileList)
        fileName = fileList(idxFile).name(1:end-length(suffixImage));
        % Copy image
        copyfile([dataRoot '/leftImg8bit/demoVideo/' seqName '/' fileName suffixImage], [genDataRoot '/leftImg8bit/demoVideo/' seqName '/' fileName suffixImage]);
        parfor_progress();
    end
    parfor_progress(0);
    
    % Create file lists
    if(flagBinFile)
        fidList = fopen([genDataRoot '/demoVideo_' seqName '.txt'], 'w');
    end
    dataList = cell(1, 1);
    countFile = 0;
    for idxFile = 1:length(fileList)
        countFile = countFile + 1;
        fileName = fileList(idxFile).name(1:end-length(suffixImage));
        if(flagBinFile)
            fprintf(fidList, ['/leftImg8bit/demoVideo/' seqName '/' fileName suffixImage '\n']);
        end
        dataList{countFile, 1} = ['/leftImg8bit/demoVideo/' seqName '/' fileName suffixImage];
    end
    if(flagBinFile)
        fclose(fidList);
    end
    save([genDataRoot '/demoVideo_' seqName '.mat'], 'dataList');
end