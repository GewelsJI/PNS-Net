%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for paper in MICCAI-2021,
% 'Progressively Normalized Self-Attention Network for Video Polyp Segmentation',
% Ge-Peng Ji, Yu-Cheng Chou, Deng-Ping Fan, Geng Chen, Debesh Jha, Huazhu Fu, Ling Shao
%
% It can only be used for non-comercial purpose. 
% If you use our code, please cite our paper.
% @InProceedings{Ji_2021_Progressively,
%    author = {Fan, Deng-Ping and Wang, Wenguan and Cheng, Ming-Ming and Shen, Jianbing}, 
%    title = {Progressively Normalized Self-Attention Network for Video Polyp Segmentation},
%    booktitle = {MICCAI},
%    year = {2021}
% }
% 
% For any questions, please contact Deng-Ping Fan (dengpingfan@mail.nankai.edu.cn)
%
% Version Control:
% % 2021-03-01, Create the init version
% This code is adopted from [DAVSOD](https://github.com/DengPingFan/DAVSOD) and re-organized by GePeng-Ji.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close; clc;
% set the path of sal/gt/results
salDir = '/Volumes/Daniel-Ji/1-Research/2021-MICCAI-PolypVideoSegmentation/MICCAI-2021/Rebuttal/Result/';
Models = {'PNS-Net'};
%05','infer_021','infer_022','infer_023','infer_024','infer_025','infer_026','NestedUNet','UNet','PraNet-29','ACSNet','ResUNet++'
gtDir = '/Volumes/Daniel-Ji/1-Research/2021-MICCAI-PolypVideoSegmentation/MICCAI-2021/Rebuttal/VPS-TestSet/';
Datasets = {'Hybrid'};

Results_Save_Path = './eval-Result/';

% set the name you benchmarking. 
% Note: Support more than one `Models`/`Datasets` in one procedure


Thresholds = 1:-1/255:0;

for  m = 1:length(Models)
    
    modelName = Models{m}
    
    resVideoPath = [salDir modelName '/'];  % video saliency map for evaluated models
    
    videoFiles = dir(gtDir);
    
    videoNUM = length(videoFiles)-2;
    
    [video_Sm, video_wFm, video_mae] = deal(zeros(1,videoNUM));
    [video_column_E, video_column_Sen, video_column_Spe, video_column_Dic, video_column_IoU] = deal(zeros(videoNUM,256));
        
    for videonum = 1:length(Datasets)
        
        videofolder = Datasets{videonum}
        
        filePath = [Results_Save_Path modelName '/'];   % results save path
    
        if ~exist(filePath, 'dir')
            mkdir(filePath);
        end
        
        fileID = fopen([filePath modelName '_' videofolder '_result.txt'], 'w');
        
        
        seqPath = [gtDir videofolder '/GT/'];  % gt sequence Path
        seqFiles = dir(seqPath);
        
        seqNUM = length(seqFiles)-2;
        
        [seq_Sm, seq_wFm, seq_mae] = deal(zeros(1,seqNUM));
        [seq_column_E, seq_column_Sen, seq_column_Spe, seq_column_Dic, seq_column_IoU] = deal(zeros(seqNUM,256));
        
        for seqnum = 1: seqNUM
            
            seqfolder = seqFiles(seqnum+2).name;
            
            gt_imgPath = [seqPath seqfolder '/'];
            [fileNUM, gt_imgFiles, fileExt] = calculateNumber(gt_imgPath); %index of stop frame
%             split_part = regexp(modelName, '-', 'split');
%             resPath = [resVideoPath  split_part{3} '_' videofolder '/' seqfolder '/'];
            resPath = [resVideoPath videofolder '/Pred/' seqfolder '/'];

            [threshold_Fmeasure, threshold_Emeasure, threshold_IoU] = deal(zeros(fileNUM-2, length(Thresholds)));
            [threshold_Sensitivity, threshold_Specificity, threshold_Dice] = deal(zeros(fileNUM-2, length(Thresholds)));

            [Smeasure, wFmeasure, MAE] =deal(zeros(1,fileNUM-2));
            
            tic;
            for i = 2:fileNUM-1 % skip the first and last gt file for some of the light-flow based method
                
                name = char(gt_imgFiles{i});
                fprintf('[Processing Info] Model: %s, Dataset: %s, SalSeq: %s (%d/%d), SalMapName: %s (%d/%d)\n',modelName, videofolder, seqfolder, seqnum, seqNUM, name, i-1, fileNUM-2);

                %load gt
                gt = imread([gt_imgPath name]);

                if (ndims(gt)>2)
                    gt = rgb2gray(gt);
                end

                if ~islogical(gt)
                    gt = gt(:,:,1) > 128;
                end

                %load resMap
                resmap  = imread([resPath name]);
                %check size
                if size(resmap, 1) ~= size(gt, 1) || size(resmap, 2) ~= size(gt, 2)
                    resmap = imresize(resmap,size(gt));
                    imwrite(resmap,[resPath name]);
                    fprintf('Resizing have been operated!! The resmap size is not math with gt in the path: %s!!!\n', [resPath name]);
                end

                resmap = im2double(resmap(:,:,1));

                %normalize resmap to [0, 1]
                resmap = reshape(mapminmax(resmap(:)',0,1),size(resmap));

                % S-meaure metric published in ICCV'17 (Structure measure: A New Way to Evaluate the Foreground Map.)
                Smeasure(i-1) = StructureMeasure(resmap,logical(gt));

                % Weighted F-measure metric published in CVPR'14 (How to evaluate the foreground maps?)
                wFmeasure(i-1) = original_WFb(resmap,logical(gt));

                MAE(i-1) = mean2(abs(double(logical(gt)) - resmap));
                [threshold_E, threshold_F, threshold_Pr, threshold_Rec, threshold_Iou]  = deal(zeros(1,length(Thresholds)));
                [threshold_Spe, threshold_Dic]  = deal(zeros(1,length(Thresholds)));
                for t = 1:length(Thresholds)
                    threshold = Thresholds(t);
                    [threshold_Pr(t), threshold_Rec(t), threshold_Spe(t), threshold_Dic(t), threshold_F(t), threshold_Iou(t)] = Fmeasure_calu(resmap,double(gt),size(gt),threshold);
                    Bi_resmap = zeros(size(resmap));
                    Bi_resmap(resmap>=threshold)=1;
                    threshold_E(t) = Enhancedmeasure(Bi_resmap, gt);
                end
                
                threshold_Emeasure(i-1,:) = threshold_E;
                threshold_Fmeasure(i-1,:) = threshold_F;
                threshold_Sensitivity(i-1,:) = threshold_Rec;
                threshold_Specificity(i-1,:) = threshold_Spe;
                threshold_Dice(i-1,:) = threshold_Dic;
                threshold_IoU(i-1,:) = threshold_Iou;
                
            end
            toc;
            
            %MAE
            seq_mae(seqnum) = mean2(MAE);
            %Sm
            seq_Sm(seqnum) = mean2(Smeasure);
            %wFm
            seq_wFm(seqnum) = mean2(wFmeasure);
            %E-m
            seq_column_E(seqnum,:) = mean(threshold_Emeasure, 1);
            seq_meanEm = mean(seq_column_E(seqnum,:));
            seq_maxEm = max(seq_column_E(seqnum,:));
            %Sensitivity
            seq_column_Sen(seqnum,:) = mean(threshold_Sensitivity, 1);
            seq_meanSen = mean(seq_column_Sen(seqnum,:));
            seq_maxSen = max(seq_column_Sen(seqnum,:));
            %Specificity
            seq_column_Spe(seqnum,:) = mean(threshold_Specificity, 1);
            seq_meanSpe = mean(seq_column_Spe(seqnum,:));
            seq_maxSpe = max(seq_column_Spe(seqnum,:));
            %Dice
            seq_column_Dic(seqnum,:) = mean(threshold_Dice,1);
            seq_meanDic = mean(seq_column_Dic(seqnum,:));
            seq_maxDic = max(seq_column_Dic(seqnum,:));
            %IoU
            seq_column_IoU(seqnum,:) = mean(threshold_IoU,1);
            seq_meanIoU = mean(seq_column_IoU(seqnum,:));
            seq_maxIoU = max(seq_column_IoU(seqnum,:));
            
            fprintf(fileID, '(Dataset:%s; %s Sequence) seq_meanDic:%.3f;seq_meanIoU:%.3f;seq_wFm:%.3f;seq_Sm:%.3f;seq_meanEm:%.3f;seq_MAE:%.3f;seq_maxEm:%.3f;seq_maxDice:%.3f;seq_maxIoU:%.3f;seq_meanSen:%.3f;seq_maxSen:%.3f;seq_meanSpe:%.3f;seq_maxSpe:%.3f.\n',...
                videofolder,seqfolder,seq_meanDic,seq_meanIoU,seq_wFm(seqnum),seq_Sm(seqnum),seq_meanEm,seq_mae(seqnum),seq_maxEm,seq_maxDic,seq_maxIoU,seq_meanSen,seq_maxSen,seq_meanSpe,seq_maxSpe);
            fprintf('(Dataset:%s; %s Sequence) seq_meanDic:%.3f;seq_meanIoU:%.3f;seq_wFm:%.3f;seq_Sm:%.3f;seq_meanEm:%.3f;seq_MAE:%.3f;seq_maxEm:%.3f;seq_maxDice:%.3f;seq_maxIoU:%.3f;seq_meanSen:%.3f;seq_maxSen:%.3f;seq_meanSpe:%.3f;seq_maxSpe:%.3f.\n',...
                videofolder,seqfolder,seq_meanDic,seq_meanIoU,seq_wFm(seqnum),seq_Sm(seqnum),seq_meanEm,seq_mae(seqnum),seq_maxEm,seq_maxDic,seq_maxIoU,seq_meanSen,seq_maxSen,seq_meanSpe,seq_maxSpe);
            
        end
        
        %MAE
        video_mae(videonum) = mean2(seq_mae);
        %Sm
        video_Sm(videonum) = mean2(seq_Sm);
        %wFm
        video_wFm(videonum) = mean2(seq_wFm);
        %E-m
        video_column_E(videonum,:) = mean(seq_column_E, 1);
        meanEm = mean(video_column_E(videonum,:));
        maxEm = max(video_column_E(videonum,:));
        %Sensitivity
        video_column_Sen(videonum,:) = mean(seq_column_Sen, 1);
        meanSen = mean(video_column_Sen(videonum,:));
        maxSen = max(video_column_Sen(videonum,:));
        %Specificity
        video_column_Spe(videonum,:) = mean(seq_column_Spe, 1);
        meanSpe = mean(video_column_Spe(videonum,:));
        maxSpe = max(video_column_Spe(videonum,:));
        %Dice
        video_column_Dic(videonum,:) = mean(seq_column_Dic,1);
        meanDic = mean(video_column_Dic(videonum,:));
        maxDic = max(video_column_Dic(videonum,:));
        %IoU
        video_column_IoU(videonum,:) = mean(seq_column_IoU,1);
        meanIoU = mean(video_column_IoU(videonum,:));
        maxIoU = max(video_column_IoU(videonum,:));
        
        fprintf(fileID, '(Dataset:%s) meanDic:%.3f;meanIoU:%.3f;wFm:%.3f;Sm:%.3f;meanEm:%.3f;MAE:%.3f;maxEm:%.3f;maxDice:%.3f;maxIoU:%.3f;meanSen:%.3f;maxSen:%.3f;meanSpe:%.3f;maxSpe:%.3f.\n',...
                videofolder,meanDic,meanIoU,video_wFm(videonum),video_Sm(videonum),meanEm,video_mae(videonum),maxEm,maxDic,maxIoU,meanSen,maxSen,meanSpe,maxSpe);
        fprintf('(Dataset:%s) meanDic:%.3f;meanIoU:%.3f;wFm:%.3f;Sm:%.3f;meanEm:%.3f;MAE:%.3f;maxEm:%.3f;maxDice:%.3f;maxIoU:%.3f;meanSen:%.3f;maxSen:%.3f;meanSpe:%.3f;maxSpe:%.3f.\n',...
            videofolder,meanDic,meanIoU,video_wFm(videonum),video_Sm(videonum),meanEm,video_mae(videonum),maxEm,maxDic,maxIoU,meanSen,maxSen,meanSpe,maxSpe);
    end
    
    fclose(fileID);
   
end

