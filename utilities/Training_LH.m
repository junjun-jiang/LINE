function [YH YL] = Training_LH(upscale,BlurWindow,nTraining)
%%% construct the HR and LR training pairs from the CAS-PEAL-R1 face database
disp('Constructing the HR-LR training set...');
for i=1:nTraining
    %%% read the HR face images from the HR training set
    strh = strcat('.\trainingFaces\',num2str(i),'_HR.tif');    
    HI = imread(strh); 
    YH(:,:,i) = HI;
    
    %%% generate the LR face image by smooth and down-sampling
    w=fspecial('average',[BlurWindow BlurWindow]);
    SI = imfilter(HI,w);
    LI = imresize(SI,1/upscale,'bicubic');
    YL(:,:,i) = LI;
end

disp('done.');