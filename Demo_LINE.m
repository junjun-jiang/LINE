
% =========================================================================
% Simple demo codes for face hallucination via LINE
%=========================================================================

clc;close all;
clear all;
addpath('.\utilities');

% set parameters
nrow        = 112;        % rows of HR face image
ncol        = 100;        % cols of LR face image
nTraining   = 1000;       % number of training sample
nTesting    = 1;         % number of test sample
upscale     = 4;          % upscaling factor 
BlurWindow  = 4;          % size of an averaging filter 
tau         = 1e-5;       % locality regularization
patch_size  = 12;         % image patch size
overlap     = 4;          % the overlap between neighborhood patches
maxiter     = 5;          % the maximum iteration number
K           = 150;        % the number of nearest neighbors

YH = zeros(nrow,ncol,nTraining); 
YL = zeros(nrow,ncol,nTraining); 

bb_psnr = zeros(1,nTesting);
sr_psnr = zeros(1,nTesting);
bb_ssim = zeros(1,nTesting);
sr_ssim = zeros(1,nTesting);

% construct the HR and LR training pairs from the CAS-PEAL-R1 face database
% [YH YL] = Training_LH(upscale,BlurWindow,nTraining);
load('YH_YL_CASPEAL.mat','YH','YL');
fprintf('\nface hallucinating for %d input test images\n', nTesting);

for TestImgIndex = 1:nTesting

    fprintf('\nProcessing  %d/%d LR image ', TestImgIndex,nTesting);

    % read ground truth of one test face image
    strh = strcat('.\testFaces\',num2str(TestImgIndex),'_test.tif');
    im_h = imread(strh);

    % generate the input LR face image by smooth and down-sampleing
    w=fspecial('average',[BlurWindow BlurWindow]);
    im_s = imfilter(im_h,w);
    im_l = imresize(im_s,1/upscale,'bicubic'); % the input LR face iamge
    im_b = imresize(im_l,upscale,'bicubic');   % initialize the HR face iamge
   
    % face hallucination via LINE
    [im_SR] = LINESR(im_l,im_b,YH,YL,upscale,patch_size,overlap,tau,K,maxiter);

    % bicubic interpolation for reference
    im_b = imresize(im_l, upscale, 'bicubic');

    % compute PSNR and SSIM for Bicubic and our method
    bb_psnr(TestImgIndex) = psnr(im_b,im_h);
    bb_ssim(TestImgIndex) = ssim(im_b,im_h);

    sr_psnr(TestImgIndex) = psnr(im_SR,im_h);
    sr_ssim(TestImgIndex) = ssim(im_SR,im_h);

%     % display the objective results (PSNR and SSIM)
    fprintf('\nPSNR for Bicubic interpolation:   %f dB\n', bb_psnr(TestImgIndex));
    fprintf('PSNR for LINE face halluciantion: %f dB\n', sr_psnr(TestImgIndex));
    fprintf('SSIM for Bicubic interpolation:    %f \n', bb_ssim(TestImgIndex));
    fprintf('SSIM for LINE face halluciantion:  %f \n', sr_ssim(TestImgIndex));

    % show the images
    figure, subplot(1,3,1);imshow(im_b);
    title('Bicubic Interpolation');    
    xlabel({['PSNR = ',num2str(bb_psnr(TestImgIndex))]; ['SSIM = ',num2str(bb_ssim(TestImgIndex))]});
    
    subplot(1,3,2);imshow(uint8(im_SR));
    title('LINE Recovery');    
    xlabel({['PSNR = ',num2str(sr_psnr(TestImgIndex))]; ['SSIM = ',num2str(sr_ssim(TestImgIndex))]});
    
    subplot(1,3,3);imshow(uint8(im_h));
    title('Original HR face');
    
    % save the result
    strw = strcat('./results/',num2str(TestImgIndex),'_LINESR.tif');
    imwrite(uint8(im_SR),strw,'bmp');
end

fprintf('===============================================\n');
fprintf('Average PSNR of Bicubic interpolation:   %f dB\n', sum(bb_psnr)/nTesting);
fprintf('Average PSNR of LINE face halluciantion: %f dB\n', sum(sr_psnr)/nTesting);
fprintf('Average SSIM of Bicubic interpolation:   %f\n', sum(bb_ssim)/nTesting);
fprintf('Average SSIM of LINE face halluciantion: %f\n', sum(sr_ssim)/nTesting);
fprintf('===============================================\n');





