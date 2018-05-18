
function im_SR = LINESR(im_l,im_b,XH,XL,upscale,patch_size,overlap,tau,K,maxiter)

[imrow imcol nTraining] = size(XH);

Img_SUM      = zeros(imrow,imcol);
overlap_FLAG = zeros(imrow,imcol);

U = ceil((imrow-overlap)/(patch_size-overlap));% the patch number in every column  
V = ceil((imcol-overlap)/(patch_size-overlap));% the patch number in every row

% hallucinate the HR patch by patch
for u = 1:U
    fprintf('.');
   for v = 1:V    
        BlockSize = GetCurrentBlockSize(imrow,imcol,patch_size,overlap,u,v);    
        if size(XH,1) ~= size(XL,1)
            BlockSizeS = GetCurrentBlockSize(imrow/upscale,imcol/upscale,patch_size/upscale,overlap/upscale,u,v);  
        else
            BlockSizeS = BlockSize;
            upscale    = 1;
        end
        
        im_l_patch = im_l(BlockSizeS(1):BlockSizeS(2),BlockSizeS(3):BlockSizeS(4));           % extract the patch at position£¨li,lj£©of the input LR face     
        im_l_patch = double(reshape(im_l_patch,patch_size*patch_size/(upscale*upscale),1));   % Reshape 2D image patch into 1D column vectors   
        
        im_b_patch = im_b(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4));
        im_b_patch = double(im_b_patch(:));
        
        XHP = Reshape3D(XH,BlockSize);    % reshape each patch of HR face image to one column
        XLP = Reshape3D(XL,BlockSizeS);   % reshape each patch of LR face image to one column  

        % represent the LR patch at  position£¨u,v£©using SR 
        [im_h_patch neighborhood w] = LINE(im_l_patch, im_b_patch, XLP, XHP, tau, K, maxiter); 

        if maxiter ~= 0
                Img =  XHP(:,neighborhood)*w;
        else
                Img =  im_h_patch;
        end
        
        % integrate all the LR patch        
        Img = reshape(Img,patch_size,patch_size);
        Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))      = Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+Img;
        overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) = overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+1;
    end
end
%  averaging pixel values in the overlapping regions
im_SR = Img_SUM./overlap_FLAG;

