
function [im_h_patch neighborhood W] = LINE(im_l_patch,im_b_patch,XLP,XHP,tau,K,maxiter)

% initialize the HR patch
im_pre_patch = im_b_patch;

if maxiter == 0
    neighborhood = [];
    W = [];
    return;
end

% updata the HR patch and weights step by step
for i=1:maxiter
    n2            = dist2(im_pre_patch', XHP');
    [value index] = sort(n2);
    neighborhood  = index(1:K);
    W             = solve_weights(XLP(:,neighborhood),im_l_patch,K,n2(:,neighborhood),tau);
    im_pre_patch  = XHP(:,neighborhood)*W;
end

im_h_patch = im_pre_patch;