clear;close all;
%% settings
folder = 'Set5';
scale = 4;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));

for i = 1 : length(filepaths)        
    im_gt = imread(fullfile(folder,filepaths(i).name));
    im_gt = modcrop(im_gt, scale);
    im_gt = double(im_gt);
    im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
    im_gt_y = im_gt_ycbcr(:,:,1) * 255.0;
    im_l_ycbcr = imresize(im_gt_ycbcr, 1/scale, 'bicubic');
    im_b_ycbcr = imresize(im_l_ycbcr, scale, 'bicubic');
    im_l_y = im_l_ycbcr(:,:,1) * 255.0;
    im_l = ycbcr2rgb(im_l_ycbcr) * 255.0;
    im_b_y = im_b_ycbcr(:,:,1) * 255.0;
    im_b = ycbcr2rgb(im_b_ycbcr) * 255.0;

    filename = sprintf('Set5-output/%s.mat',filepaths(i).name);
    save(filename, 'im_gt_y', 'im_b_y', 'im_gt', 'im_b', 'im_l_ycbcr', 'im_l_y', 'im_l');
end
