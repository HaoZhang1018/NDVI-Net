%% Histogram specification based on image decomposition
%%%It can not only reduce the drift of network, but also enhance the
%%%generalization ability of the model.
clc;
clear all;
ratio=4;
 for j=1:90
     A=double((imread(strcat('.\init_result\',num2str(j),'.tif'))));
     h = fspecial('gaussian',[3,3]); 
     base = imfilter(A, h, 'replicate');
     detail = A-base;        
     LR_NDVI_target=imread(strcat('.\LR_NDVI_target\',num2str(j),'.tif'));
     HR_NDVI_target=imresize(LR_NDVI_target,ratio,'nearest');
     Jmatch1=imhist(HR_NDVI_target);
     A_base=double(histeq(uint8(base),Jmatch1)); 
     final_result=A_base+detail;  
     imwrite(uint8(final_result),strcat('.\final_result\',num2str(j),'.tif')); 
 end




