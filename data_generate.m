clc;
clear all;

%% generate the LRNDVI from LRMS images
for i=1:100
 A=((imread(strcat('.\LRMS\',num2str(i),'.tif'))));
 A_NIR=double(A(:,:,4));
 A_R=double(A(:,:,3));
 A_diff=(A_NIR-A_R);
 A_add=(A_NIR+A_R+eps);
 A_NDVI=A_diff./A_add;
 A_NDVI_norm=A_NDVI*127.5+127.5;
imwrite(uint8(A_NDVI_norm),strcat('.\LR_NDVI\',num2str(i),'.tif'))
end

%% generate the HRVI from LRMS images and HRPAN images
for i=1:100
 B=(imread(strcat('.\PAN\',num2str(i),'.tif')));
 A_small=(imread(strcat('.\LRMS\',num2str(i),'.tif')));
 A(:,:,1)=imresize(A_small(:,:,1),4);
 A(:,:,2)=imresize(A_small(:,:,2),4);
 A(:,:,3)=imresize(A_small(:,:,3),4);
 A(:,:,4)=imresize(A_small(:,:,4),4);
 PAN=double(B);
 A_R=double(A(:,:,3));
 A_diff=(PAN-A_R);
 A_add=(PAN+A_R+eps);
 A_HRVI=A_diff./A_add;
 A_HRVI_norm=A_HRVI*127.5+127.5;
 imwrite(uint8(A_HRVI_norm),strcat('.\HRVI\',num2str(i),'.tif'))
end



  