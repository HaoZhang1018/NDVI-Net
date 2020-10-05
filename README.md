# NDVI-Net
Code of paper NDVI-Net: A fusion network for generating high-resolution normalized difference vegetation index in remote sensing.
````
@article{zhang2020ndvi,
  title={NDVI-Net: A fusion network for generating high-resolution normalized difference vegetation index in remote sensing},
  author={Zhang, Hao and Ma, Jiayi and Chen, Chen and Tian, Xin},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={168},
  pages={182--196},
  year={2020},
  publisher={Elsevier}
}
````
#### Prepare data :<br>
Run "data_generate.m" to produce the NDVI and HRVI from LRMS images and PAN images.

#### To train :<br>
Put training image pairs in the "Train_NDVI", "Train_HRVI" and "Train_Label" folders, and run "CUDA_VISIBLE_DEVICES=0 python train.py" to train the network.

#### To test :<br>
Put test image pairs in the "Test_NDVI" and "Test_HRVI" folders, and run "CUDA_VISIBLE_DEVICES=0 python demo.py" to test the trained model.
You can also directly use the trained model we provide (only Quickbird and GF-2).

#### Post-processing (optional): Histogram specification based on image decomposition :<br>
Run "Decomp_Hist_specification.m" to reduce the drift of network and enhance the generalization ability of the model.
