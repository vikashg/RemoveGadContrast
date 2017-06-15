import numpy as np 
import nibabel as nib
import tflearn as tf
data_dir='/ifs/loni/faculty/thompson/four_d/vgupta/Data/Gad_data/pre-post-spgr-test-3.24.17/3556055/20170125_154448/Nifti_files/aligned_MNI'

image_name='postGad_alignedMNI.nii.gz'

fullImagepath=data_dir + '/' + image_name

postImg = nib.load(fullImagepath)
postImg_data = postImg.get_data()

print(postImg_data.shape)
