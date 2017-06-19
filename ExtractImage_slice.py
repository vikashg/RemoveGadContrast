import numpy as np
import os 
import sys
from scipy import misc
import nibabel as nib
import random
import Normalize3DImage as norm3D
from sklearn.feature_extraction import image

patch_size = 100 
num_patches_per_direction = 750
total_num_patches = 3*num_patches_per_direction
x_range = np.array([72, 413])
y_range = np.array([28, 206])
z_range = np.array([83, 403])

def extract_patches_random(pre_nifti_image, post_nifti_image, patch_size, direction):
    preGad_patches = np.empty([num_patches_per_direction, patch_size*patch_size], dtype = float)
    postGad_patches = np.empty([num_patches_per_direction, patch_size*patch_size], dtype = float)

    if (direction == 0):
        slice_range = x_range
    elif (direction == 1):
        slice_range = y_range
    elif (direction == 2):
        slice_range = z_range

    num=0
    while (num< num_patches_per_direction):
        random_slice = np.random.random_integers(slice_range[0], slice_range[1])
        if (direction == 0):
            preGad_img_slice = pre_nifti_image[random_slice,:]
            postGad_img_slice = post_nifti_image[random_slice,:]
        elif (direction == 1):
            preGad_img_slice = pre_nifti_image[:,random_slice,:]
            postGad_img_slice = post_nifti_image[:,random_slice,:]
        elif (direction == 2):
            preGad_img_slice = pre_nifti_image[:,:,random_slice]
            postGad_img_slice = post_nifti_image[:,:,random_slice]
            
        patch_select_random_seed = np.random.random_integers(1,10)
        patches_pre = image.extract_patches_2d(preGad_img_slice, (patch_size, patch_size), max_patches =4, random_state = patch_select_random_seed)
        patches_post = image.extract_patches_2d(postGad_img_slice, (patch_size, patch_size), max_patches =4, random_state = patch_select_random_seed)

        #find nonzero patches 
        #Calculating percentage zero voxels 
        patch_flatten_pre = patches_pre.reshape(4,-1)
        patch_flatten_post = patches_post.reshape(4,-1)
        #print(patch_flatten_pre.shape)
        patch_flatten_pre_tmp = patch_flatten_pre
        mask = patch_flatten_pre_tmp >0 
        num_nonzero = np.sum(mask, axis =1)
        percentage_nonzero = num_nonzero/(patch_size*patch_size)
        indices = np.array(np.where(percentage_nonzero > 0.75))
        indices_shape = np.array(indices.shape)
        if (len(indices) > 0):
            for i in range(indices_shape[1]):
                idx = indices[0,i]
                preGad_patches[num,] = patch_flatten_pre[idx,]
                postGad_patches[num,] = patch_flatten_post[idx,]
                num+=1
                if (num == num_patches_per_direction):
                    break
    
    return preGad_patches, postGad_patches
            
data_dir= sys.argv[1]
out_dir= sys.argv[2]
preGadlist_n = data_dir + 'preGad_file_list.txt'
postGadlist_n = data_dir + 'postGad_file_list.txt'

preGadlist = open(preGadlist_n, 'r').readlines()
postGadlist = open(postGadlist_n, 'r').readlines()

# The image is contained in
# Sample such that there are always patch size available
#Generate 1000 points 100 in each direction 
#Generate  one numpy file containing 3000 lines per subject

count = 0
for preGad_file_name, postGad_file_name in zip(preGadlist, postGadlist):
    print(preGad_file_name.strip('\n'))
    print(postGad_file_name.strip('\n'))
    preGadimg = nib.load(preGad_file_name.strip('\n')).get_data()
    postGadimg = nib.load(postGad_file_name.strip('\n')).get_data()
    list_file_name_pre = preGad_file_name.strip('\n').split('/')
    list_file_name_post = postGad_file_name.strip('\n').split('/')
    base_file_name_pre = list_file_name_pre[-1].split('.')
    base_file_name_post= list_file_name_post[-1].split('.')
    numpy_file_pre = out_dir + base_file_name_pre[0] + '.txt'
    numpy_file_post = out_dir + base_file_name_post[0] + '.txt'


    preGad_patch_matrix_full = np.empty([total_num_patches, patch_size*patch_size], dtype = float)
    postGad_patch_matrix_full = np.empty([total_num_patches, patch_size*patch_size], dtype = float)

    for i in range(3):
        preGad_patch, postGad_patch = extract_patches_random(preGadimg, postGadimg, patch_size, i )
        start_idx = i*num_patches_per_direction 
        end_idx = start_idx + num_patches_per_direction 
        preGad_patch_matrix_full[start_idx:end_idx,] = preGad_patch
        postGad_patch_matrix_full[start_idx:end_idx,] = postGad_patch
    np.savetxt(numpy_file_pre, preGad_patch_matrix_full, fmt = "%.3f")
    np.savetxt(numpy_file_post, postGad_patch_matrix_full, fmt = "%.3f")

    count = count + 1
    if (count > 2):
        sys.exit(1)


