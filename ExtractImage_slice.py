import numpy as np
import os 
import sys
from scipy import misc
import nibabel as nib
import random
import Normalize3DImage as norm3D

def get_slices (img, coord, size)
    x_start = coord[0]
    x_end = coord[0] + size
    y_start = coord[1]
    y_end = coord[1] + size
    z_start = coord[2]
    z_end = coord[2] + size 

    x_slice = img[x_start:x_end, y_start, z_start:z_end]
    y_slice = img[x_start:x_end, y_start:y_end, z_start]
    z_slice = img[x_start, y_start:y_end, z_start:z_end]

    return x_slice, y_slice, z_slice

def extract_patch(preGadimg, postGadimg coord_list, size, num_patches):

    preGad_patch_matrix = np.empty([num_patches+1000, size* size], dtype = float)
    postGad_patch_matrix = np.empty([num_patches+1000, size* size], dtype = float)
    count = 0 
    for coord in coord_list:
        x_slice, y_slice, z_slice = get_slices(preGadimg, coord, size)
        preGad_patch_matrix[count,] = x_slice.reshape(-1)
        preGad_patch_matrix[count+1,] = y_slice.reshape(-1)
        preGad_patch_matrix[count+2,] = z_slice.reshape(-1)
        
        x_slice, y_slice, z_slice = get_slices(postGadimg, coord, size)
        postGad_patch_matrix[count,] = x_slice.reshape(-1)
        postGad_patch_matrix[count+1,] = y_slice.reshape(-1)
        postGad_patch_matrix[count+2,] = z_slice.reshape(-1)
        count+=3
        
        #Getting indices  // only need to do it on one type either preGad or postGad
        sum_patch_matrix = np.sum(preGad_patch_matrix, axis =1 )
        num_nonzero = np.count_nonzero(sum_patch_matrix)

        if (num_nonzero < 2000):
            # return whatever we have
            idx_nonzero = np.array(np.where(sum_patch_matrix >0))
            preGad_patch_matrix = preGad_patch_matrix[idx_nonzero,]
            postGad_patch_matrix = postGad_patch_matrix[idx_nonzero,]
        else:
            # return first 2000 elements
            idx_sorted = np.argsort(-preGad_patch_matrix)
            preGad_patch_matrix = preGad_patch_matrix(idx_sorted[0:2000,])
            postGad_patch_matrix = postGad_patch_matrix(idx_sorted[0:2000,])
            
        return preGad_patch_matrix, postGad_patch_matrix

num_slices = 150 
#list_numbers = random.sample(range(range_img[0], range_img[1]), num_slices)

data_dir='/home/rcf-proj2/vg/Data/data_prep_nn/MNI_aligned_images/'
preGadlist_n = data_dir + 'preGad_file_list.txt'
postGadlist_n = data_dir + 'postGad_file_list.txt'

preGadlist = open(preGadlist_n, 'r').readlines()
postGadlist = open(postGadlist_n, 'r').readlines()

x_range = np.arange(72, 413)
y_range = np.arange(28, 206)
z_range = np.arange(83, 403)
#Generate 1000 points 100 in each direction 
#for filename in preGadlist:
filename_pre = preGadlist[0]
filename_post = postGadlist[0]
x_sample = list(random.sample(list(x_range), 100))
y_sample = list(random.sample(list(y_range), 100))
z_sample = list(random.sample(list(z_range), 100))

list_of_coordinates =[]
for x in x_sample:
    for y in y_sample:
        for z in z_sample:
            coord = [ x, y , z]
            list_of_coordinates.append(coord)

img = nib.load(filename_pre.strip('\n'))
 
pre_img  = nib.load(filename_pre.strip('\n')).get_data()
post_img = nib.load(filename_post.strip('\n')).get_data()
'''
postImage_mean, preImage_mean = norm3D.Normalize3DMean(post_img, pre_img)
postImage_max, preImage_max = norm3D.Normalize3DMax(post_img, pre_img)
postImage_based_on_post, preImage_based_on_post = norm3D.Normalize3DBasedOnPost(post_img, pre_img)
affine = np.diag([1, 2, 3, 1])
postImage_mean_img = nib.Nifti1Image(postImage_mean, img.affine, img.header)
preImage_mean_img = nib.Nifti1Image(preImage_mean, img.affine, img.header)

postImage_max_img = nib.Nifti1Image(postImage_max, img.affine, img.header)
preImage_max_img = nib.Nifti1Image(preImage_max, img.affine, img.header)

postImage_based_on_post_img = nib.Nifti1Image(postImage_based_on_post, img.affine, img.header)
preImage_based_on_post_img = nib.Nifti1Image(preImage_based_on_post, img.affine, img.header)

nib.save(postImage_mean_img, './scratch_dir/post_mean.nii.gz')
nib.save(preImage_mean_img, './scratch_dir/preImage_mean.nii.gz')

nib.save(postImage_max_img, './scratch_dir/postImage_max.nii.gz')
nib.save(preImage_max_img, './scratch_dir/preImage_max.nii.gz')

nib.save(postImage_based_on_post_img, './scratch_dir/postImage_based_on_post.nii.gz')
nib.save(preImage_based_on_post_img, './scratch_dir/preImage_based_on_post.nii.gz')

#Save Image 
nib.save(postImage_mean, './scratch_dir/post_mean.nii')
nib.save(preImage_mean, './scratch_dir/pre_mean.nii')
nib.save(postImage_max, './scratch_dir/post_max.nii')
nib.save(preImage_max, './scratch_dir/pre_max.nii')
nib.save(postImage_based_on_post, './scratch_dir/post_based_on_post.nii')
nib.save(preImage_based_on_post, './scratch_dir/pre_based_on_post.nii')
'''

counter = 1
num_patches_extracted = 3000
num_patches_per_image = 2000
num_subjects = len(preGadlist)
patch_size = 150 
preImage_matrix = np.empty([num_patches_per_image, patch_size, patch_size], dtype = float)
postImage_matrix = np.empty([num_patches_per_image, patch_size, patch_size], dtype = float)

'''
#x_slice, y_slice, z_slice = extract_patch(pre_img, coord, patch_size)
preImage_matrix = extract_patch(pre_img, list_of_coordinates, patch_size, num_patches_per_image)
np.save('./scratch_dir/patch_matrix.npy', preImage_matrix)    
    
#misc.imsave('./scratch_dir/x_slice_' + str(counter) + '.png', x_slice)
#misc.imsave('./scratch_dir/y_slice_' + str(counter) + '.png', y_slice)
#misc.imsave('./scratch_dir/z_slice_' + str(counter) + '.png', z_slice)
counter += 1
print(counter)
'''
