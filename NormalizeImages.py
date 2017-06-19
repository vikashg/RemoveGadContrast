#This takes the postGad images and transforms the preGadImages depending on the mean of the postGadImages
import numpy as np 
from scipy import misc
import os
import sys
import readFileNames as rfn


def SaveImage(imageName, imgarray, out_dir):
    imageNamelist = imageName.split('/')
    baseName = imageNamelist[-1]
    direction = imageNamelist[-2]
    output_fileName = out_dir + '/' +  direction + '/' +  baseName  
    misc.imsave(output_fileName, imgarray)
    return output_fileName + '\n'
    

#Write a test module for checking if the postGadImages are positive
data_dir = sys.argv[1]
out_dir = sys.argv[2]

flag = 2

if (flag == 1):
    preGad_train_list, preGad_test_list, postGad_train_list, postGad_test_list =  rfn.readFileNames(data_dir, 2)
if (flag == 2):
    preGad_list, postGad_list = rfn.readFileNames(data_dir, 2)

new_preGad_file_list_name = out_dir + '/preGad_list.txt'
new_postGad_file_list_name = out_dir + '/postGad_list.txt'

preGad_file_name = open(new_preGad_file_list_name, 'w')
postGad_file_name = open(new_postGad_file_list_name, 'w')

num_train_Images = len(preGad_list)
num_train_Images = 10
for i in range(num_train_Images):
    pre_tmpImage_name = preGad_list[i].strip('\n')
    post_tmpImage_name = postGad_list[i].strip('\n')

    preImage = misc.imread(pre_tmpImage_name)
    postImage = misc.imread(post_tmpImage_name)
    mask = postImage>0

    print(preImage.max())
    print(postImage.max())
    postImage = postImage/postImage.max()
    preImage = preImage/preImage.max()
    if (i == 1): 
        diffImage = postImage - preImage
        misc.imsave(out_dir + '/diff.png', diffImage)

    # save thetransformedImage
    fileName = SaveImage(pre_tmpImage_name, preImage, out_dir)
    preGad_file_name.write(fileName)
    fileName = SaveImage(post_tmpImage_name, postImage, out_dir)
    postGad_file_name.write(fileName)
