#This takes the postGad images and transforms the preGadImages depending on the mean of the postGadImages
import numpy as np 
from scipy import misc
import os
import sys
import readFileNames as rfn


#Write a test module for checking if the postGadImages are positive
data_dir = sys.argv[1]
out_dir = sys.argv[2]

preGad_train_list, preGad_test_list, postGad_train_list, postGad_test_list =  rfn.readFileNames(data_dir)



