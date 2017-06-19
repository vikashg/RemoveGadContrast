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


def extract_patches_sklearn(nifti_image, patch_size):
    #Generate slices in x
    maxpatches_perslice = 10
    num_patches = 1000
    patches_main =  np.empty([num_patches, patch_size* patch_size], dtype = float)
    main_counter = 0

    while (num < 1000):
        x_idx = list(np.random.choice(list(x_range), 100), replace = False)

        for idx in x_idx:
            img_slice_x = nifti_image[idx,]
            patches_x = image.extract_patches_2d(img_slice_x, (patch_size, patch_size), max_patches = maxpatches_perslice)
            patches_flatten = patches_x.reshape(maxpatches_perslice,-1)

            num_nonzero = np.count_nonzero(patches_flatten)
            percentage_nonzero = num_nonzero/(patch_size*patch_size)
            indices = np.where(percentage_nonzero > 0.75)
            patches_remain = patches_flatten[indices,]
            count=len(patches_remain)
            main_counter += count



def get_slices (img, coord, size):
    x_start = coord[0]
    x_end = coord[0] + size
    y_start = coord[1]
    y_end = coord[1] + size
    z_start = coord[2]
    z_end = coord[2] + size
    #print(coord)
    #print(img.shape)
    #print(x_start, x_end, y_start, y_end, z_start, z_end)

    x_slice = img[x_start,       y_start:y_end, z_start:z_end]
    y_slice = img[x_start:x_end, y_start,       z_start:z_end]
    z_slice = img[x_start:x_end, y_start:y_end, z_start]
    #print(x_slice.shape, y_slice.shape, z_slice.shape)
    return x_slice, y_slice, z_slice

def extract_patch(preGadimg, postGadimg, coord_list, size, num_patches):

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
