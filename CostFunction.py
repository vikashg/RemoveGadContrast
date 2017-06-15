import numpy as np
import tensorflow as tf
import scipy

def CreateBatch(fileNamearray, batch_num, batch_size):
    num_files = len(fileNamearray)
    start_idx = batch_num*batch_size
    end_idx = start_idx + batch_size
    img_temp_name = fileNamearray[0].strip('\n')
    img = scipy.misc.imread(img_temp_name)
    imgShape = np.array(img.shape)
    numVox = imgShape[0]*imgShape[1]

    img_matrix = np.full((batch_size, numVox), 0, dtype = 'float')
    i=0
    for idx in range(start_idx, end_idx):
        fileName = fileNamearray[idx].strip('\n')
        img = scipy.misc.imread(fileName).astype(float)
        img_reshaped = np.reshape(img, [-1, numVox])
        img_matrix[i,:] = img_reshaped
        i= i + 1

    return img_matrix

def AddBatchLoss(total_loss, batch_loss):
    return total_loss+batch_loss 

def SSD(obs_batch, pred_batch, numVox):
    pred_batch = tf.reshape(pred_batch, [-1, numVox])
    cost = tf.reduce_mean(tf.squared_difference(obs_batch, pred_batch))
    return cost

def ComputeHistogram(img_1, img_2, num_bins):
    img1_mod = tf.divide(img_1, num_bins)
    img2_mod = tf.divide(img_2, num_bins)

    img1_mod = tf.floor(img1_mod)
    img2_mod = tf.floor(img2_mod)

    n_bins = tf.ceil(255/num_bins)
    tf.cast(n_bins, tf.int8)
   

    #hist = np.full([n_bins.eval(), n_bins.eval()], 0 , dtype = float)
    hist = np.full([10, 10], 0, dtype = float)
    imgShape = np.asarray(img_1.shape)

    for i in range(imgShape[0]):
        for j in range(imgShape[1]):
            idx_x = int(img1_mod[i,j])
            idx_y = int(img2_mod[i,j])
            hist[idx_x, idx_y] =hist[idx_x, idx_y] + 1

    max_frequency = np.max(hist)
    hist_norm = hist/max_frequency

    return hist_norm

def MutualInfo(img1, img2, num_bins):

    histo = ComputeHistogram(img1, img2, num_bins)

    pxy = histo/tf.reduce_sum(histo)
    px = tf.reduce_sum(histo, axis=0)
    py = tf.reduce_sum(histo, axis =1 )
    px_py = tf.reduce_sum(tf.multiply(px, py))

    histoShape = pxy.shape
    zero_mat = tf.constant(0, dtype= float, shape=histoShape )
    zero_ind = tf.greater(pxy, zero_mat)
    #cost = np.sum(pxy[num_zeros] * np.log(pxy[num_zeros] / px_py[num_zeros]))
    cost = tf.reduce_sum(pxy[zero_ind] * tf.log(pxy[zero_ind]/px_py[zero_ind]))
    return cost

def MutualInfoWrapper(batch_obs, batch_pred, imageShape):
    batch_size = len(batch_obs)
    num_bins = 10 
    totalCost = 0
    for i in range(batch_size):
        img_obs = batch_obs[i,:]
        img_obs = tf.reshape(img_obs, imageShape)
        img_pred = batch_pred[i,:]
        img_pred = tf.reshape(img_pred, imageShape)
        cost = MutualInfo(img_pred, img_obs, num_bins)
        totalCost = totalCost + cost

    avg_cost = totalCost/batch_size
    return avg_cost   
        

