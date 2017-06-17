import numpy as np 

def Normalize3DMean(postImg, preImg):
    mask = postImg > 0
    postImg[mask] = postImg[mask] - np.mean(postImg[mask])
    preImg[mask] = preImg[mask] - np.mean(preImg[mask])
    return postImg, preImg

def Normalize3DMax(postImg, preImg):
    postImg = postImg/postImg.max()
    preImg = preImg/preImg.max() 
    return postImg, preImg
    
def Normalize3DBasedOnPost(postImg, preImg):
    mask = postImg > 0 
    postImg[mask] = postImg[mask] - np.mean(postImg[mask])
    preImg[mask] = preImg[mask] - np.mean(preImg[mask])
    return postImg, preImg


