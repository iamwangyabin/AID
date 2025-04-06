import torch
import numpy as np
import pickle
from PIL import Image


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath, noise_type=None):
    # img = lycon.load(filepath)
    img = np.array(Image.open(filepath).convert('RGB'))
    # import pdb;pdb.set_trace()
    img = img.astype(np.float32)
    img = img/255.
    return img


def unpack_raw(im):
    bs,chan,h,w = im.shape 
    H, W = h*2, w*2
    img2 = torch.zeros((bs,H,W))
    img2[:,0:H:2,0:W:2]=im[:,0,:,:]
    img2[:,0:H:2,1:W:2]=im[:,1,:,:]
    img2[:,1:H:2,0:W:2]=im[:,2,:,:]
    img2[:,1:H:2,1:W:2]=im[:,3,:,:]
    img2 = img2.unsqueeze(1)
    return img2

def pack_raw(im):
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    ## R G G B
    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:],
                       im[1:H:2,1:W:2,:]), axis=2)
    return out

def pack_raw_torch(im):
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    ## R G G B
    out = torch.cat((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:],
                       im[1:H:2,1:W:2,:]), dim=2)
    return out
