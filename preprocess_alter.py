#%%
from os import listdir, mkdir
from os.path import exists
from tqdm.autonotebook import tqdm
import skimage.io as io
from io import BytesIO
import dippykit as dip
import numpy as np
from PIL import Image
import cv2
import matplotlib


indir = "D:/processed/validation"
outdir = "D:/processed/jpeg"

indir = "E:/Cours/DIP/Coco/processed/validation"

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]

def alterDataset(indir, outdir, alteration, alterargs={}):
    ''' Generates the altered dataset.
    Parameters: 
        indir (string): Path to the original dataset
        outdir (string): Path where the altered dataset is saved
        alteration (function): alteration function
        alterargs (dict): arguments for the alteration function
    '''
    filelist = listdir(indir)
    if not exists(outdir):
        mkdir(outdir)
    for img_path in tqdm(filelist):
        try:
            img = io.imread(f'{indir}/{img_path}')
            img_altered = alteration(img, **alterargs)
            if img_altered is not None:
                io.imsave(f'{outdir}/{img_path}', img_altered)
        except Exception as e:
            print(e)
            continue


def alter_jpeg(indir, outdir, quality):
    ''' Introduce JPEG artifacts.
    Parameters: 
        indir (string): Path to the original dataset
        outdir (string): Path where the altered dataset is saved
        quality (int): jpeg quality in percentage
    '''
    filelist = listdir(indir)
    if not exists(outdir):
        mkdir(outdir)
    for img_path in tqdm(filelist):
        try:
            img = io.imread(f'{indir}/{img_path}')
            impil = Image.fromarray(img.astype('uint8'), 'RGB')
            impil.save(f'{outdir}/{img_path}', "JPEG", quality=quality)
        except:
            continue

def alter_huesat(img, hueshift, satmult):
    ''' Changes the hue and saturation.
    Parameters: 
        img (np.array): Image to be altered
        hueshift (float): amount by which the hue is shifted
        satmult (float): amount by which the saturation is multiplied
    '''
    img_hsv = matplotlib.colors.rgb_to_hsv(img)
    img_hsv[:,:,1] *= satmult
    img_hsv[:,:,0] += hueshift
    return matplotlib.colors.hsv_to_rgb(img_hsv)

def alter_flare(img, alpha):
    ''' Add lens flare effect.
    Parameters: 
        img (np.array): Image to be altered
        alpha (float): alpha of overlay
    '''
    flare = io.imread('3.jpg')
    out = img.astype(float)+alpha*crop_center(flare,img.shape[1],img.shape[0]).astype(float)
    out = np.minimum(out, 255*np.ones(out.shape))
    return out

def alter_rain(img, alpha):
    ''' Add rain overlay.
    Parameters: 
        img (np.array): Image to be altered
        alpha (float): alpha of overlay
    '''
    rain = io.imread('rain.png')
    out = img.astype(float)+alpha*crop_center(rain,img.shape[1],img.shape[0]).astype(float)
    out = np.minimum(out, 255*np.ones(out.shape))
    return out

def alter_chrab(img, shift):
    ''' Introduce chromatic aberations.
    Parameters: 
        img (np.array): Image to be altered
        shift (int): shift of aberation in pixels
    '''
    img[:,:,0] = np.roll(img[:,:,0], shift)
    img[:,:,1] = np.roll(img[:,:,1], -shift)
    return img


def moving_average(img, size):
	img_blur =  1/size*sum([np.roll(img, i, axis=1) for i in range(size)])
	return img_blur

def blur(img,type,kernel_size=21,std=10):
	if type=="median":
		img_blur = dip.medianBlur(img,kernel_size)
	elif type=="gaussian":
		img_blur = cv2.GaussianBlur(img,(kernel_size,kernel_size),std)
	else:
		print("type of blur not supported.")
	return img_blur

def noise(img,type,var=0.01,mean=0):
	if type == "gaussian":
		img_noisy = dip.image_noise(img,type,var=var,mean=mean)
	elif type in ["pepper", "salt", "s&p", "speckle"]:
		img_noisy = dip.image_noise(img,type)
	else:
		print("type of noise not supported.")
	return img_noisy


# img = io.imread(f'{indir}/000000000724.jpg')

# %%