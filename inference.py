from makedata import *
from makemodel import *
#from feature import get_feature
import radiomics
from radiomics import featureextractor

import six
import skimage.io as skio
from skimage.color import rgb2gray
from scipy import ndimage
import urllib
from loadweights import *

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import skimage.io as skio
from skimage import measure
from skimage.draw import polygon
from skimage.color import rgb2gray
import pandas as pd

def init_extractor():
    # First define the settings
    settings = {}
    settings['force2D'] = True
    settings['binWidth'] = 1
    # Instantiate the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)  # ** 'unpacks' the dictionary in the function call

    # enable 2D shape
    extractor.enableFeatureClassByName('shape',False)
    extractor.enableFeatureClassByName('shape2D')

    #extractor.enableImageTypeByName('Wavelet')
    print('Enabled features:\n\t', extractor.enabledFeatures)  # Still the default parameters
    return extractor

# shape feature
def get_feature(imagePath,maskPath,extractor):
    f = {}
    result = extractor.execute(imagePath, maskPath,label=255)
    for key, val in six.iteritems(result):
        if "shape2D" in key:
            f[key] = val
    return f


def creat_model():
    
    isTrain = False
    isContinue = False
    savedir = "./"
    loadpath= "dk1_UNet_pre60_3000.pth"
    file_id = "1-7KaljDeB-Li0nCTkrJQtlcPe4NEkEHe"
    download_file_from_google_drive(file_id, loadpath)
    print("weight download")
    model = GlomNet(isTrain, isContinue, savedir, loadpath, "UNet")
    return model
"""
def test_sample(model):
    # whole slide
    input_img = skio.imread("./static/input.png") 
    img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
            ])
    img = img_transform(input_img)

    model.set_input(img.unsqueeze(0),img)
    model.test()
    pred = model.get_pred()
    pred = pred.detach().squeeze().cpu().numpy()
    pred = np.moveaxis(pred,0,-1) 
    result = np.argmax(pred,axis=2)
    result = (ndimage.binary_fill_holes(result).astype(int))*255
    #skio.imsave("./static/output.png", result[:,:,None].repeat(3,axis=2))

    return result
"""
def test_sample(model):
    # whole slide
    input_img = skio.imread("./static/input1.png") 
    img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
            ])

    # image slices:
    

    img = img_transform(input_img)

    model.set_input(img.unsqueeze(0),img)
    model.test()
    pred = model.get_pred()
    pred = pred.detach().squeeze().cpu().numpy()
    pred = np.moveaxis(pred,0,-1) 
    result = np.argmax(pred,axis=2)
    result = (ndimage.binary_fill_holes(result).astype(int))*255
    #skio.imsave("./static/output.png", result[:,:,None].repeat(3,axis=2))

    return result

from skimage.measure import label, regionprops, find_contours
import json
def get_json(fileid):
    extractor = init_extractor()
    js = {}
    im = skio.imread("./log/{}".format(fileid))[:,:,0]
    img = skio.imread("./static/{}".format(fileid))
    filepth = fileid.split('.')[0]
    if not os.path.exists('./log/feature_masks/{}'.format(filepth)):
        os.makedirs('./log/feature_masks/{}'.format(filepth))
        os.makedirs('./log/feature_slides/{}'.format(filepth))

    # initial gloms, get attributes
    lb = label(im[:,:])
    regs = regionprops(lb)
    for i, reg in enumerate(regs[:]):
        js[str(i)]={}
        js[str(i)]["id"] = str(i)
        if(i<20):
            js[str(i)]["cat"] = "#sclGlom"
        else:
            js[str(i)]["cat"] = "#normalGlom"
        js[str(i)]["area"] = int(reg.area)
        (top, left, down, right) = reg.bbox
        js[str(i)]["left"] = left
        js[str(i)]["top"] = top
        # save glom image and mask
        mask = im[top:down+1,left:right+1]
        slide = img[top:down+1,left:right+1]
        slide = rgb2gray(slide)
        # create url for this glom to extract feature from
        maskurl = "./log/feature_masks/{}/mask_{}.png".format(filepth,i)
        imageurl = "./log/feature_slides/{}/slide_{}.png".format(filepth,i)
        skio.imsave(maskurl, mask)
        skio.imsave(imageurl, slide)
        
        # fetch and extract features
        features = get_feature(imageurl, maskurl, extractor)
        for key, val in six.iteritems(features):
            keyy = key.split('_')[2]
            if(not isinstance(val, np.ndarray)):
                js[str(i)][keyy] = val
            else:
                js[str(i)][keyy] = val.tolist()

    # add contours
    coords = find_contours(im[:,:],0.6)
    for i, coord in enumerate(coords):
        jos = []
        for x,y in zip(coord[:,1], coord[:,0]):
            pos = {}
            pos["x"]=x
            pos["y"]=y
            jos.append(pos) 
        js[str(i)]["coord"]=jos
    
    return js
    


if __name__ == "__main__":
    
    regs = get_json()

    with open('./hello_app/static/glomglom.json', 'w') as fp:
        json.dump(regs, fp)

