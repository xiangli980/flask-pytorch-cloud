from makedata import *
from makemodel import *
#from feature import get_feature
import six
import skimage.io as skio
from skimage.color import rgb2gray
from scipy import ndimage
import urllib
from loadweights import *


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

def test_sample(model):

    input_img = skio.imread("./input.png") 
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
from skimage.measure import label, regionprops, find_contours
import json
def get_json(fileid):
    js = {}
    im = skio.imread("./static/masks/mask_{}.png".format(fileid))
    img = skio.imread("./static/slides/slide_{}.png".format(fileid))
    if not os.path.exists('./static/feature_masks/{}'.format(fileid)):
        os.makedirs('./static/feature_masks/{}'.format(fileid))
        os.makedirs('./static/feature_slides/{}'.format(fileid))

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
        maskurl = "./static/feature_masks/{}/mask_{}.png".format(fileid,i)
        imageurl = "./static/feature_slides/{}/slide_{}.png".format(fileid,i)
        skio.imsave(maskurl, mask)
        skio.imsave(imageurl, slide)
        # fetch and extract features
        features = get_feature(imageurl, maskurl)
        for key, val in six.iteritems(features):
            keyy = key.split('_')[2]
            if(not isinstance(val, np.ndarray)):
                js[str(i)][keyy] = val
            else:
                js[str(i)][keyy] = val.tolist()

    
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

"""