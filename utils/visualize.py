import numpy as np
from PIL import Image
#from scipy.misc import imresize

def _save_image(img):
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3,1,1))
    std  = np.array([0.2023, 0.1994, 0.2010]).reshape((3,1,1))

    img = img * std + mean
    img = np.minimum(img, np.ones_like(img))
    img = np.maximum(img, np.zeros_like(img))
    
    img = (img*255).astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    save_img = Image.fromarray(img)
    #save_img.save(filename)
    return save_img

def _save_gradCAM(fs):
    size = fs[0].shape
    img = fs[0]
    for f in fs[1:]:
        img = img + np.array(Image.fromarray(f).resize(size))

    #TODO: Consider normalizing CAM with maximum value
    img = img / np.max(img)
    #img = np.minimum(img, np.ones_like(img))
    img = np.maximum(img, np.zeros_like(img))
    img = (img*255).astype(np.uint8)
    save_img = Image.fromarray(img)
    #save_img.save(filename)
    
    return save_img

def img_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im1.width + 3, im1.height))
    dst.paste(im1, (0,0))
    dst.paste(im2.resize((im1.width, im1.height)), (im1.width+3, 0))
    return dst

def img_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width,im1.height + im1.height + 3))
    dst.paste(im1, (0,0))
    dst.paste(im2.resize((im1.width, im1.height)), (0, im1.height+3))
    return dst

def _save_2x2(imgs, filename):
    im1 = img_concat_h(imgs[0], imgs[1].convert('RGB'))
    im2 = img_concat_h(imgs[2], imgs[3].convert('RGB'))
    dst = img_concat_v(im1, im2)
    dst.save(filename)
    print('%s saved'%filename)


