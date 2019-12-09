import numpy as np
from PIL import Image

def _save_image(img, filename):
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3,1,1))
    std  = np.array([0.2023, 0.1994, 0.2010]).reshape((3,1,1))

    img = img * std + mean
    img = np.minimum(img, np.ones_like(img))
    img = np.maximum(img, np.zeros_like(img))
    
    img = (img*255).astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    save_img = Image.fromarray(img)
    save_img.save(filename)

