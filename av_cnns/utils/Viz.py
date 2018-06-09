import PIL.Image
import numpy as np
from PIL import ImageDraw 
from PIL import ImageFont 
from skimage.transform import resize 
import torchvision.transforms as transforms
from skimage.measure import compare_ssim as sk_ssim

def create_img_panno(img,mod_image,target):
    
    img = renormalize_img(img)
    mod_image = renormalize_img(mod_image)
    target = renormalize_img(target)
    
    img = np.array(transforms.ToPILImage()(img))
    mod_image = np.array(transforms.ToPILImage()(mod_image))    
    target = np.array(transforms.ToPILImage()(target))
    
    # img = img.astype(np.float32).transpose((1, 2, 0))
    # img += -img.min()
    # img *= (1/img.max()) * 255
    
    # mod_image = mod_image.astype(np.float32).transpose((1, 2, 0))
    # mod_image += -mod_image.min()
    # mod_image *= (1/mod_image.max()) * 255
    
    # target = target.astype(np.float32).transpose((1, 2, 0))
    # target += -target.min()
    # target *= (1/target.max()) * 255    
    
    panno = img_stack_horizontally([PIL.Image.fromarray(img.astype('uint8')),
                           PIL.Image.fromarray(mod_image.astype('uint8')),
                           PIL.Image.fromarray(img.astype('uint8')-mod_image.astype('uint8')),
                           PIL.Image.fromarray(target.astype('uint8'))])
    
    panno = panno.resize((panno.size[0]*2,panno.size[1]*2))
    
    draw = ImageDraw.Draw(panno)
    draw.text((0, 0),str(round(sk_ssim(img, mod_image, multichannel=True),3)),fill=(255,255,255), font=ImageFont.truetype("arial.ttf", 40))
    
    panno = np.asarray(panno)[:,:,0:3]
 
    return panno

def compare_img(img,mod_image):
    
    panno = img_stack_horizontally([img,
                                    mod_image,
                                    PIL.Image.fromarray(np.array(img) - np.array(mod_image) )])
    
    panno = panno.resize((panno.size[0]*2,panno.size[1]*2))
    
    # draw = ImageDraw.Draw(panno)
    # draw.text((0, 0),str(round(sk_ssim(img, mod_image, multichannel=True),3)),fill=(255,255,255), font=ImageFont.truetype("arial.ttf", 40))
    
    panno = np.asarray(panno)[:,:,0:3]
 
    return panno

def renormalize_img(img):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    img[0, :, :] = img[0, :, :] * std[0] + mean[0]
    img[1, :, :] = img[1, :, :] * std[1] + mean[1]
    img[2, :, :] = img[2, :, :] * std[2] + mean[2]
    
    return img

def img_stack_horizontally(images, sep=10):
    w, h = 0, 0
    for img in images:
        h = max(img.size[1], h)
        w += img.size[0] + sep
    fin = PIL.Image.new('RGBA', (w, h))
    cw = 0
    for img in images:
        fin.paste(img, (cw, 0))
        cw += img.size[0] + sep
    return fin

def img_stack_vertically(images, sep=10):
    w, h = 0, 0
    for img in images:
        w = max(img.size[0], w)
        h += img.size[1] + sep
    fin = PIL.Image.new('RGBA', (w, h))
    ch = 0
    for img in images:
        fin.paste(img, (0, ch))
        ch += img.size[1] + sep
    return fin

def back_to_int(img):
    img += -img.min()
    img *= (1/img.max())
    img *= 255
    return img.astype('uint8')    