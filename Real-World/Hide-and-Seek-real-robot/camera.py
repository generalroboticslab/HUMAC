import torch
import cv2
import numpy as np
from defisheye import Defisheye
import torchvision.transforms as transforms
# from scipy.stats import zscore
import numpy as np
from PIL import Image


def get_seeker_pixel(image_tensor,seeker_position,vicon_offset,image_size = 960):
    
    if image_tensor is None:
        return None,None
    resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size , image_size )),
    transforms.ToTensor(),
    ])

    image_tensor = resize_transform(image_tensor)


    if seeker_position is None:
        return image_tensor,None


    seeker_in_image_position_x = int((vicon_offset[0] - seeker_position[0])/(2*vicon_offset[0]) * image_size )
    seeker_in_image_position_y = int((vicon_offset[1] + seeker_position[1])/(2*vicon_offset[1]) * image_size )


    seeker_in_image_position = (seeker_in_image_position_y, seeker_in_image_position_x)

    final_pixel = seeker_in_image_position
    return image_tensor,final_pixel


def process_image(raw):
 # expects numpy arraw (H, W, C)

    processed = raw[: , 50: -110, :]
    width, height = processed.shape[:2]

    corners = np.float32([[17,19], [467,12], [454,454], [29,451]])
    output = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])
    
    # compute perspective matrix
    matrix = cv2.getPerspectiveTransform(corners,output)

    #pad the image
    processed = cv2.copyMakeBorder(processed, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0,0,0])
    processed = cv2.warpPerspective(processed, matrix, (550, 550), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    dtype = 'linear'
    format = 'fullframe'
    fov = 40
    pfov = 40
    img = processed

    obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)
    new_image = obj.convert()[27:-29, 27:-29, :]

    return new_image


class AccumulatedCamera():
    def __init__(self, h, w, ref_img, mask_size=40):
        """AccumulatedCamera
        Args:
            w (int): width of the camera
            h (int): height of the camera
            mask_size (int): size of the mask
        """
        self.mask = torch.zeros((1, 1, h, w), dtype=torch.bool)
        self.mask_size = mask_size
        self.w = w
        self.h = h


        self.ref_img = ref_img.unsqueeze(0)


    def reset_mask(self):
        self.mask.fill_(0)


    def add_mask(self, img, target_position):
        # print("function:",target_position)
        if target_position is None:
            return img
        # print(img.max(), img.min())
        img = img.unsqueeze(0)
        start_x = target_position[0] - self.mask_size//2
        start_y = target_position[1] - self.mask_size//2
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        
        final_x = target_position[0] + self.mask_size//2
        final_y = target_position[1] + self.mask_size//2
        if final_x >= self.w:
            final_x = self.w
        if final_y >= self.h:
            final_y = self.h
        
        
        self.mask[:, :, start_x:final_x, start_y:final_y] = 1


        history_img = self.ref_img * self.mask.float()

        history_img[:, :, start_x:final_x, start_y:final_y] = img[:, :, start_x:final_x, start_y:final_y]
        return history_img

