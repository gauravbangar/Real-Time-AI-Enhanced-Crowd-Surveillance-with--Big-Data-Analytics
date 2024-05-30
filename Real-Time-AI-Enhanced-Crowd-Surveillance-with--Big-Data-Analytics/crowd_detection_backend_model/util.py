import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def cvtransform(piltransform): 
    return Compose([
        _convert_image_to_rgb,
        ToTensor(),
        piltransform.transforms[0],
        piltransform.transforms[1],
        piltransform.transforms[4],
    ])

def _convert_image_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)