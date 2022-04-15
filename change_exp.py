import os
import numpy as np
import cv2

path = 'D:/github/stylegan2-ada-pytorch/target_image/63989.png'
save_path = 'D:/github/stylegan2-ada-pytorch/target_image/'

img = cv2.imread(path).astype(np.float32)/255.
img_exp_1 = img*2**-0.5
save_name_1 = save_path + '63989_exp1.png'
cv2.imwrite(save_name_1, img_exp_1*255)

img_exp_2 = img*2**1.0
save_name_2 = save_path + '63989_exp3.png'
cv2.imwrite(save_name_2, img_exp_2*255)


