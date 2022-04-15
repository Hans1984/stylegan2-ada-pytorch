import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'D:/github/stylegan2-ada-pytorch/out_exp0/projected_w.npz'
path_exp_plus = 'D:/github/stylegan2-ada-pytorch/out_exp+1/projected_w.npz'
path_exp_minus = 'D:/github/stylegan2-ada-pytorch/out_exp-1/projected_w.npz'

data = np.load(path)
data_exp_plus = np.load(path_exp_plus)
data_exp_minus = np.load(path_exp_minus)
data_latent = data['w']
data_latent_exp_plus = data_exp_plus['w']
data_latent_exp_minus = data_exp_minus['w']
# print(data.files)
# print(data['w'])
# print(data['w'].shape)

for i in range(0,1):
    #plt.imshow(data_latent[:,i,:])
    #cv2.imwrite(str(i)+".png",image[i,:,:])
    #plt.show()
    diff = data_latent_exp_minus[:,i,:] - data_latent[:,i,:]
    #diff = data_latent[:,i,:] - data_latent_exp_plus[:,i,:]
    diff = np.reshape(diff, [512])
    print(diff)
    dataframe = pd.DataFrame({'diff_exp0_exp-1': diff})
    dataframe.to_csv("test.csv",index=False,sep=',')