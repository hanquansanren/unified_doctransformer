# -*- coding: utf-8 -*-
import pickle
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def check_vis(idx, im, lbl):
    '''
    im : distorted image   # HWC 
    lbl : fiducial_points  # 61*61*2 
    '''
    im=np.uint8(im)
    im=im[:,:,::-1]
    h=im.shape[0]*0.01
    w=im.shape[1]*0.01
    im = Image.fromarray(im)
    im.convert('RGB').save("./data_vis/img_{}.png".format(idx))
    
    # fig= plt.figure(j,figsize = (6,6))
    # fig, ax = plt.subplots(figsize = (10.24,7.68),facecolor='white')
    fig, ax = plt.subplots(figsize = (w,h),facecolor='white')
    ax.imshow(im)
    ax.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
    ax.axis('off')
    plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
    # plt.tight_layout()
    plt.savefig('./synthesis_code/test/kk_{}.png'.format(idx))
    plt.close()


if __name__ == '__main__':
    env = lmdb.Environment('./dataset/biglmdb/merged_0.lmdb')
    txn = env.begin()
    print(env.stat())
    new_db = lmdb.Environment('./dataset_fast_train/merged_0_small.lmdb', map_size = 1099511627776)
    nex_txn = new_db.begin(write=True)
    for i, (key, value) in enumerate(txn.cursor()):
        if i<96:
            print(key)
            nex_txn.put(key, value)
        else: 
            break
    nex_txn.commit()
    print(new_db.stat())
    env.close()
    new_db.close()










