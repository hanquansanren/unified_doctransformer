# -*- coding: utf-8 -*-
import pickle
from tkinter import N
import cv2
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon
import time


def check_vis(idx, im, lbl=None):
    '''
    im : distorted image   # HWC 
    lbl : fiducial_points  # 61*61*2 
    '''
    im=np.uint8(im)
    im=im[:,:,::-1]
    h=im.shape[0]*0.01
    w=im.shape[1]*0.01
    im = Image.fromarray(im)
    im.convert('RGB').save("./simple_test/interpola_vis/img_{}.png".format(idx))
    
    if lbl is not None:
        # fig= plt.figure(j,figsize = (6,6))
        # fig, ax = plt.subplots(figsize = (10.24,7.68),facecolor='white')
        fig, ax = plt.subplots(figsize = (w,h),facecolor='white')
        ax.imshow(im)
        # ax.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
        ax.scatter(lbl[:,0].flatten(),lbl[:,1].flatten(),s=1.2,c='red',alpha=1)
        ax.axis('off')
        plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
        # plt.tight_layout()
        plt.savefig('./simple_test/interpola_vis/kkpt_{}.png'.format(idx))
        plt.close()


if __name__ == '__main__':
    env_db = lmdb.Environment('./dataset_for_debug/warp_for_debug.lmdb')
    # env_db = lmdb.open("./test.lmdb")
    txn = env_db.begin()
    print(env_db.stat()) 

    for i,(key, value) in enumerate(txn.cursor()):
        if i<(pickle.loads(txn.get(b'__len__'))):
            print(key)
            value=pickle.loads(value)
            if key[-2:].decode()=='d1' or key[-2:].decode()=='d2':
                pt_edge = value['label'][0,:,:] 
                for num in range(1,60,1):
                    pt_edge=np.append(pt_edge, value['label'][num,60,:][None,:] ,axis=0)
                pt_edge = np.vstack((pt_edge,value['label'][60,:,:][::-1,:]))
                for num in range(59,0,-1):
                    pt_edge=np.append(pt_edge, value['label'][num,0,:][None,:] ,axis=0)
                
                # check_vis(key, value['image'], pt_edge) # (240,2)
                img = np.zeros((1024, 768, 3), dtype=np.int32)
                pts = pt_edge.round().astype(int)
                # x, y = pts[:, 0], pts[:, 1]

                a = cv2.fillPoly(img, [pts], (255, 255, 255))
                cv2.imwrite('./simple_test/interpola_vis/pt_{}.png'.format(key), a)

                # start = time.time()
                # print(Polygon(pts).area)
                # end = time.time()
                # print("多边性面积计算时间：", end - start)






    env_db.close()