import numpy as np 
# from scipy import interpolate 
# import pylab as pl 
# import matplotlib as mpl 
 
# def func(x, y): 
#     return (x+y)*np.exp(-5.0*(x**2 + y**2)) 
 
# # X-Y轴分为15*15的网格 
# y,x= np.mgrid[-1:1:15j, -1:1:15j] 
 
# fvals = func(x,y) # 计算每个网格点上的函数值 15*15的值 
# print (len(fvals[0]) )
 
# #三次样条二维插值 
# newfunc = interpolate.interp2d(x, y, fvals, kind='cubic') 
 
# # 计算100*100的网格上的插值 
# xnew = np.linspace(-1,1,198)#x 
# ynew = np.linspace(-1,1,198)#y 
# fnew = newfunc(xnew, ynew)#仅仅是y值 100*100的值 

# # 绘图 
# # 为了更明显地比较插值前后的区别，使用关键字参数interpolation='nearest' 
# # 关闭imshow()内置的插值运算。 
# pl.subplot(121) 
# im1=pl.imshow(fvals, extent=[-1,1,-1,1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")#pl.cm.jet 
# #extent=[-1,1,-1,1]为x,y范围 favals为 
# pl.colorbar(im1) 
 
# pl.subplot(122) 
# im2=pl.imshow(fnew, extent=[-1,1,-1,1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower") 
# pl.colorbar(im2) 
# pl.savefig("./simple_test/interpola_vis/2dinterpo.png")

######################################################################################
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
rng = np.random.default_rng()
x = rng.random(10) - 0.5
y = rng.random(10) - 0.5

aaaa = y[range(1,3)]
print(aaaa)

z = np.hypot(x, y)
X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
interp = LinearNDInterpolator(list(zip(x, y)), z)
Z = interp(X, Y)

plt.pcolormesh(X, Y, Z, shading='auto')
plt.plot(x, y, "ok", label="input point")
plt.legend()
plt.colorbar()
plt.axis("equal")
plt.savefig("./simple_test/interpola_vis/2dinterpo.png")
# plt.show()

###################################################################
# import cv2
# im = np.random.randn(31,31,2)

# im_new = cv2.resize(im, (198,198), interpolation=cv2.INTER_LINEAR) 
# print(im_new.shape)

###################################################################
            # BM[norm_batch_trg_pt]
            # BM_list = []
            # for batch in range(batch_num):
            #     z = np.hypot(norm_batch_trg_pt[batch,:,0], norm_batch_trg_pt[batch,:,1])
            #     X = np.linspace(np.min(norm_batch_trg_pt[batch,:,0]), np.max(norm_batch_trg_pt[batch,:,0]))
            #     Y = np.linspace(np.min(norm_batch_trg_pt[batch,:,1]), np.max(norm_batch_trg_pt[batch,:,1]))
            #     X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
            #     interp = LinearNDInterpolator(list(zip(norm_batch_trg_pt[batch,:,0], norm_batch_trg_pt[batch,:,1])), z)
            #     Z = interp(X, Y)
            #     BM_list.append(Z)