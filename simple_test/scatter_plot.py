import numpy as np
import matplotlib.pyplot as plt
 
np.random.seed(0)
x=np.random.rand(20)
y=np.random.randn(20)

plt.scatter(x,y,s=30,alpha=0.5)
plt.savefig('./simple_test/scatter{}.png'.format(6))