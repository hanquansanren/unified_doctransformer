import re
import os


print(divmod(11,2))

#################################################################
# a=min([os.cpu_count(), 24 if 24 > 1 else 0, 8])
# print(os.cpu_count())
# print(a)

#################################################################
# print(list(map(int,'01'))) #字符串转列表
# print(list(map(int,['1', '3']))) #字符串转列表

# s='012'
# ss=''
# for ch in range(len(s)):
#     ss+=s[ch]+',' if ch<(len(s)-1) else ''

# print(ss)
###############################################################
# #r(raw) #用在pattern之前，表示单引号中的字符串为原生字符，不会进行任何转义
# d=re.match(r'l','liuyan1').group()  #返回l
# e=re.match(r'y','liuyan1')  #返回None

# f=re.search(r'y','liuyan1y').group(0) #返回y
# print(d,e,f,sep='\n')

###############################################################

def demo(data) ->list:
    # raise BaseException("data is None") if not data else data
    # assert data,'please choice optimizer,error'
    assert 'please choice optimizer,error'
    exit()
    exit('error')
    print("*"*30)
 
demo([])

###############################################################
# import numpy as np
# class Test3:
#     def __init__(self,aa=88):
#         self.x=6
#         self.y=8
#         self.aa=aa
#     def square_a(self,a):
#         result=pow(a,2)
#         return result
#     def sqrt_sum(self,a,b,c,d):
#         a = self.square_a(a)
#         b = self.square_a(b)
#         c = self.square_a(self.aa)
#         d = self.square_a(self.y)
#         print(np.sqrt(a+b),np.sqrt(c+d))

# t3=Test3()
# t3.sqrt_sum(3,4,None,None)