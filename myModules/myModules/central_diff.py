
# coding: utf-8

# In[ ]:


#From the central difference approximations in Table 5.1
  
def central_diff(xData,fxData,xi,h):
    for i in range(len(xData)):
        if xData[i] == xi:
            fx = fxData[i] 
            fxph = fxData[i+1] 
            fxmh = fxData[i-1]
            fd = (-1*fxmh+0*fx+1*fxph)/(2*h)
            fdd = (1*fxmh-2*fx+1*fxph)/(h**2)
    return fd,fdd 

