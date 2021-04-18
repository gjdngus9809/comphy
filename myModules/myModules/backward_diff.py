
# coding: utf-8

# In[ ]:


# From the backward difference tables Table 5.3b:
def backward_diff(xData,fxData,xi,h):
    for i in range(len(xData)):
        if xData[i] == xi:
            fx = fxData[i] 
            fxh = fxData[i-1] 
            fx2h = fxData[i-2]
            fx3h = fxData[i-3]
            fd = (3*fx-4*fxh+1*fx2h)/(2*h)
            fdd = (2*fx-5*fxh+4*fx2h-1*fx3h)/(h**2)
    return fd,fdd

