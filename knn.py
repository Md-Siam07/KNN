# import matplotlib.pyplot as plt

# img = plt.imread('orig.png')
# rows,cols,colors = img.shape # gives dimensions for RGB array
# img_size = rows*cols*colors
# img_1D_vector = img.reshape(img_size)
# # you can recover the orginal image with:
# img2 = img_1D_vector.reshape(rows,cols,colors)

# plt.imshow(img) # followed by 
# plt.show() # to show the first image, then 
# plt.imshow(img2) # followed by
# plt.show() # to show you the second image.

from msilib.schema import Directory
from pickle import LONG4
from pickletools import long4
import pandas as pd
import os
import math
import numpy as np
import cv2
try:
    import Image
except ImportError:
    from PIL import Image

fireFolder = os.listdir('fire')
iceFolder = os.listdir('ice')

n_fire_image = len(fireFolder)
n_ice_image = len(iceFolder)

iceVector = []
fireVector = []

for _ in range(0, n_fire_image):
    fire = 'fire/'+fireFolder[_]
    ice = 'ice/'+iceFolder[_]
    fireImg = Image.open(fire)
    fireImg = fireImg.resize((200,200))
    fireImageSequence = fireImg.getdata()
    imageArray = np.array(fireImageSequence)
    #print(imageArray)
    imageVector = imageArray.flatten()
    #print(imageArray)
    fireVector.append(imageVector)

    iceImg = Image.open(ice)
    iceImg = iceImg.resize((200,200))
    iceImgSequence = iceImg.getdata()
    imageArray = np.array(iceImgSequence)
    imageVector = imageArray.flatten()
    iceVector.append(imageVector)

testImg = Image.open('test2.png')
testImg = testImg.resize((200,200))
testImgSequence = testImg.getdata()
testImgArray = np.array(testImgSequence)
testImgVector = testImgArray.flatten()

# for _ in range (0, n_fire_image):
#    print(len(fireVector[_]))

# for _ in range (0, n_fire_image):
#     print(len(iceVector[_]))

distanceWithFire =  []
distanceWithIce = []

for i in range (0, len(fireVector)):
    distance = 111111111111111111111111111111111111111111111111111111111111111111
    for j in range (min(len(testImgVector), len(fireVector[i]))):
        distance += (testImgVector[j] - fireVector[i][j])*(testImgVector[j] - fireVector[i][j])
        #if(j==0):
            #distance -= 111111111111111111111111111111111111111111111111111111111111111111
        #print(distance)
    distance -= 111111111111111111111111111111111111111111111111111111111111111111
    #print(distance)
    distance = math.sqrt(distance)
    distanceWithFire.append(distance)

for i in range (0, len(iceVector)):
    distance = 111111111111111111111111111111111111111111111111111111111111111111
    for j in range (min(len(testImgVector), len(iceVector[i]))):
        distance += (testImgVector[j] - iceVector[i][j])*(testImgVector[j] - iceVector[i][j])
    distance -= 111111111111111111111111111111111111111111111111111111111111111111
    #print(distance)
    distance = math.sqrt(distance)
    distanceWithIce.append(distance)

distanceWithFire.sort()
distanceWithIce.sort()

k = int(input("enter the value for k: "))
i=0
j=0
fireCount = 0
iceCount = 0

for itr in range(0,k):
    if(distanceWithFire[i]< distanceWithIce[j]):
        fireCount += 1
        i += 1
    else:
        iceCount +=1
        j+=1

if iceCount >fireCount:
    print('input image is ice')
else:
    print("input image is fire")

    





