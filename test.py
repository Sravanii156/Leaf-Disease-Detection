import cv2
import os
import numpy as np
from twilio.rest import Client
import schedule
import datetime

IMG_SIZE=50
LR=1e-3

account_sid='AC167f35105be04139dcdffac84bb41d31'
auth_token='d946954600bc7b9477b893d638a90e23'

MODEL_NAME ='healthyvsunhealthy-{}-{}.model'.format(LR,'2conv-basic')

import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

convnet =input_data(shape=[None,IMG_SIZE,IMG_SIZE,3],name='input')

convnet =conv_2d(convnet,32,3,activation='relu')
convnet =max_pool_2d(convnet,3)

convnet =conv_2d(convnet,64,3,activation='relu')
convnet =max_pool_2d(convnet,3)

convnet =conv_2d(convnet,128,3,activation='relu')
convnet =max_pool_2d(convnet,3)

convnet =conv_2d(convnet,32,3,activation='relu')
convnet =max_pool_2d(convnet,3)

convnet =conv_2d(convnet,64,3,activation='relu')
convnet =max_pool_2d(convnet,3)

convnet =fully_connected(convnet,1024,activation='relu')
convnet =dropout(convnet,0.8)

convnet =fully_connected(convnet,4,activation='softmax')
convnet =regression(convnet,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')

model =tflearn.DNN(convnet,tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('model loaded')

img=cv2.imread('h.JPG',cv2.IMREAD_COLOR)
img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
img=np.asarray(img,np.float32)

data=img.reshape(IMG_SIZE,IMG_SIZE,3)
model_out=model.predict([data])[0]
print(model_out)
print(max(model_out))

if max(model_out) == model_out[0]:
	Client=Client(account_sid,auth_token)
	Client.messages.create(to='+919490206263',
        from_='+18317041254',
        body="LEAF IS HEALTHY")
	print("HEALTHY")
elif max(model_out) == model_out[1]:
	Client=Client(account_sid,auth_token)
	Client.messages.create(to='+919490206263',
        from_='+18317041254',
        body="LEAF WITH BACTERIAL SPOT")
	print("BACTERIAL SPOT")
elif max(model_out) == model_out[2]:
	Client=Client(account_sid,auth_token)
	Client.messages.create(to='+919490206263',
        from_='+18317041254',
        body="LEAF WITH YELLOW COLOR VIRUS")
	print("virus")
elif max(model_out) == model_out[3]:
	Client=Client(account_sid,auth_token)
	Client.messages.create(to='+919490206263',
        from_='+18317041254',
        body="LEAF WITH LATEBLIGHT DISEASE")
	print("LATEBLIGHT")
