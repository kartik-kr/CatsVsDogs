import cv2
import tensorflow as tf
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TEST_DIR = "D:/projects/CatsVsDogs/test/test"
TRAIN_DIR = "D:/projects/CatsVsDogs/train/train"
IMG_SIZE = 50
Alpha = 0.001

MODEL_NAME = "dogsvscats{}.{}.model".format(Alpha, '2layerconvolution')
def label_img(img):
	word_label = img.split(".")[-3]
	## cat,dog
	if word_label == 'cat':
		return [1,0]
	elif word_label == 'dog':
		return [0,1]

def create_train_data():
	train_data = []
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label = label_img(img)
		path =os.path.join(TRAIN_DIR,img)
		img =cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
		train_data.append([np.array(img), np.array(label)])
	shuffle(train_data)
	np.save('D:/projects/CatsVsDogs/train_data.npy', train_data)
	return train_data
#print(len(create_train_data()))

def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR,img)
		img_num = img.split('.')[0]
		img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img), img_num])
	np.save('D:/projects/CatsVsDogs/test_data.npy',testing_data)
	return testing_data 
#train_data = create_train_data()
#if the train model exists then
train_data = np.load('D:/projects/CatsVsDogs/train_data.npy')
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE,IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 512, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 512, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 512, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=Alpha, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
os.system("cls")
print(model)

if os.path.exists('{}.meta'.format(MODEL_NAME) ):
	model.load(MODEL_NAME)
	print ("MODEL LOADED!!")

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train] 

test_X = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_Y = [i[1] for i in test] 

model.fit({'input' : X}, {'targets' : Y}, n_epoch=15, validation_set = ({'input': test_X}, {'targets' : test_Y}), 
	snapshot_step = 500, show_metric = True, run_id = MODEL_NAME )
model.save(MODEL_NAME)


import matplotlib.pyplot as plt 

test_data= process_test_data()

fig = plt.figure()
for num , data in enumerate(test_data[:12]):
	img_num = data[1]
	img_data = data[0]
	y =fig.add_subplot(3,4,num+1)
	orig = img_data
	data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)

	model_out = model.predict([data])[0]
	if np.argmax(model_out)  == 1: str_label = 'Dog'
	else: str_lavel ='Cat'
	y.imshow(orig, cmap='gray')
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()

with open('submission-file.csv','w') as f:
	f.write('id,label\n')
with open('submission-file.csv','a') as f:
	for data in tqdm(test_data):
		img_num = data[1]
		img_data =data[0]
		orig = img_data
		data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
		model_out = model.predict([data])[0]
		f.write('{},{}\n'.format(img_num, model_out[1]))
f.close()