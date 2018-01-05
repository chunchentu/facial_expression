import numpy as np
import os
import re
import scipy.misc
from PIL import Image
'''
The original image size is 48x48 and is resize to 200x200 to fit the 
smallest input size of resnet50.
'''
filename = "fer2013/fer2013.csv"
if not os.path.exists(filename):
	print("Facial data not found: {}".format(filename))
	print("You can download it from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
	

# load data
count = 0
train_data = []
train_labels = []
validation_data = []
validation_labels = []
test_data = []
test_labels = []

if not os.path.exists("facial_imgs"):
	os.mkdir("facial_imgs")
	os.mkdir("facial_imgs/train_dir")
	os.mkdir("facial_imgs/validation_dir")
	os.mkdir("facial_imgs/test_dir")
	for i in range(7):
		os.mkdir("facial_imgs/train_dir/{}".format(i))
		os.mkdir("facial_imgs/test_dir/{}".format(i))
		os.mkdir("facial_imgs/validation_dir/{}".format(i))

with open(filename, "r") as f:
	for l in f:
		count += 1
		# skip the first header
		if count == 1:
			continue
		# extract columns
		try:
			label, pixel, usage = l.split(",")
		except ValueError:
			print("End at index index:{}".format(count))
			break

		# use one-hot encoding for labels
		label = int(label)

		# load image
		pixel_value = [int(x) for x in pixel.split()]
		data = np.reshape(pixel_value, (48, 48))
		data = scipy.misc.imresize(data, (200, 200))
		usage = usage.rstrip()
		if usage == "Training":
			img_name = "facial_imgs/train_dir/{}/{}_{}.png".format(label, label, count)

		elif usage == "PublicTest":
			img_name = "facial_imgs/validation_dir/{}/{}_{}.png".format(label, label, count)

		elif usage == "PrivateTest":
			img_name = "facial_imgs/test_dir/{}/{}_{}.png".format(label, label, count)

		else:
			print("End at index index:{}".format(count))
			break

		img = Image.fromarray(data)
		img.save(img_name)