import numpy as np
import os
import re
import scipy.misc

'''
The original image size is 48x48 and is resize to 200x200 to fit the 
smallest input size of resnet50.
'''

class FACIAL:
	def __init__(self):
		filename = "fer2013"
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
				temp_label = int(label)
				label = np.zeros(7)
				label[temp_label] = 1

				# load image
				pixel_value = [int(x)/255-0.5 for x in pixel.split()]
				data = np.reshape(pixel_value, (48, 48))
				data = scipy.misc.imresize(data, (200, 200))

				usage = usage.rstrip()
				if usage == "Training":
					train_data.append(data)
					train_labels.append(label)

				elif usage == "PublicTest":
					validation_data.append(data)
					validation_labels.append(label)

				elif usage == "PrivateTest":
					test_data.append(data)
					test_labels.append(label)

				else:
					print("End at index index:{}".format(count))
					break
				
		self.train_data = np.array(train_data)
		self.train_labels = np.array(train_labels)
		self.validation_data = np.array(validation_data)
		self.validation_labels = np.array(validation_labels)
		self.test_data = np.array(test_data)
		self.test_labels = np.array(test_labels)