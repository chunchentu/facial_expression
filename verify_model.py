from keras.models import Model, model_from_json
from setup_facial import FACIAL
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import numpy as np
import warnings
from keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")

# load model
json_file_name = "resnet50_model.json"
if not os.path.isfile(json_file_name):
	raise Exception("The model specification file is missing:{}".format(json_file_name))
json_file = open(json_file_name, "r")
model = model_from_json(json_file.read())

weight_file_name = "resnet50_weights.h5"
if not os.path.isfile(weight_file_name):
	raise Exception("The model weights file is missing:{}".format(weight_file_name))
model.load_weights(weight_file_name)

data = FACIAL()

def get_metric(class_true, class_pred):
	accuracy = accuracy_score(class_true, class_pred)
	precision = precision_score(class_true, class_pred, average="macro")
	recall = recall_score(class_true, class_pred, average="macro")
	f1 = f1_score(class_true, class_pred, average="macro")

	print("Accuracy:{:.4f}, Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}". format(accuracy, precision, recall, f1))

train_true = np.argmax(data.train_labels, axis=1)
train_pred = np.argmax(model.predict(data.train_data), axis=1)
print("Train data:")
get_metric(train_true, train_pred)

validation_true = np.argmax(data.validation_labels, axis=1)
validation_pred = np.argmax(model.predict(data.validation_data), axis=1)
print("Validation data:")
get_metric(validation_true, validation_pred)


test_true = np.argmax(data.test_labels, axis=1)
test_pred = np.argmax(model.predict(data.test_data), axis=1)
print("Test data:")
get_metric(test_true, test_pred)