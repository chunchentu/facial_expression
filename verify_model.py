from setup_facial import FACIAL, FACIALModel
from keras.models import Model, model_from_json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import numpy as np
import warnings
import argparse
from keras.preprocessing.image import ImageDataGenerator


warnings.filterwarnings("ignore")

def get_metric(class_true, class_pred):
	accuracy = accuracy_score(class_true, class_pred)
	precision = precision_score(class_true, class_pred, average="macro")
	recall = recall_score(class_true, class_pred, average="macro")
	f1 = f1_score(class_true, class_pred, average="macro")

	print("Accuracy:{:.4f}, Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}". format(accuracy, precision, recall, f1))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_weights", default=None, help="the path of the model weights")
	parser.add_argument("--image_size", default=48, type=int, choices=[48, 200])
	args = vars(parser.parse_args())
	if args["model_weights"] is None:
		raise Exception("You should specify the model weights for prediction")

	if not os.path.exists(args["model_weights"]):
		raise Exception("Model weight file does not exist:{}".format(args["model_weights"]))

	data, model = FACIAL(resize=args["image_size"]), FACIALModel(args["model_weights"], use_log=False, image_size=args["image_size"])

	train_true = np.argmax(data.train_labels, axis=1)
	train_pred = np.argmax(model.model.predict(data.train_data), axis=1)
	print("Train data:")
	get_metric(train_true, train_pred)

	validation_true = np.argmax(data.validation_labels, axis=1)
	validation_pred = np.argmax(model.model.predict(data.validation_data), axis=1)
	print("Validation data:")
	get_metric(validation_true, validation_pred)


	test_true = np.argmax(data.test_labels, axis=1)
	test_pred = np.argmax(model.model.predict(data.test_data), axis=1)
	print("Test data:")
	get_metric(test_true, test_pred)
