from keras.callbacks import ModelCheckpoint
from keras import metrics
import numpy as np
from setup_facial import FACIAL, FACIALModel
import os
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_weights", default=None, help="the path of the pre-trained model weights")
	parser.add_argument("--save_prefix", default="resnet50", help="the file name prefix for saving the model")

	args = vars(parser.parse_args())
	data, model = FACIAL(), FACIALModel(restore=args["model_weights"], use_log=True)
	model = model.model

	model.compile(optimizer="sgd", loss="categorical_crossentropy",
		metrics=[metrics.categorical_accuracy])
	checkpointer = ModelCheckpoint(filepath="resnet50.ckpt", verbose=1, save_best_only=True)


	model.fit(data.train_data, data.train_labels,
		batch_size = 50,
		validation_data = (data.validation_data, data.validation_labels),
		epochs = 30,
		shuffle = True,
		callbacks = [checkpointer])

	model_json = model.to_json()
	json_file_name = "{}_model.json".format(args["save_prefix"])
	with open(json_file_name, "w") as json_file:
		json_file.write(model_json)
	print("Save model spec to {}".format(json_file_name))

	weight_file_name = "{}_weights.h5".format(args["save_prefix"])
	model.save_weights(weight_file_name)
	print("Save model weights to {}".format(weight_file_name))