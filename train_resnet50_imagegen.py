from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from setup_facial import FACIAL, FACIALModel
import os
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_weights", default=None, help="the path of the pre-trained model weights")
	parser.add_argument("--save_prefix", default="resnet50", help="the file name prefix for saving the model")
	parser.add_argument("--data_dir", default="facial_imgs")
	args = vars(parser.parse_args())
	data, model = FACIAL(), FACIALModel(restore=args["model_weights"], use_log=True)
	model = model.model

	preprocess_fun = lambda x: x/255.0 - 0.5

	train_datagen = ImageDataGenerator(preprocessing_function=preprocess_fun,
										shear_range=0.2,
										zoom_range=0.2,
										horizontal_flip=True,
										fill_mode="nearest")
	train_generator = train_datagen.flow_from_directory(
						"facial_imgs/train_dir/",
						color_mode="grayscale",
						target_size=(200, 200),
						batch_size=50,
						class_mode="categorical")
	validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_fun)
	validation_generator = validation_datagen.flow_from_directory(
							"facial_imgs/validation_dir/",
							color_mode="grayscale",
							target_size=(200, 200),
							batch_size=50,
							class_mode="categorical")

	model.compile(optimizer="sgd", loss="categorical_crossentropy",
		metrics=[metrics.categorical_accuracy])
	checkpointer = ModelCheckpoint(filepath="resnet50.ckpt", verbose=1, save_best_only=True)
	model.fit_generator(train_generator,
			steps_per_epoch=575,
			epochs=50,
			validation_data=validation_generator,
			validation_steps=72,
			callbacks= [checkpointer])

	model_json = model.to_json()
	json_file_name = "{}_model.json".format(args["save_prefix"])
	with open(json_file_name, "w") as json_file:
		json_file.write(model_json)
	print("Save model spec to {}".format(json_file_name))

	weight_file_name = "{}_weights.h5".format(args["save_prefix"])
	model.save_weights(weight_file_name)
	print("Save model weights to {}".format(weight_file_name))
