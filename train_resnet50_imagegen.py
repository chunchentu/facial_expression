from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Dropout, LeakyReLU, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


input_layer = Input(shape=(200, 200, 1))
base_model = ResNet50(weights=None, input_tensor = input_layer)
x = base_model.output
x = Dense(128, kernel_regularizer=regularizers.l2(0.01),)(x)
x = Dropout(0.5)(x)
x = LeakyReLU()(x)
x = Dropout(0.7)(x)
logits = Dense(7, kernel_regularizer=regularizers.l2(0.01),)(x)
predictions = Activation("softmax")(logits)
model = Model(inputs=base_model.input, outputs=predictions)


train_datagen = ImageDataGenerator(rescale=1./255,
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
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
						"facial_imgs/validation_dir/",
						color_mode="grayscale",
						target_size=(200, 200),
						batch_size=50,
						class_mode="categorical")

# load checkpoint if exists
if os.path.exists("resnet50.ckpt"):
	print("Loading previous checkpoints")
	model.load_weights("resnet50.ckpt")



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
json_file_name = "resnet50_model.json"
with open(json_file_name, "w") as json_file:
	json_file.write(model_json)
print("Save model spec to {}".format(json_file_name))

weight_file_name = "resnet50_weights.h5"
model.save_weights(weight_file_name)
print("Save model weights to {}".format(weight_file_name))