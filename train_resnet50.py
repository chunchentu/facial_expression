from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Dropout, LeakyReLU, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import SGD
import numpy as np
from setup_facial import FACIAL
import os

data = FACIAL()

input_layer = Input(shape=(200, 200, 1))
base_model = ResNet50(weights=None, input_tensor = input_layer)
x = base_model.output
x = Dense(128)(x)
x = Dropout(0.2)(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
logits = Dense(7)(x)
predictions = Activation("softmax")(logits)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers:
	layer.trainable = True

model.summary()
# load checkpoint if exists
if os.path.exists("resnet50.ckpt"):
	print("Loading previous checkpoints")
	model.load_weights("resnet50.ckpt")


sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss="categorical_crossentropy",
	metrics=[metrics.categorical_accuracy])
checkpointer = ModelCheckpoint(filepath="resnet50.ckpt", verbose=1, save_best_only=True)


model.fit(data.train_data, data.train_labels,
	batch_size = 90,
	validation_data = (data.validation_data, data.validation_labels),
	epochs = 100,
	shuffle = True,
	callbacks = [checkpointer])

model_json = model.to_json()
json_file_name = "resnet50_model.json"
with open(json_file_name, "w") as json_file:
	json_file.write(model_json)
print("Save model spec to {}".format(json_file_name))

weight_file_name = "resnet50_weights.h5"
model.save_weights(weight_file_name)
print("Save model weights to {}".format(weight_file_name))