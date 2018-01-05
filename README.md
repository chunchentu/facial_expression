# Train the facial expression dataset using resnet50 provided by keras

## Dataset

The data can be downloaded from [here].(https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

There are 28709 training samples, 3589 validation samples and 3589 testing samples. Each image is a 48x48 grayscale image corresponding to one of the seven classes (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). Please note the minority portion of the class *Disgust*.

## Preprocess data

There are two ways to preprocess the data, depends on how you want to train the model: in-memory v.s. ImageDataGenerator.

### In-memory
The script `setup_facial.py` provides a class that reads in the data. To use it simply import the class and initialize it:

```python
from setup_facial import FACIAL
data = FACIAL()
```

There are 6 attributes: `data.train_data`, `data.train_labels`, `data.validation_data`, `data.validation_labels`, `data.test_data`, `data.test_labels`. All these are numpy arrays with shape `(dataNum, 200, 200, 1)` for `xxx_data` (grayscale images) and `(dataNum, 7)` for one-hot encoding labels `xxx_labels`. Images are rescaled to 200x200 (originally 48x48) to fulfill the smallest image size requirement of ResNet50. In addition, the pixel values are normalized to -0.5 to 0.5.

### ImageDataGenerator

[ImageDataGenerator](https://keras.io/preprocessing/image/) is a powerfull API provided by Keras to augment image data. Before training the model using ImageDataGenerator, it is required to put data into correct folder structure (see an [example](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)). The python script `preprocess_facial_data.py` extracts images and save them to fit the requirement. Simply run

```python
python3 preprocess_facial_data.py
```

This will create a folder `facial_imgs` and sub-folders for training/validation/testing data. Each image will be stored under its corresponding class label.

## Train ResNet50

We use [ResNet50](https://keras.io/applications/#resnet50) provided by Keras. You can train the model with in-memory setup with

```python
python3 train_resnet50.py
```

Similarly, you can train the model using ImageDataGenerator support by 

```python
python3 train_resnet50_imagegen.py
``` 

For both, the model is trained on the images that are normalized to -0.5 to 0.5. Data augmentation is adopted for the second option and thus it is more recommended. We train the model using NVIDIA GeForce GTX 1080 Ti and it takes about 4 minutes on each epoch for both options. 


# Model performance (2018/01/05)

This is the best model we have achieved so far.

|            | Accuracy | Precision | Recall | F1    |
|------------|----------|-----------|--------|-------|
| Training   | 72.95    | 62.77     | 61.53  | 59.51 |
| Validation | 58.04    | 48.57     | 48.13  | 46.10 |
| Testing    | 58.90    | 49.19     | 48.85  | 46.86 |

