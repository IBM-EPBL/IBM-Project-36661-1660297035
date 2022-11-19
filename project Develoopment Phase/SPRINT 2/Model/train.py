from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings

warnings.filterwarnings('ignore')


def train(path, model_name):
    train_path = path + '/training'
    valid_path = path + '/validation'
    train_data_generator = ImageDataGenerator(rescale=1 / 255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    test_data_generator = ImageDataGenerator(rescale=1 / 255)
    x_train_generator = train_data_generator.flow_from_directory(train_path, target_size=(224, 224), batch_size=10,
                                                                 class_mode='categorical')
    x_test_generator = test_data_generator.flow_from_directory(valid_path, target_size=(224, 224), batch_size=10,
                                                               class_mode='categorical')
    vgg16 = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    for layer in vgg16.layers:
        layer.trainable = False
    x = Flatten()(vgg16.output)
    prediction = Dense(3, activation='softmax')(x)
    model = Model(vgg16.input, prediction)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(generator=x_train_generator, steps_per_epoch=len(x_train_generator), epochs=25,
                        validation_data=x_test_generator, validation_steps=len(x_test_generator))
    model.save(model_name)


train(os.getcwd()+'/Dataset/body','bodyl.h5')
train(os.getcwd() + 'Dataset/level', 'levell.h5')


# def detect(model, frame, labels):
#     img = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
#     if np.max(img) > 1:
#         img = img / 225.0
#     img = np.array([img])
#     prediction = model.predict(img)
#     return labels[np.argmax(prediction)]
#
#
# body_model = load_model(os.getcwd() + '/body.h5')
# level_model = load_model(os.getcwd() + '/level.h5')

# body_labels = ['Front', 'Rear', 'Side']
# level_labels = ['Minor', 'Moderate', 'Severe']
# data = "/home/siva/Documents/bro/Damage/testimage.jpg"
# image = cv2.imread(data)
# print(detect(model=body_model, frame=image, labels=body_labels))
# print(detect(model=level_model, frame=image, labels=level_labels))