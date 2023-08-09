from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

# Define directories
train_dir = "bladder_data/train"
validation_dir = "bladder_data/validation"
test_dir = "bladder_data/test"

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Use ImageDataGenerator.flow_from_directory() to load the images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(1080, 1730),
    batch_size=20,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(1080, 1730),
    batch_size=20,
    class_mode='categorical')

# Create the model
model = Sequential()

# Add layers to the model
model.add(Conv2D(8, (3, 3), padding="same", input_shape=(1080, 1730, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
num_classes = len(train_generator.class_indices)
model.add(Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0005), metrics=['accuracy'])

# Training model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

# Save the model
model.save("model.h5")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predicting on test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(1080, 1730),
    batch_size=20,
    class_mode='categorical',
    shuffle=False)  # keep data in same order as labels

# Calculate the test accuracy
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

# Making prediction
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Comparison of prediction and actual results
print("Originally: ", test_generator.classes[50])
print("Predicted: ", predicted_classes[50])

# Visualize an image from the test set
img_path = '/path/to/test/image'  # put here the path of an image from the test set
img = image.load_img(img_path, target_size=(1080, 1730))
plt.imshow(img)
plt.show()
