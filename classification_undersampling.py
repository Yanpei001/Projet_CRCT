import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

# Define parameters
IMAGE_SIZE = (250, 250)
BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS = 20

# Define data generators for training and validation
train_datagen = ImageDataGenerator(
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    '/users/m2ida/m2ida/cancer_du_sein/dataset/image_50000/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/users/m2ida/m2ida/cancer_du_sein/dataset/image_50000/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Calculate steps per epoch for training
train_steps_per_epoch = np.ceil(train_generator.samples / BATCH_SIZE)
validation_steps = np.ceil(validation_generator.samples / BATCH_SIZE)

def standardize_generator(data_generator, data_gen, batch_size):
    scaler = StandardScaler()
    undersampler = RandomUnderSampler()
    
    all_images = []
    all_labels = []
    
    for i in range(len(data_generator)):
        batch_images = data_generator[i][0]
        batch_labels = data_generator[i][1]
        
        standardized_images = scaler.fit_transform(batch_images.reshape(batch_images.shape[0], -1))
        standardized_images = standardized_images.reshape(batch_images.shape)
        
        all_images.append(standardized_images)
        all_labels.append(batch_labels)
    
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    new_generator = data_gen.flow(all_images, all_labels, batch_size=batch_size)
    
    return new_generator

# Usage example
train_generator = standardize_generator(train_generator, train_datagen, BATCH_SIZE)
validation_generator = standardize_generator(validation_generator, train_datagen, BATCH_SIZE)

# Create an EfficientNet model
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=(250, 250, 3)  # EfficientNet expects RGB images
)

for layer in base_model.layers:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # Replace num_classes with your actual number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use 'categorical_crossentropy' if your labels are one-hot encoded
              metrics=['accuracy'])
              

# Callback for saving the model
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}.h5',
    save_freq='epoch'
)

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint_callback]
)

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.show()
# Plot training & validation accuracy values
plt.plot(train_accuracy)
plt.plot(val_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('train_val_loss.pdf', bbox_inches='tight', format='pdf')
plt.close()

# Save the entire model to a HDF5 file
model.save('my_model3.h5')