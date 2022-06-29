import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

training_dir = 'PATH'
validation_dir = 'PATH'

## Image data generator
train =ImageDataGenerator(
        rescale=1/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

validation = ImageDataGenerator(rescale = 1/255)
size = (300, 300)

train_dataset = train.flow_from_directory(training_dir, target_size = size, batch_size = 32, color_mode = "grayscale")
validation_dataset = validation.flow_from_directory(validation_dir, target_size = size, batch_size = 32, color_mode = "grayscale")

## assign indices for each class
train_dataset.class_indices
validation_dataset.class_indices

## early callback for preventing the overfitting 
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                    min_delta=0, 
                    patience=2,
                    verbose=1,
                    mode='min',
                    restore_best_weights=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
     
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),

    ## fully connected Dense layers 
    tf.keras.layers.Dense(128, activation='relu'), ## 128 neurons 
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'), ## 64 neurons 
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(27, activation='softmax')
])
optimize = tf.keras.optimizers.Adam(learning_rate=0.00002)
model.compile(loss = 'categorical_crossentropy', optimizer = optimize, metrics= ['accuracy'])
model.summary()

history = model.fit(train_dataset,
                    epochs=20, 
                    steps_per_epoch= 1250,              ## steps_per_epochs = total_data/batch_size
                    validation_data=validation_dataset, 
                    verbose=1, 
                    validation_steps= 290, 
                    callbacks = [callback])


## saving the model architecture and weigths

model_json = model.to_json()
with open("model-bw1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model-bw1.h5')

