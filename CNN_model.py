import tensorflow as tf
import keras

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
    
    tf.keras.layers.Dense(128, activation='relu'), ## 128 neurons 
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'), ## 64 neurons 
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(27, activation='softmax')
])
optimize = tf.keras.optimizers.Adam(learning_rate=0.00002)
model.compile(loss = 'categorical_crossentropy', optimizer = optimize, metrics= ['accuracy'])
model.summary()