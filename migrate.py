import tensorflow as tf

print("Loading old Keras 2 H5 model...")
model = tf.keras.models.load_model("model/skin_tone_model.h5", compile=False)
print("Saving migrated model as Keras 3 .keras format...")
model.save("model/skin_tone_model.keras")
print("Migration successful!")
