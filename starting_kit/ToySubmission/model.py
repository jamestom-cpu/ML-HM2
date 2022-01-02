import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X):
        
        # Insert your preprocessing here

        out = self.model.predict(X)

        # Insert your postprocessing here

        return out