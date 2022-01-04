import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))
        self.scaler_dir = 'vars/scalers'
        self.scaler = joblib.load(os.path.join(path, 'vars/scalers', 'min_max_scaler.gz'))
        self.window = 300
        self.telescope = 72
        self.reg_telescope = 864

    def regressive_predict(self, reg_telescope, X_temp_in, telescope):
        predictions= np.array([])
        X_temp = X_temp_in
        for _ in range(0,reg_telescope,telescope):
            pred_temp = self.model.predict(X_temp)
            if(len(predictions)==0):
                predictions = pred_temp
            else:
                predictions = np.concatenate((predictions, pred_temp), axis=1)
            X_temp = np.concatenate((X_temp[:, telescope: , :], pred_temp), axis=1)
            #this way X_temp always has the same shape
        return predictions 

    def predict(self, X):
        #take the final slice from the end of the sequence that we want to forecast
        X = X[-self.window:]

        #preprocess the data
        X = self.scaler.transform(X)

        #add the batch dimesion = 1
        X=np.expand_dims(X, axis=0)

        # predict the output sequence
        out = self.regressive_predict(self.reg_telescope, X, self.telescope)

        # Insert your postprocessing here
        #invert the preprocessing transform
        out = np.squeeze(out, axis=0)
        out = self.scaler.inverse_transform(out)
        
        # output MUST BE TENSOR
        out = tf.convert_to_tensor(out)
        return out