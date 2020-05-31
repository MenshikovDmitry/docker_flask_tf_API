import numpy as np
import os
import urllib.request
import logging

#from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.models import load_model


class Predictor:
    def __init__(self):
        self.model_name="truck_detector"
        self.model_url="https://dimafrankfurtbucket.s3.eu-central-1.amazonaws.com/public/truck_model.h5"
        self.threshold=0.6
        self.input_size=(244,244)

        logging.info ("Tensorflow version: "+str(tf.__version__))
        from tensorflow.python.client import device_lib
        for device in device_lib.list_local_devices():
            logging.info(device.physical_device_desc)
        home_dir=os.environ.get("HOME")
        model_dir=os.path.join(home_dir,'models_cache')
        if not os.path.exists(model_dir):os.mkdir(model_dir)

        self.model_file=os.path.join(model_dir,self.model_name+'.h5')
        self.load_model()
        

    def load_model(self):

        if os.path.exists(self.model_file):
            logging.info('loading model from file')
            self.model = load_model(self.model_file)
        else:
            logging.info("No model file. Downloading..")
            try:
                urllib.request.urlretrieve(self.model_url, self.model_file)
            except Exception as e:
                logging.critical('Unable to download the model from URL:'+str(e))
            logging.info('model has been successfully downloaded. saved to '+self.model_file)
            self.model=self.model = load_model(self.model_file)
            logging.info("loaded from file")

    def update_model(self):
        logging.info("removing model file")
        os.remove(self.model_file)
        logging.info("reloading the model")
        self.load_model()
        return 0

    def predict(self,input_image):

        img_data = image.img_to_array(input_image)
        img_data = np.expand_dims(img_data, axis=0)        
        img_data = preprocess_input(img_data.astype('float32'))
        
        confidence=self.model.predict(img_data)[0][0]
        conf=str(round(confidence,2))
        
        label ="TRUCK" if confidence>self.threshold else "not truck"

        output={'prediction': label,
                'score': conf}

        return output