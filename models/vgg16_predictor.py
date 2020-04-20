import numpy as np
#import cv2
import os

#from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from keras.preprocessing import image
from tensorflow.keras.models import load_model


class VGG16_predictor:
    def __init__(self):
        print ("Tensorflow version:",tf.__version__)
        from tensorflow.python.client import device_lib
        for device in device_lib.list_local_devices():
            print(device.physical_device_desc)

        self.model_file=os.path.join(os.path.dirname(__file__),'VGG16.h5')
        
        if os.path.exists(self.model_file):
            print('loading VGG from file')
            self.model = load_model(self.model_file)
        else:
            print("No model file. Loading..")
            self.model=VGG16(weights='imagenet', include_top=True)
            print('model loaded')
            self.model.save(self.model_file)
            print('model saved to',self.model_file)

    
    def predict(self,input_image):
        #image=np.array(input_image)
        #image=cv2.resize(image,(224,224))
        #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_data = image.img_to_array(input_image)
        #print('img loaded,',img_data.shape)
        img_data = np.expand_dims(img_data, axis=0)        
        img_data = preprocess_input(img_data.astype('float32'))
        
        prediction=self.model.predict(img_data)
        
        pred_class = decode_predictions(prediction)
        result = pred_class[0]
        output=[(v[1],str(round(v[2],3))) for v in result]

        return output
