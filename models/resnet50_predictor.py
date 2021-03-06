import numpy as np
#import cv2
import os

#from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
from keras.preprocessing import image
from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self):
        self.input_size=(224,224)
        self.model_name='resnet50'
        print ("Tensorflow version:",tf.__version__)
        from tensorflow.python.client import device_lib
        for device in device_lib.list_local_devices():
            print(device.physical_device_desc)
        home_dir=os.environ.get("HOME")
        model_dir=os.path.join(home_dir,'models_cache')
        if not os.path.exists(model_dir):os.mkdir(model_dir)

        self.model_file=os.path.join(model_dir,self.model_name+'.h5')
        #self.model_file='resnet50.h5'

        if os.path.exists(self.model_file):
            print('loading ResNet from file')
            self.model = load_model(self.model_file)
        else:
            print("No model file. Loading..")
            self.model=ResNet50(weights='imagenet', include_top=True)
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
        output={'prediction':[(v[1],str(round(v[2],3))) for v in result]}

        return output