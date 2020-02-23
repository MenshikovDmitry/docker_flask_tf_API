import os
import numpy as np

# Keras
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,flash,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# VGG16
from models.vgg16_predictor import VGG16_predictor

# Define a flask app
app = Flask(__name__)

#loading predictor
predictor=VGG16_predictor()

print('Running............')


def process_files(request):
    '''
    extracts files from the request. saving files to ../uploads.
    returns a dictionary
    '''
    filenames={}
    basepath = os.path.dirname(__file__)
    for filename, file in request.files.items():
        file_path = os.path.join(basepath, 'uploads', secure_filename(filename))
        file.save(file_path)
        filenames[filename]=file_path
    return filenames


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('upload.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        filenames=process_files(request)
        results={}
        for fname,file_path in filenames.items():
            #loading each file from the request and predict labels
            img = image.load_img(file_path, target_size=(224, 224))
            results[fname]=predictor.predict(img)
            #print(result_print)
        return jsonify(results)
    return None


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()