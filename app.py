import os,datetime
import logging

home_dir = os.environ.get("HOME")
logfile=os.path.join(home_dir,'log.txt')
backlog_file=os.path.join(home_dir,'backlog.txt')

logging.basicConfig(level = logging.DEBUG,filename=logfile,
                    format = u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s')
logger=logging.getLogger()
logger.info("=========================================================")


# Keras
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,flash,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


#ReNet
#from models.resnet50_predictor import Predictor

#Truck detector
from models.truck_detector import Predictor


# Define a flask app
app = Flask(__name__)

#loading predictor
predictor=Predictor()

logger.info('Running............')


def process_files(request):
    '''
    extracts files from the request. saving files to ../uploads.
    returns a dictionary
    '''
    filenames={}
    uploads_dir=os.path.join(home_dir,"uploads")
    if not os.path.exists(uploads_dir):os.mkdir(uploads_dir)

    now = datetime.datetime.now()
    for filename, file in request.files.items():
        file_path = os.path.join(uploads_dir, secure_filename(filename+str(now)[:19]+".jpg"))
        file.save(file_path)
        filenames[filename]=file_path
    return filenames


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('upload.html')



@app.route('/test', methods=['GET'])
def test():
    # test page
    return 'I am alive'

@app.route('/log', methods=['GET'])
def log():
    # return log file
    with open(logfile,'r') as f:
        s=f.readlines()
    #last 100 lines of log file in. last events - on top
    s=s[-100:]
    s=s[::-1]
    s='<br>'.join(s)
    return s

@app.route('/v', methods=['GET'])
def version():
    # return backlog file
    with open(backlog_file,'r') as f:
        s=f.readlines()
    #first 100 lines of log file in. last events - on top
    s=s[:100]
    s='<br>'.join(s)
    return s


@app.route('/predict/istruck', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        filenames=process_files(request)
        if len(filenames)==0:
            logger.warning("No files attached")
            return jsonify({'error':"No files attached"})
        results={}
        for fname,file_path in filenames.items():
            #loading each file from the request and predict labels
            try:
                img = image.load_img(file_path, target_size=predictor.input_size)
            except Exception as e:
                logger.error("Error loading file "+ file_path)
                return jsonify({'error':"error loading file. is it a picture???"})
            results[fname]=predictor.predict(img)
            logger.info("{}: {}".format(fname,results[fname]))
            #remove the file from the disk
            os.remove(file_path)
            #print(result_print)
        return jsonify(results)
    logger.info("GET request")
    return jsonify({'error':"GET request is not supported"})

@app.route('/update',methods=['GET'])
def update():
    result=predictor.update_model()
    if result==0:
        return 'success'
    if result ==1:
        return 'Unable to update'


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()