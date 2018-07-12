import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from facedetector import facedetect
app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATE_AUTO_RELOAD'] = True
model = facedetect() 

@app.route('/')
def hello_world():

    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    fsave = os.path.join(app.config['UPLOAD_FOLDER'], "fuck"+file.filename)
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f) 
    pic,labels = model.predict(f)
    cv2.imwrite(fsave,pic)
    labels = np.unique(labels)
    print(fsave)
    text = "[ "
    for i in labels:
        text+=i+" "
    text+=']'
    return render_template('index.html',paths = fsave,texts = text)
@app.route('/uploads/<path:path>')
def send_img(path):
    return send_from_directory('uploads', path)
@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('templates', path)

if __name__ == '__main__':
    app.run(debug =True)