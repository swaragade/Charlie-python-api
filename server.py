from flask import Flask, jsonify, request, send_file , send_from_directory
import sys
import cv2
import os
from sys import platform
import numpy as np
import socket
from werkzeug.utils import secure_filename
import json
import uuid
from datetime import datetime
from scipy import fftpack
# Heartrate analysis package
import heartpy as hp
from heartpy.exceptions import BadSignalWarning
# Allows for filtering
from scipy import signal
from scipy.signal import butter, lfilter
# For file saving
import time
import array
from datetime import datetime
import csv
import io
from time import sleep
import pyqtgraph as pg
pg.mkQApp()

# Initalize
camData = np.random.normal(size=50)
camBPMData = np.zeros(50)

psData = np.random.normal(size=50)
psBPMData = np.zeros(50)

middleRow = int(5/2)
middleCol = int(5/2)

fs = 100
t = np.linspace(start=0,stop=5.0,num=50)
ptr = 0
filename=""

boxH = int(5*0.15)
boxW = int(5*0.15)

finalBpm = 80
spo = 1

box = pg.RectROI( (middleRow-boxH/2,middleCol-boxW/2), \
    (boxH,boxW), pen=9, sideScalers=True, centered=True)

# creating a Flask app 
app = Flask(__name__)
classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
loc = "D:/FinalCode/covidtracker-ui/public/uploads/"
cropped = "D:/FinalCode/covidtracker-ui/public/uploads/cropped/"


@app.route("/")
def hello():
    return "Hello World!"

@app.route('/cropped/<path>', methods = ['GET'])
def send_js(path):
    return send_from_directory(cropped,path)
   
# route http posts to this method
@app.route('/api/temp', methods=['POST'])
def test():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(cropped, filename))
    
    tmp,avg = tempCalculator(cropped+filename)
    data_set = {"temperature": tmp, "average": avg}
    json_dump = json.dumps(data_set)
    print(json_dump)
    #jsonObj = jsonify(temprature=tmp,average=avg)
    #jsonObj = jsonify({temperature=tmp,average=avg})
    return json_dump
    
@app.route('/api/crop', methods=['POST'])
def crop():
    #file = request.files['file']
    #filename = secure_filename(file.filename)
    #file.save(os.path.join(loc, filename))
    content = request.json
    fileName=cropFromVideoOrg(loc+content['filename'])
    
    bpm,spo,filename = getPPGO(loc+content['filename'])
    data_set = {"filename": fileName, "heartbeat": bpm, "oxygenRate": spo,"csvLoc":filename}
    #data_set = {"heartbeat": bpm, "oxygenRate": spo,"csvLoc":filename}
    json_dump = json.dumps(data_set)
    print(json_dump)
    #jsonObj = jsonify(temprature=tmp,average=avg)
    #jsonObj = jsonify({temperature=tmp,average=avg})
    return json_dump
     

@app.route('/api/bpm', methods=['POST'])
def bpm():
    #file = request.files['file']
    #filename = secure_filename(file.filename)
    #file.save(os.path.join(loc, filename))
    content = request.json
    bpm,spo,filename = getPPGO(loc+content['filename'])
    data_set = {"heartbeat": bpm, "oxygenRate": spo,"csvLoc":filename}
    json_dump = json.dumps(data_set)
    return json_dump

#@app.route('/api/values/<fname>', methods = ['GET', 'POST'])
#def calcTemp(fname):
#    content = request.json
#    tmp,avg = tempCalculator(content['filename'])
#    return jsonify({"temprature":tmp,"average":avg})

#@app.route('/api/crop/<fname>', methods = ['GET', 'POST'])
#def cropImage(fname):
#    content = request.json
#    cropping(content['filename'])
#    return jsonify({"operations":"done"})

def setup_csv(csvStr=None):
    now = datetime.now()
    if csvStr is None:
        csvFileName = 'ppg_'+now.strftime("%Y-%m-%d_%I_%M_%S")
    else:
        csvFileName = csvStr+'_'+now.strftime("%Y-%m-%d_%I_%M_%S")
    headers = (u'ps_time'+','+u'ps_waveform'+','+u'ps_bpm'+','+u'cam_time'+','+u'cam_waveform'+','+u'cam_bpm')
    with io.open(csvFileName + '.csv', 'w', newline='') as f:
        f.write(headers)
        f.write(u'\n')
    return csvFileName

def save_to_csv(csvFileName, data):
    with io.open(csvFileName + ".csv", "a", newline="") as f:
        row = str(data['ps_time'])+","+str(data['ps_pulseWaveform'])+","+str(data['ps_pulseRate'])+"," \
             +str(data['cam_time'])+","+str(data['cam_pulseWaveform'])+","+str(data['cam_bpm'])
        f.write(row)
        f.write("\n")
def getIntensity(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.25, 6)
    col,row = box.pos()
    row = int(row)
    col = int(col)

    x,y = box.size()
    width = int(x)
    height = int(y)
    if(len(faces) !=0):
        row = int(faces[0][0]/2)
        col = int(faces[0][1]/2)
        width = int(faces[0][2]/2)
        height = int(faces[0][3]/2)

    roi = gray[row:row+width, col:col+height]

    # Find intensity (average or median or sum?)
    rowSum = np.sum(roi, axis=0)
    colSum = np.sum(rowSum, axis=0)
    allSum = rowSum + colSum

    intensity = np.median(np.median(allSum))
    return gray, intensity


def getPPGO(filenameVAl):
    global camData, ptr, t, filename,finalBpm,fs,spo
    cap = cv2.VideoCapture(filenameVAl)
    filename=setup_csv()
    spoVal=1
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image, signal=getIntensity(frame)
            sig = camData - np.mean(camData)
            cam_bpm = camBPMData[-1]
            camSig = camData - np.mean(camData)
            try:
                working_data, measures = hp.process(camSig, 10.0)
            except BadSignalWarning:
                print("Bad signal")
            else:
                if(measures['bpm'] > 50 and measures['bpm'] < 120):
                    cam_bpm = measures['bpm']
                    if finalBpm > cam_bpm:
                        finalBpm = cam_bpm
                    if(measures['sd1/sd2'] > .50 and measures['sd1/sd2'] < 1.20):
                        spoVal = measures['sd1/sd2']
                        if spoVal < spo:
                            spo =spoVal
            #print(measures)
            image = image.T[:, ::-1]
            camData[:-1] = camData[1:]
            camData[-1] = signal
            camBPMData[:-1] = camBPMData[1:]
            camBPMData[:-1] = cam_bpm
            single_record = {}
            single_record['ps_time'] = 10
            single_record['ps_pulseRate'] = 10
            single_record['ps_pulseWaveform'] = 10
            single_record['cam_pulseWaveform'] = camData[-1]
            single_record['cam_bpm'] =cam_bpm
            single_record['cam_time'] = t[-1]
            save_to_csv(loc+filename, single_record)
        else:
            break
    return finalBpm,spoVal*100,filename
def tempCalculator(inputImageToProcess):
    try:
        imageToProcess = cv2.imread(inputImageToProcess)
        print('temp calc hit')
        if imageToProcess is None:
            print('imageToProcess is none')
        try:
            h, w, c = imageToProcess.shape
        except:
            h,w,c = 200,400,40
        #processing
        gray = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2GRAY)
        gray_inverted = cv2.bitwise_not(gray)
        imageToProcess = cv2.cvtColor(gray_inverted, cv2.COLOR_GRAY2BGR)
        size_x = int(h)-10
        size_y = int(w)-10
        minVal = size_x < size_y and size_x or size_y
        size_x,size_y = minVal,minVal
        reference_x = 0
        reference_y = 0
        offset_x = 0
        offset_y = 0
        average = 0
        counter = 0

        # Calculate average values in face rect
        for y in range(reference_x-size_x+offset_x, reference_x+size_x+offset_x):
            for x in range(reference_y-size_y+offset_y, reference_y+size_y+offset_y):
                average += imageToProcess[x, y][0]
                counter += 1
        #Calculate average
        if counter!=0:
            average = average / counter
        #Print data
        if counter!=0:
            # Get pixel value of reference point
            reference_temperature = imageToProcess[size_x, size_y][0];

            #Temperature calculation with reference point temperature
            temperature = (average * float(45))/reference_temperature

            #Print some data about temperature
            print("Face rect temperature: T:{0:.2f}C, {1:.2f}".format(temperature, average))

            return temperature,average
    
    except Exception as e:
        print(e)

def cropping(fileName):
    image = cv2.imread(loc+fileName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = np.copy(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = classifier.detectMultiScale(gray_image, 1.25, 6)
    print('Number of faces detected:', len(faces))
    face_crop = []
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
        face_crop.append(gray_image[y:y+h, x:x+w])
    #facelen = 0
    for face in face_crop:
        print("check")
        cv2.imwrite(cropped+fileName,face)
    filename = cropped+fileName
    return send_file(filename, mimetype='image/png',attachment_filename=fileName, as_attachment=True)
        #facelen += 1

def croppingFrame(fileName):
    image = fileName
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = np.copy(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = classifier.detectMultiScale(gray_image, 1.25, 6)
    print('Number of faces detected:', len(faces))
    if(len(faces)!=0):
       face_crop = []
       for f in faces:
         x, y, w, h = [ v for v in f ]
         cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
         face_crop.append(gray_image[y:y+h, x:x+w])
    #facelen = 0
       for face in face_crop:
         croppedFileName = '{}{:-%Y%m%d%H%M%S}.jpg'.format(str(uuid.uuid4().hex), datetime.now())
         cv2.imwrite(cropped+croppedFileName,face)
         return croppedFileName     
    else:
        croppedFileName= ''
        return  croppedFileName
    
def cropFromVideoOrg(fileName):
    cap = cv2.VideoCapture(fileName)
    getFileName = "no face found"
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            getFileName=croppingFrame(frame)
            if len(getFileName) > 0 :
                break
        # Break the loop
        else:
            break
    return getFileName

        
PORT = int(os.getenv('PORT', 8000))
# Change current directory to avoid exposure of control files
#os.chdir('/static')
host_name = socket.gethostname() 
host_ip = socket.gethostbyname(host_name)
        
# driver function 
if __name__ == '__main__': 
    app.run(debug = True , host=host_ip, port=PORT )
