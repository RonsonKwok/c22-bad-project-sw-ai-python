# from pyexpat import model
import tensorflow as tf
from sanic import Sanic
from sanic.response import json
from sanic_cors import CORS, cross_origin
import os
import base64
import Settings
import pathlib
# import mediapipe as mp
import cv2
import numpy as np
import time
import json as JSON
import logging
logging.basicConfig(filename=Settings.SERVER1_LOG_FILE, level=logging.DEBUG)
app = Sanic("eatwat7")
CORS(app) #fetch all sites API

data_dir ="datasets"
data_dir = pathlib.Path(data_dir)
# class_names=list(data_dir.glob('*/'))
class_names=['Bakery','BBQ','Bervage','Burger','Curry','Dessert','Dim_sum','Fast_food','Hot_pot','Japanese','Noodles','Pizza','Seafood','Steak']
# logging.info(class_names)
IMAGE_HEIGHT=Settings.IMAGE_HEIGHT
IMAGE_WIDTH=Settings.IMAGE_WIDTH
#camera food detection? (add-on)

UPLOAD_DIR="uploads"

def load():
    global model
    model=tf.keras.models.load_model(Settings.MOBILE_NET_V2)

def predict_image(image_path:str)->str:
    global model
    # logging.info("image_path", image_path)
    image = cv2.imread(image_path)
    if image is None:
        logging.error("Image not found")
        return
    if model is None:
        logging.error("Model not found")
        load()
    crop_img=cv2.resize(image,(IMAGE_HEIGHT,IMAGE_WIDTH))
    img_array = tf.expand_dims(crop_img, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
   #Printing
    logging.info(
        "This image most likely belongs to {} with a {:.2f} percent confidence.{}"
        .format(os.path.basename((class_names[np.argmax(score)])), 100 * np.max(score),score)
    )
    classname=os.path.basename((class_names[np.argmax(score)]))
    result=dict()
    result["classname"]=classname
    result["confidence"]=f"{np.max(score)}"
    return result

@app.post("/get-food-identity")
async def get_food_identity(request):
    logging.info("get-food-identity started")
    #1 This not using formittable, it send the pic and put into JSON to send to server/ Sanic doesnt accept formittable/ 
    # or can use S3, upload to there and get from S3 like google drive)
    url_blob=request.body 
    print (url_blob)
    #2 JSON becomes an data array
    blob=base64.b64decode(url_blob)
    current_time=time.time()
    
    #3 save the byte stream into the the upload file into JPG format
    image_path=f"{UPLOAD_DIR}/{current_time}.jpg"
    image_result = open(image_path, 'wb')
    image_result.write(blob)
    #4 start prediction
    return json(predict_image(image_path))
    

@app.post("/predict_server")
async def test(request):
    return json("Accessing Predict AI Server...")


app.static("","uploads")
# app.static("","uploads")
if __name__ == "__main__":
    load()
    app.run(host="0.0.0.0", port=Settings.SERVER1_PORT)

# logging.info(list(data_dir.glob('*/')))