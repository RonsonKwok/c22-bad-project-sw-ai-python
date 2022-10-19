# from pyexpat import model
# import tensorflow as tf
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


app = Sanic("eatwat7")
CORS(app) #fetch all sites API


@app.post("/predict_server")
async def test(request):
    return json("Accessing Predict AI Server...")


app.static("","public")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
