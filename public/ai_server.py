from pyexpat import model
import tensorflow as tf
from sanic import Sanic, json
from sanic_cors import CORS, cross_origin
import os
import base64
import pathlib
import mediapipe as mp
import cv2
import numpy as np
import time
import json as JSON

app = Sanic("flower-recognition")
CORS(app)


def load():
    global model
    model = tf.keras.models.load_model("mobile_net_v2")


app.static("public")


if __name__ == "__main__":
    load()
