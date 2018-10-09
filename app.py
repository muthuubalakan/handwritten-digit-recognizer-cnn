from flask import Flask, render_template, request
from keras.models import load_model
import sys
import numpy as np
from scipy.misc import imread, imresize
import re
import base64


model = load_model("CNNDigit.h5")
model._make_predict_function()
app = Flask(__name__)


def read_image(image):

    # change to str
    image = image.decode('utf-8')
    image = image.split(',')[1]
    with open("image.png", 'wb') as f:
        f.write(base64.b64decode(image))

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/prediction/', methods=["GET", "POST"])
def prediction():
    image = request.get_data()
    read_image(image)
    x = imread("image.png", mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28,1)
    result = model.predict(x)
    result = np.array_str(np.argmax(result, axis=1))
    result = str(result)
    result = result.strip('[]')
    return result


if __name__ == '__main__':
    sys.stdout.write("Starting the application...\n")
    app.run(host="localhost", port=8000)
