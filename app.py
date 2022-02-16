from flask import Flask, request, jsonify
import numpy as np
import keras
from PIL import Image, ImageOps


def character_recog(gray):
    img_arr = []
    img_arr.append(gray)
    img_arr = np.array(img_arr)
    model = keras.models.load_model("nmodelgi0")
    result = model.predict(img_arr)
    return str(int(np.argmax(result[0])))


app = Flask(__name__)


@app.route('/')
def home():
    return "lol"


@app.route('/predict/', methods=['POST'])
def loadImage():
    img = Image.open(request.files['image'])
    newsize = (32, 32)
    img = img.resize(newsize)
    img = ImageOps.grayscale(img)
    img = np.array(img)
    img = np.reshape(img, (32, 32, 1))
    return jsonify({'message': character_recog(img)})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
