from flask import Flask, Response, request, jsonify, redirect, url_for
import cv2
import numpy as np
import keras


def character_recog(image_path):
    img_arr = []
    gray = cv2.imread('new.jpg', cv2.IMREAD_GRAYSCALE)
    #gray = cv2.imread("xc.jpg", cv2.COLOR_BGR2GRAY)
    reteval, img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = np.reshape(np.array(img), (32, 32, 1))
    cv2.imwrite("check.jpg", img)
    img_arr.append(img)
    img_arr = np.array(img_arr)
    model = keras.models.load_model("nmodelgi0")
    result = model.predict(img_arr)
    return str(int(np.argmax(result[0])))


app = Flask(__name__)


@app.route('/', methods=['POST'])
def loadImage():
    if request.method == 'POST':
        temp_file = request.files['image']
        temp_file.save("new.jpg")
        return jsonify({'message': character_recog(temp_file.filename)})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
