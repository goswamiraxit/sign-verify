import traceback
import logging

logging.basicConfig(level=logging.INFO)

from flask import Flask, json, jsonify, request
from PIL import Image

app = Flask(__name__)

import sign_image
import sign_finger


@app.route('/verify-image', methods=['POST'])
def handleImage():
    # print(list(request.files.keys()))
    try:
        sign = Image.open(request.files['sign'])
        vsign = Image.open(request.files['vsign'])
        return jsonify({'result': sign_image.predict(sign, vsign)})
    except Exception as ex:
        return jsonify({
            'error': str(ex),
            'stack': traceback.format_exc()
        })

@app.route('/verify-finger', methods=['POST'])
def handleFinger():
    try:
        sign = request.files['sign']
        vsign = request.files['vsign']
        output = sign_finger.predict(sign, vsign)
        return jsonify({'result': 1 - output})
    except Exception as ex:
        fex = traceback.format_exc()
        print(fex)
        return jsonify({
            'error': str(ex),
            'stack': fex
        })
    

@app.route('/verify-stylus', methods=['POST'])
def handleStylus():
    return jsonify({'result': 1})


if __name__ == '__main__':
    app.run(debug=True)

