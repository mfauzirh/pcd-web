import base64
import cv2
from flask import Flask, request, render_template, jsonify
import numpy as np
import pcd


app = Flask(__name__)

def stringify(img):
    # encode into base 64
    im_b64 = base64.b64encode(img)
    # decode with utf-8
    im_decode = im_b64.decode("utf-8")

    return im_decode

def encode_to_image(img, format):
    # encode to image
    img = cv2.imencode(format, img)
    # get image data
    img = np.array(img[1])

    return img

def bytes_to_numpy(file_bytes):
    #convert string data to numpy array
    im_bytes = np.fromstring(file_bytes, np.uint8)
    
    # convert numpy array to image
    img = cv2.imdecode(im_bytes, cv2.IMREAD_UNCHANGED)

    return img

def read_upload_file(input_file_name):
    # if input_file_name not in request.files:
    #     return 'there is no image uploaded'

    # if(request.files[input_file_name].filename == ''):
    #     return 'there is no image uploaded'
    
    # read binary file
    im_file = request.files[input_file_name].read()

    return im_file

def read_uploaded_file_to_numpy(input_file_name):
    file_bytes = read_upload_file(input_file_name)
    img = bytes_to_numpy(file_bytes)

    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/extract-rgb', methods=['GET', 'POST'])
def extract_rgb():
    if request.method == 'POST':
        result = {}

        try:
            img = read_uploaded_file_to_numpy("im_file")
            rgb_channel = pcd.extractRGB(img)

            img = encode_to_image(img, '.png')
            im_decode = stringify(img)

            result.update({'img': im_decode, 'rgb': rgb_channel, 'status': True})
        except:
            result.update({'status': False, 'msg': 'There is an error occurred'})

        return render_template('extract_rgb.html', data=result)
           

    return render_template('extract_rgb.html')

@app.route('/grayscale', methods=['GET', 'POST'])
def grayscale():
    if request.method == 'POST':
        result = {}
        try:
            img = read_uploaded_file_to_numpy("im_file")

            grayscale = pcd.weightedAverageGrayscale(img)

            img = encode_to_image(img, '.png')
            grayscale = encode_to_image(grayscale, '.png')

            img_decode = stringify(img)
            grayscale_decode = stringify(grayscale)

            result.update({'img': img_decode, 'grayscale': grayscale_decode, 'status': True})
        except:
            result.update({'status': False, 'msg': 'There is an error occurred'})

        return render_template('grayscale.html', data=result)

    return render_template('grayscale.html')

@app.route('/brightness-manipulation', methods=['GET', 'POST'])
def brightnessManipulation():
    if request.method == 'POST':
        result = {}
        try:
            img = read_uploaded_file_to_numpy("im_file")

            img = encode_to_image(img, '.png')
            im_decode = stringify(img)

            result.update({'img': im_decode, 'status': True})
        except:
            result.update({'status': False, 'msg': 'There is an error occurred'})

        return render_template('brightness_manipulation.html', data=result)

    return render_template('brightness_manipulation.html')

@app.route('/brightness', methods=['POST'])
def brightness():
    data_uri = request.form['img']
    value = request.form['value']
    mode = request.form['mode']
    operation = request.form['operation']

    nparr = np.fromstring(base64.b64decode(data_uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if mode == 'manual':
        if operation == 'addSub':
            img = pcd.brightnessAddSub(img.copy(), int(value))
        elif operation == 'multiplication':
            img = pcd.brightness_multiplication(img.copy(), float(value))
        elif operation == 'division':
            img = pcd.brightness_divide(img.copy(), float(value))
            
    elif mode == 'opencv':
        if operation == 'addSub':
            img = pcd.brightnessOpenCV(img.copy(), int(value))
        elif operation == 'multiplication':
            img = pcd.brightness_multiplicationcv(img.copy(), float(value))
        elif operation == 'division':
            img = pcd.brightness_dividecv(img.copy(), float(value))
        

    img = cv2.imencode('.png', img)
    img = np.array(img[1])

    im_b64 = base64.b64encode(img)
    im_decode = im_b64.decode("utf-8")
    
    return jsonify({"img": im_decode})

@app.route('/bitwise-operation', methods=['GET', 'POST'])
def bitwiseOperation():
    if request.method == 'POST':
        result = {}
        try:
            mode = request.form.get('bitwiseModeValue')
            img1 = read_uploaded_file_to_numpy("im_file1")

            if mode == 'multi':
                img2 = read_uploaded_file_to_numpy("im_file2")

            img1 = encode_to_image(img1, '.png')
            im1_decode = stringify(img1)

            if mode == 'multi':
                img2 = encode_to_image(img2, '.png')
                im2_decode = stringify(img2)
            else:
                im2_decode = im1_decode

            result.update({'img1': im1_decode, 'img2': im2_decode, 'mode': mode, 'status': True})
        except:
            result.update({'status': False, 'mode': 'single', 'msg': 'There is an error occurred'})

        return render_template('bitwise_operation.html', data=result)

    return render_template('bitwise_operation.html')

@app.route('/bitwise', methods=['POST'])
def bitwise():
    data_uri_1 = request.form['img1']
    data_uri_2 = request.form['img2']
    operation = request.form['operation']

    nparr_1 = np.fromstring(base64.b64decode(data_uri_1), np.uint8)
    img1 = cv2.imdecode(nparr_1, cv2.IMREAD_COLOR)

    nparr_2 = np.fromstring(base64.b64decode(data_uri_2), np.uint8)
    img2 = cv2.imdecode(nparr_2, cv2.IMREAD_COLOR)

    if operation == 'and':
        img = pcd.bitwise_and(img1.copy(), img2.copy())
    elif operation == 'or':
        img = pcd.bitwise_or(img1.copy(), img2.copy())
    elif operation == 'not':
        img = pcd.bitwise_not(img1.copy())
    elif operation == 'xor':
        img = pcd.bitwise_xor(img1.copy(), img2.copy())

    img = cv2.imencode('.png', img)
    img = np.array(img[1])

    im_b64 = base64.b64encode(img)
    im_decode = im_b64.decode("utf-8")
    
    return jsonify({"img": im_decode})

if __name__ == '__main__':
    app.run(debug=True)