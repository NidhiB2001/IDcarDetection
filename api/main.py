import cv2 
import numpy as np
import tensorflow as tf
from PIL import Image
from pyzbar.pyzbar import decode
from pyzbar import pyzbar
import os
from app import app
from flask import request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
# import base64
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
# from datetime import datetime
# import glob
# import qrdetect
# import keras_ocr
import pytesseract

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
# credentials = ServiceAccountCredentials.from_json_keyfile_name("flowing-radio-347411-f6b13db7738b.json", scopes)
# file = gspread.authorize(credentials) 
# sheet = file.open("Number plate Recognition").get_worksheet(15)
# header = ["Date", "Number plate"]
# sheet.insert_row(header)

# pipeline = keras_ocr.pipeline.Pipeline()

def allowed_file(filename):
    print("in allowed file''''''''''", filename)
    allowd = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return allowd

@app.route('/input', methods=['GET', 'POST'])

def upload_file():
    file = request.files['file']
    print("file", file)
    if request.method == "GET":
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            resp = jsonify({"Successfully GET file": filename})
            resp.status_code = 200
            return resp
        
    elif request.method == "POST":
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            print("file name =====================", filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resp = jsonify({"filename": filename, "Status":200})
            print("resp!!!!!!!!!!!!!!!",resp)
            resp.status_code = 200
            imag='Crop/'+filename
            # print('image"""""""""',image)
            
            # qrdetect.image
            # qrdetect.detection_result_image
            
            # decoded = base64.b64decode(image)
            # print("decodeed base6444444444444444", decoded)
            # pytesseract.pytesseract.tesseract_cmd = '/home/rao/Documents/pyPro/CV/Recognition/tesseract.wiki'
            # im = cv2.imread(imag)
            # try:
            #     image = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
            # except:
            #     pass
            # image = imutils.resize(image, width=500)
            print("Tesseract OCR print:\n ",pytesseract.image_to_string(imag, lang='eng'))
            
            # now = datetime.now()
            # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            # try:
            #     images = [
            #         keras_ocr.tools.read(img) for img in ['Crop/'+im]
            #     ]
            #     prediction_groups = pipeline.recognize(images)

            #     for i in prediction_groups:
            #         for text, box in i:
            #             print("text :::::::::", text)
            # except:
            #     pass
            # sheet.update('B'+str(l),[tlist])
            
            model_path = 'model.tflite'

            def preprocess_image(image_path, input_size):
                """Preprocess the input image to feed to the TFLite model"""
                img = tf.io.read_file(image_path)
                img = tf.io.decode_image(img, channels=3)
                img = tf.image.convert_image_dtype(img, tf.uint8)
                original_image = img
                resized_img = tf.image.resize(img, input_size)
                resized_img = resized_img[tf.newaxis, :]
                resized_img = tf.cast(resized_img, dtype=tf.uint8)
                return resized_img, original_image

            def detect_objects(interpreter, image, threshold):
                """Returns a list of detection results, each a dictionary of object info."""

                signature_fn = interpreter.get_signature_runner()

                # Feed the input image to the model
                output = signature_fn(images=image)

                # Get all outputs from the model
                count = int(np.squeeze(output['output_0']))
                scores = np.squeeze(output['output_1'])
                boxes = np.squeeze(output['output_3'])
                results = []
                for i in range(count):
                    if scores[i] >= threshold:
                        result = {
                            'bounding_box': boxes[i],
                            'score': scores[i]
                        }
                        results.append(result)
                return results

            def run_odt_and_draw_results(image_path, interpreter, threshold=0.9):
                """Run object detection on the input image and draw the detection results"""
                # Load the input shape required by the model
                _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

                # Load the input image and preprocess it
                preprocessed_image, original_image = preprocess_image(
                    image_path,
                    (input_height, input_width)
                    )

                # Run object detection on the input image
                results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

                # Plot the detection results on the input image
                original_image_np = original_image.numpy().astype(np.uint8)
                for obj in results:
                    # Convert the object bounding box from relative coordinates to absolute
                    # coordinates based on the original image resolution
                    ymin, xmin, ymax, xmax = obj['bounding_box']
                    xmin = int(xmin * original_image_np.shape[1])
                    xmax = int(xmax * original_image_np.shape[1])
                    ymin = int(ymin * original_image_np.shape[0])
                    ymax = int(ymax * original_image_np.shape[0])

                    boundb = cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
                    crop = boundb[ymin:ymax, xmin:xmax]
                    
                    # con = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    # gaussian_blur = cv2.GaussianBlur(con, (7,7), 2)
                    # sharpened1 = cv2.addWeighted(con, 1.5, gaussian_blur, -0.5, 0)
                    try:
                        cv2.imwrite("cropQR/"+'crop.jpg', crop)
                    
                    # read the QRCODE image
                    # qrc = cv2.imread(crop)
                    
                    # initialize the cv2 QRCode detector
                    # detector = cv2.QRCodeDetector()
                    # # detect and decode
                    # data, vertices_array, binary_qrcode = detector.detectAndDecode(crop)
                    # # if there is a QR code
                    # # print the data
                    # if vertices_array is not None:
                    #     print("QRCode data:")
                    #     print(data)
                    # else:
                    #     print("There was some error")
                    # barcodes = pyzbar.decode(crop)
                        for barcode in pyzbar.decode(crop):
                            barcodeData = barcode.data.decode("utf-8")
                            barcodeType = barcode.type
                            print(barcodeData, '\n', barcodeType)
                    # for barcode in decode(crop):
                    #     print(barcode.data)
                    #     myData = barcode.data.decode('utf-8')
                    #     print(myData)
                    #     if myData:
                    #         pts = np.array([barcode.polygon], np.int32)
                    #         pts = pts.reshape((-1,1,2))
                    #         cv2.polylines(crop,[pts], True,(255,0,255), 3)
                    except:
                        pass   
                original_uint8 = original_image_np.astype(np.uint8)
                return original_uint8

            DETECTION_THRESHOLD = 0.9
                                                                           
            # Load the TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            # Run inference and draw detection result on the local copy of the original file
            detection_result_image = run_odt_and_draw_results(
                imag,
                interpreter,
                threshold=DETECTION_THRESHOLD
            )
            # Show the detection result
            image = Image.fromarray(detection_result_image)
            return resp   
    return 'file' 

if __name__=="__main__":
    # context = ('/etc/letsencrypt/live/npr.mylionsgroup.com/cert.pem','/etc/letsencrypt/live/npr.mylionsgroup.com/privkey.pem')
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 8011)))
            # , ssl_context=context)