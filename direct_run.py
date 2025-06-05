#open saved model
import joblib
#load saaved model
model = joblib.load("face_recognition_model.pkl")
path=str(input("Enter the path of the image: "))
path=path.split('"')[1]

#import json dictionary
import json
with open('class_dict.json', 'r') as file:
    class_dictionary = json.load(file)

#base for predicting
import cv2
import numpy as np
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')   #face cascade
eyes_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')                  #eye cascade



if face_detect.empty():
    raise IOError("Failed to load haarcascade_frontalface_default.xml")

if eyes_detect.empty():
    raise IOError("Failed to load haarcascade_eye.xml")


def cropped_image_with_2eyes(image_path):
    img=cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return None
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eyes_detect.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            return roi_color
    return None


import pywt
def wavelet_transform(image_path,mode='haar',level=2):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    array=img
    #array=cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    array=np.float32(array)
    array /= 255.0  # Normalize to [0, 1]
    coeffs = pywt.wavedec2(array, 'haar', level=level)
    coeffs_H=list(coeffs)
    coeffs_H[0] *=0  # Set the approximation coefficients to zero

    array_H = pywt.waverec2(coeffs_H, 'haar')
    array_H *= 255.0  # Scale back to [0, 255]
    array_H=np.uint8(array_H)  # Convert to uint8

    return array_H



def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return None
    
    #cut the face and eyes then predict
    cropped_img = cropped_image_with_2eyes(image_path)
    if cropped_img is None:
        print(f"Error: No valid face with two eyes detected in {image_path}")
        return None
    img = cropped_img
    scalled_raw_image = cv2.resize(img, (32, 32))
    img_har = wavelet_transform(image_path)
    scalled_raw_har = cv2.resize(img_har, (32, 32))
    combined_image = np.vstack((scalled_raw_image.reshape(32*32*3, 1), scalled_raw_har.reshape(32*32, 1)))
    combined_image = combined_image.reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(combined_image)
    return prediction[0]



# Predict the image
result = predict_image(path)
for key, value in class_dictionary.items():
    if value == result:
        result = key
        break

if result is not None:
    print(f"Predicted class for the image: {result}")
else:
    print("Prediction could not be made due to an error in processing the image.")