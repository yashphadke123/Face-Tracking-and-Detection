import cv2
import os 
from PIL import Image
import numpy as np

path = 'api/Dataset'
recognizer = cv2.face.createLBPHFaceRecognizer()
detector = cv2.CascadeClassifer('api/Cascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagepath in imagePaths:
        PIL_img = Image.open(imagepath).convert('L')
        img_np = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagepath)[-1].split(".")[1])
        faceSamples.append(img_np)
        ids.append(id)
    return faceSamples,ids
print("\n [INFO] Training faces. It will take a few seconds. Please Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))