import cv2 

cap = cv2.VideoCapture(0)
facedetector = cv2.CascadeClassifier('api/Cascades/haarcascade_frontalface_default.xml')
cap.set(3,640) # set Width
cap.set(4,480) # set Height
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame,1) # Flip camera vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetector.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20,50))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        count +=1
        cv2.imwrite("api/Dataset/User."+str(face_id)+'.'+str(count)+'.jpg',gray[y:y+h,x:x+h])
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    elif count >=30:
        break
cap.release()
cv2.destroyAllWindows()