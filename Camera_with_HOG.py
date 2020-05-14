import cv2
import numpy as np
from time import sleep
from twilio.rest import Client
import datetime
import imutils
from imutils.object_detection import non_max_suppression


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video_capture = cv2.VideoCapture(0)


# account_sid = '*'
# auth_token = "*"
#
# client = Client(account_sid, auth_token)

current_time = datetime.datetime.now()
int_current_time = int(current_time.strftime('%H'))
#print(int_curr_time)

count = 0

while True:
    if not video_capture.isOpened():  
        print('Unable to load camera.')
        sleep(5) 
    else:
        sleep(.01)
        ret, frame = video_capture.read()  # Grabs, decodes and returns the next video frame (Capture frame-by-frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion of the image to the grayscale
        

        boxes, weights = hog.detectMultiScale(gray_frame, winStride=(8, 8), padding=(12,12), scale=1.095)


        for (x, y, w, h) in boxes:
            ROI_gray = gray_frame[y: y +h, x: x +w] # Extraction of the region of interest (face) from the frame
            cv2.rectangle(frame, (x, y), ( x +w, y+ h), (0, 0, 255), 2)
            

            if int_curr_time <= 23:
                print("message trigger")

                # client.messages.create(
                #     to="*",
                #     from_="*",
                #     body="Person Spotted"
                # )
                cv2.imwrite("frames/frame" + str(count) + ".jpg", frame)
                count += 1
                #sleep(1)
        # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # for (xA, yA, xB, yB) in pick:
        #     cv2.rectangle(gray_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)


        frame = cv2.resize(frame, (900, 500))
        cv2.imshow('Video', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
