import cv2
from time import sleep
import datetime

cascPath = 'haarcascade_upperbody.xml'

faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture('http://192.168.3.24:8081')

current_time = datetime.datetime.now()
str_current_time = current_time.strftime('%H')
int_curr_time = int(str_current_time)
print(int_curr_time)

start_time = 14
end_time = 20
print("start" + str(start_time))
print("end" + str(end_time))

count = 0

while True:
    if not video_capture.isOpened():  # If the previous call to VideoCapture constructor or VideoCapture::open succeeded, the method returns true
        print('Unable to load camera.')
        sleep(5)  # Suspend the execution for 5 seconds
    else:
        sleep(.01)
        ret, frame = video_capture.read()  # Grabs, decodes and returns the next video frame (Capture frame-by-frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion of the image to the grayscale

        # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
        # image:		Matrix of the type CV_8U containing an image where objects are detected
        # scaleFactor:	Parameter specifying how much the image size is reduced at each image scale
        # minNeighbors:	Parameter specifying how many neighbors each candidate rectangle should have to retain it
        # minSize:		Minimum possible object size. Objects smaller than that are ignored
        faces = faceCascade.detectMultiScale(
            gray_frame,
            scaleFactor	= 1.05,
            minNeighbors = 10,
            minSize = (10, 10))


        # prediction = None
        # x, y = None, None

        for (x, y, w, h) in faces:
            ROI_gray = gray_frame[y: y +h, x: x +w] # Extraction of the region of interest (face) from the frame
            cv2.rectangle(frame, (x, y), ( x +w, y+ h), (0, 0, 255), 2)

            #print("box drawn")


            if int_curr_time <= 23:
                print("message trigger")
                cv2.imwrite("frames/frame" + str(count) + ".jpg", frame)
                count += 1
                #sleep(2)


        # Display the resulting frame
        frame = cv2.resize(frame, (800, 500))
        cv2.imshow('Video', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
