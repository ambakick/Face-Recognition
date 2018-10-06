import cv2
import time

#importing the cascade from the xml file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#detection functions
def detect(gray, frame):
    #x and y coordinates of the left corner
    #w and h width and height
    #cascades works in black and white images
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    # image, scale factor (1.3 times reduced size), 5(atleast 5 neighbour zones accepted also need to be accepted))

    #faces contains coordinates x,y,w,h
    #drawing rectangle with these cordinates
    id = 0
    for (x,y,w,h) in faces:
        id += 1
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_color = frame[ y:y+h, x:x+w]
        cv2.putText(roi_color, str(id), (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)
        #roi_gray = gray[ y:y+h, x:x+w ]
        # # roi_color = frame[ y:y+h, x:x+w ]
        #eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 6)
        #for(ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0, 255, 0), 2)
    return frame
    #as roi is a part of frame, drawing in roi will affect the frame


#video_capture = cv2.VideoCapture('face.mp4')
video_capture = cv2.VideoCapture(0)
fps = video_capture.get(cv2.CAP_PROP_FPS)

#print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

# Start time
start = time.time()
frame_count=0
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #calling the function to draw rectangles
    canvas = detect(gray, frame)
    #diplaying the result in a window
    cv2.imshow('Video', canvas)
    frame_count +=1

    end = time.time()
    seconds = end - start
    print ("runnning at fps %d" % (frame_count/seconds))
    #to interrupt the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# End time
end = time.time()
# Time elapsed
seconds = end - start
print ("Average fps %d" % (frame_count/seconds))

#releasing video camera resource
video_capture.release()
cv2.destroyAllWindows()