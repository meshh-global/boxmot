import cv2

cap = cv2.VideoCapture('rtsp://10.11.0.8:8554/cam1')

#cap = cv2.VideoCapture('http://345.63.46.1256/html/')

# cv2.namedWindow('live cam', cv2.WINDOW_NORMAL)
print(cap)
while(True):
    frame = None
    ret, frame = cap.read()
    #img_resize = cv2.resize(frame, (960, 540))
    # cv2.imshow('live cam', frame)
    if ret:
        print(".")
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
# cv2.destroyAllWindows()