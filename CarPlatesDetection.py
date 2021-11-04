import cv2
import imutils


path = '.\haarcascade\\'
models = ['haarcascade_cars.xml',
          'haarcascade_russian_plate_number.xml', ]

fullPath = f'{path}{models[0]}'
platesClassifiers = cv2.CascadeClassifier(fullPath)

videosPath = '.\\videos\\'
videosSet = ['video1', 'video2']
fullVideoPath = f'{videosPath}{videosSet[1]}.mp4'

# Create a video capture object, in this case we are reading the video from a file
vido_capture = cv2.VideoCapture(fullVideoPath)

if (vido_capture.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
    # Get frame rate information
    # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    fps = vido_capture.get(5)
    print('Frames per second : ', fps, 'FPS')

    # Get frame count
    # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    frame_count = vido_capture.get(7)
    print('Frame count : ', frame_count)

while(vido_capture.isOpened()):
    # vido_capture.read() methods returns a tuple, first element is a bool
    # and the second is frame
    ret, frame = vido_capture.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Using gray scale
        plates = platesClassifiers.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(300, 300))  # 1.2 or 1.3

        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame[y:y+h, x:x +
                  w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (15, 15), cv2.BORDER_DEFAULT)

        cv2.imshow('Frame', frame)
        # 20 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(20)

        if key == ord('q'):
            break
    else:
        break

# Release the video capture object
vido_capture.release()
cv2.destroyAllWindows()
