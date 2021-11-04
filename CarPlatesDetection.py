import cv2
import imutils


videosPath = '.\\videos\\'
videosSet = ['video1']
fullVideoPath = f'{videosPath}{videosSet[1]}.mp4'


# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture(fullVideoPath)

if (vid_capture.isOpened() == False):
    print("Error opening the video file")
else:

    # For debuggin purposes
    fps = vid_capture.get(5)
    print('Frames per second : ', fps, 'FPS')
    frame_count = vid_capture.get(7)
    print('Frame count : ', frame_count)

while(vid_capture.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is frame
    ret, frame = vid_capture.read()
    if ret == True:
        # print(frame)
        # print(type(frame))
        #img = cv2.imread(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise Reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
        keypoints = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                x, y, w, h = cv2.boundingRect(contour)
                break

        print(location)
        cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
        frame[y:y+h, x:x +
              w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (15, 15), cv2.BORDER_DEFAULT)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(20)

        if key == ord('q'):
            break
    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()
