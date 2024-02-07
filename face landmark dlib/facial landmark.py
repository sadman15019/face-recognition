# import the necessary packages
from imutils import face_utils
import dlib
import cv2

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()   #this is dlib library for HOG+SVM based face detection, will     be reaplaced by neha's code
predictor = dlib.shape_predictor(p)  #this is used to predict landmarks,can  be trained using dataset , https://pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/



'''image2,gray2 and 'convert and trim_bb' function are used only for displaying detected facee'''


image=cv2.imread("test11.jpeg") 
image2=cv2.imread("test11.jpeg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
# detect faces in the grayscale image
rects = detector(gray, 0)
rects2 = detector(gray2, 0)

boxes = [convert_and_trim_bb(image2, r) for r in rects2]
# loop over the bounding boxes
for (x, y, w, h) in boxes:
	# draw the bounding box on our image
	cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
# show the output image
cv2.imwrite("face_detect11.jpeg", image2)
cv2.waitKey(0)
    
    # loop over the face detections
for (i, rect) in enumerate(rects):
  # determine the facial landmarks for the face region, then
  # convert the facial landmark (x, y)-coordinates to a NumPy
  # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
    cnt=0
    for (x, y) in shape:
            cnt+=1
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
    print(cnt)
    
    # show the output image with the face detections + facial landmarks
    
    
#cv2.imshow("Output", image)
cv2.imwrite("result11.jpeg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
