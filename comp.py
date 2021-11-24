# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")
args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# if the length the contours tuple returned by cv2.findContours
	# is '2' then we are using either OpenCV v2.4, v4-beta, or
	# v4-official
	if len(cnts) == 2:
		cnts = cnts[0]
	# if the length of the contours tuple is '3' then we are using
	# either OpenCV v3, v4-pre, or v4-alpha
	elif len(cnts) == 3:
		cnts = cnts[1]
	# otherwise OpenCV has changed their cv2.findContours return
	# signature yet again and I have no idea WTH is going on
	else:
		raise Exception(("Contours tuple must have length 2 or 3, "
			"otherwise OpenCV changed their cv2.findContours return "
			"signature yet again. Refer to OpenCV's documentation "
			"in that case"))
	# return the actual contours array
	return cnts
	# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
