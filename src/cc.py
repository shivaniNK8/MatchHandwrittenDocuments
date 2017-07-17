import cv2
import numpy as np


class Segmentation:

	@staticmethod
	def getWords(img):
		ksize=(5,5)	
		size = (128,48)	
		#ret, binary = cv2.threshold(img,200 , 255, cv2.THRESH_BINARY)
		shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
		gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		kernel = np.ones(ksize,np.uint8)
		
		dilated = cv2.dilate(thresh, kernel, iterations = 1)
		
		connectivity = 4  
		# Perform the operation
		output = cv2.connectedComponentsWithStats(dilated, connectivity, cv2.CV_32S)
		# Get the results
		# The first cell is the number of labels
		num_labels = output[0]
		# The second cell is the label matrix
		labels = output[1]
		# The third cell is the stat matrix
		stats = output[2]
		# The fourth cell is the centroid matrix
		centroids = output[3]
		
		#print(len(stats))
		words = []
		print('here')
		for i in range(len(stats)):
			x = stats[i,cv2.CC_STAT_LEFT]
			y = stats[i,cv2.CC_STAT_TOP]
			w = stats[i,cv2.CC_STAT_WIDTH]
			h = stats[i,cv2.CC_STAT_HEIGHT]
			print('width:'+str(w-x)+'height:'+str(y-h))
			#if abs(w-x) >10 and abs(y-h) > 10:
			temp = img[y:y+h, x:x+w]
			#cv2.imshow('im'+str(i),temp)
			temp = cv2.resize(temp, size)
			words.append(temp)
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			#Write component words
			

		cv2.imshow('img',img)
		#cv2.imwrite('2outputcc5.jpg',img)
		
		cv2.imshow('thresh',thresh)
		cv2.imshow('dilated',dilated)
		
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		return words

# a = Segmentation()
# a.getWords()