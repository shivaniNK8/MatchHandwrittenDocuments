import cv2
import numpy as np
import glob
import fnmatch
import argparse
import os
import collections
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from src.cnn.networks import CNN
from cc import Segmentation

def image_to_feature_vector(image, size=(128, 48)):
	return cv2.resize(image, size)


def getTrainData(dataPath):
	trainData = []
	trainLabels = []
	count = 0
	for root, dirnames, filenames in os.walk(dataPath):
		for filename in fnmatch.filter(filenames, '*.png'):
			
			label = root.split("/")[2]		
			imgPath = os.path.join(root, filename)
			img = cv2.imread(imgPath)
			count = count + 1
			trainData.append(img)
			trainLabels.append(label)
			if count % 10000 == 0:
				print(count)

	return trainData, trainLabels

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-s", "--save-model", type=int, default=-1,	help="(optional) whether or not model should be saved to disk")
	ap.add_argument("-l", "--load-model", type=int, default=-1,	help="(optional) whether or not pre-trained model should be loaded")
	ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
	args = vars(ap.parse_args())

	dataPath = 'dataset/Final2'
	classes = 40
	#classes = len(trainData) / 100	

	trainData = []
	trainLabels = []
	
	print("[INFO] Compiling model...")
	opt = SGD(lr=0.1)
	model = CNN.build(width=128, height=48, depth=3, classes=classes,weightsPath=args["weights"] if args["load_model"] > 0 else None)
	model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])		#rmsprop
	print("[INFO] model compiled")
	model.summary()

	if args["load_model"] < 0:
		print("[INFO] Reading images...")
		trainData, trainLabels = getTrainData(dataPath)	
		
	
		print(len(trainData))
																																																																																																																																																																								
		#print(trainLabels)
		le = LabelEncoder()
		trainLabels = le.fit_transform(trainLabels)		
	
		
		data = np.array(trainData) / 255.0
		#data = data[:, np.newaxis, :, :]
		labels = np_utils.to_categorical(trainLabels,classes)
																																																					
		print("[INFO] Performing training/testing split")
		#print(labels)
		(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size = 0.25, random_state=42)
		
	
		print("[INFO] Train/testLabels split performed")

		print("[INFO] training...")
		print("shape train data:", trainData.shape)
		model.fit(trainData,trainLabels,epochs=50,batch_size=128, verbose = 1)	#batch_size = 128 / 1000
		#show the accuracy on the testing set
		print("[INFO] evaluating...")
		(loss, accuracy) = model.evaluate(testData, testLabels,batch_size=128, verbose=1)	#batch_size = 128
		print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

	if args["save_model"] > 0:
		print("[INFO] dumping weights to file...")
		model.save_weights(args["weights"], overwrite=True)

	#image = cv2.imread('words/1.jpg')
	words1 = getWordSegments('rsz_1markov1.png')
	words1 = np.array(words1)
	preds1 = model.predict(words1)
	prediction1 = preds1.argmax(axis=1)	
	doc1Histogram = collections.Counter(prediction1)

	words2 = getWordSegments('rsz_page3.png')
	words2 = np.array(words2)
	preds2 = model.predict(words2)
	prediction2 = preds2.argmax(axis=1)	
	doc2Histogram = collections.Counter(prediction2)
	count, score = getScore(doc1Histogram, doc2Histogram, len(words1), len(words2))	

	# print(preds1)
	# print(prediction1)
	print('prediction 1: '+str(prediction1.shape))
	print(prediction1)
	print('prediction 2: '+str(prediction2.shape))
	print(prediction2)
	print('Document 1 histogram: ')
	print(doc1Histogram)
	print('Document 2 histogram: ')
	print(doc2Histogram)

	print("Score: "+ str(score) + " Match count: "+str(count))

	for j in range(len(words1)):
		#print('hi')
		cv2.putText(words1[j], str(prediction1[j]), (5, 20),	cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		cv2.imwrite('words1/'+str(j)+'.jpg',words1[j])
		#cv2.waitKey(0)

	for i in range(len(words2)):
		cv2.putText(words2[i], str(prediction2[i]), (5, 20),	cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		cv2.imwrite('words2/'+str(i)+'.jpg',words2[i])
		
	print("No of words doc1: "+str(len(words1)))
	print("No of words doc2: "+str(len(words2)))
	cv2.destroyAllWindows()
	
def getScore(doc1, doc2, len1, len2):
	count = 0
	for key in doc1:
		if key != 0:
			word = doc2.get(key, None)
			
			if word:
				count += min(word, doc1.get(key))
	
	score = (count *2)/ float(len1+len2 - 2)
	return count, score

def getWordSegments(path):
	img = cv2.imread(path)
	seg = Segmentation()
	words = seg.getWords(img)
	#print(words)
	return words


if __name__ == '__main__':
	main()
	

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# args = vars(ap.parse_args())
 
# grab the list of images that we'll be describing
#print("[INFO] describing images...")
# imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the data matrix and labels list
# data = []
# labels = []

# for (i, imagePath) in enumerate(imagePaths):
# 	image = cv2.imread(imagePath,0)
# 	label = 

# 	features = image_to_feature_vector(image)
# 	data.append(features)
# 	labels.append(label)

# if i > 0 and i % 1000 == 0:
# 		print("[INFO] processed {}/{}".format(i,len(imagePaths)))



