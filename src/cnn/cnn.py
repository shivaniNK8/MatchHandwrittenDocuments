# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout

class CNN:
	@staticmethod
	def build(width, height, depth, classes, weightsPath=None):
		# initialize the model
		model = Sequential()

		input_shape = (height, width,depth)

	# #model 1 (5 layers)
	# 	model.add(Convolution2D(64, (5, 5), padding="same",
	# 		input_shape=input_shape))
	# 	model.add(Activation("relu"))
	# 	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# 	# second set of CONV => RELU => POOL
	# 	model.add(Convolution2D(128, (5, 5), padding="same"))
	# 	model.add(Activation("relu"))
	# 	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# 	# third set of CONV => RELU => POOL
	# 	model.add(Convolution2D(256, (3, 3), padding="same"))
	# 	model.add(Activation("relu"))
	# 	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# 	# fourth set of CONV => RELU => POOL
	# 	model.add(Convolution2D(512, (3, 3), padding="same"))
	# 	model.add(Activation("relu"))
	# 	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# 	#fifth set of CONV => RELU => POOL
	# 	model.add(Convolution2D(512, (3, 3), padding="same"))
	# 	model.add(Activation("relu"))
	# 	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# 	# set of FC => RELU layers
	# 	model.add(Flatten())
	# 	model.add(Dense(2048))
	# 	model.add(Activation("relu"))

	# 	# softmax classifier
	# 	model.add(Dropout(0.5))
	# 	model.add(Dense(classes))
	# 	model.add(Activation("sigmoid"))
	
		
	# #model 2  (Highest accuracy till now)

		
	# 	model.add(Conv2D(32, (3, 3), input_shape=input_shape))
	# 	model.add(Activation('relu'))
	# 	model.add(MaxPooling2D(pool_size=(2, 2)))
		
	# 	model.add(Conv2D(32, (3, 3)))
	# 	model.add(Activation('relu'))
	# 	model.add(MaxPooling2D(pool_size=(2, 2)))
		
	# 	model.add(Conv2D(64, (3, 3)))
	# 	model.add(Activation('relu'))
	# 	model.add(MaxPooling2D(pool_size=(2, 2)))
		
	# 	model.add(Flatten())  
	# 	model.add(Dense(2048))
	# 	model.add(Activation('relu'))
	# 	model.add(Dropout(0.5))
	# 	model.add(Dense(classes))
	# 	model.add(Activation('sigmoid'))

	#model 3 (Basic model)
		# print(input_shape)
		# model.add(Convolution2D(20,5,5,border_mode="same",input_shape=input_shape))
		# model.add(Activation("relu"))
		# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		
		# model.add(Convolution2D(50,5,5,border_mode="same"))
		# model.add(Activation("relu"))
		# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		
		# model.add(Flatten())
		# model.add(Dense(64))
		# model.add(Activation("relu"))
		
		# #softmax classifier
		# model.add(Dense(classes))
		# model.add(Activation("softmax"))

	#model 4 ()
		model.add(Conv2D(25, (5, 5), input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
		model.add(Conv2D(25, (5, 5)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
		model.add(Conv2D(25, (4, 4)))
		model.add(Activation('relu'))
		
		model.add(Flatten())  
		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation('softmax'))


		if weightsPath is not None:
			model.load_weights(weightsPath)		

		# return the constructed network architecture
		return model