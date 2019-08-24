# Digit - Recognizer Kaggle Competition

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding extra convolution layers
classifier.add(Conv2D(128, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(256, kernel_size=2, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flatten
classifier.add(Flatten())

# Step 4 - Fully Connected Layer
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(10, activation='softmax'))

# Compile the Model
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Importing the Dataset
import pandas as pd
train = pd.read_csv('train.csv').values
test = pd.read_csv('test.csv').values
sample = pd.read_csv('sample_submission.csv').values

# Reshape and normalize training data
trainX = train[:, 1:].reshape(train.shape[0],28, 28, 1).astype( 'float32' )
X_train = trainX / 255.0

y_train = train[:,0]


# Reshape and normalize test data
testX = test.reshape(test.shape[0],28, 28, 1).astype( 'float32' )
X_test = testX / 255.0

# Encoding the output
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
#y_test = lb.fit_transform(y_test)

# Training the dataset
classifier.fit(X_train, y_train, epochs=5)

# Predicting the test dataset
prediction = classifier.predict_classes(X_test)
#print(prediction.shape)

# Writing to csv
dict = {'ImageId': sample[:,0], 'Label': prediction}
df = pd.DataFrame(dict)
df.to_csv('submission.csv',index=False)