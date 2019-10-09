
#Description: This program detects/predicts if a person has diabetes (1) or not (0)

#Data: https://www.kaggle.com/uciml/pima-indians-diabetes-database

'''
The pima-indians-diabetes data set comes from the Pima people.
The Pima are a group of Native Americans living in an area consisting of what is now central and southern Arizona. 
Thy have the highest reported prevalence of diabetes of any population in the world, 
and have contributed to numerous scientific gains through their willingness to participate in some research. 
Their involvement has led to significant findings with regard to the epidemiology, physiology,
clinical assessment, and genetics of both type 2 diabetes and obesity. - National Center for Biotechnology Information
'''

#Load libraries
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Load the data
from google.colab import files # Use to load data on Google Colab
uploaded = files.upload() # Use to load data on Google Colab

#Store the data set
df = pd.read_csv('diabetes.csv')

#Look at first 7 rows of data
df.head(7)

#Show the shape (number of rows & columns)
df.shape

#Checking for duplicates and removing them
df.drop_duplicates(inplace = True)

#Show the shape to see if any rows were dropped (number of rows & columns)
df.shape

#Show the number of missing (NAN, NaN, na) data for each column
df.isnull().sum()

#Convert the data into an array
dataset = df.values
dataset

# Get all of the rows from the first eight columns of the dataset
X = dataset[:,0:8] #X = dataset[:,0:8]   #X = df.iloc[:, 0:8] 
# Get all of the rows from the last column
y = dataset[:,8] #y = dataset[:,8]     #y = df.iloc[:, 8]

#Process the data
#the min-max scaler method scales the dataset so that all the input features lie between 0 and 1 inclusive
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale

#Split the data into 80% training and 20%

#train_test_split splits arrays or matrices into random train and test subsets. 
#That means that everytime you run it without specifying random_state, you will get a different result, this is expected behavior.
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state = 4)

#Build the model and architecture of the neural network

# The models architechture 3 layers,
# 1st layer with 12 neurons and activation function 'relu'
# 2nd layer with 15 neurons and activation function 'relu'
# the last layer has 1 neuron with an activation function = sigmoid function which returns a value btwn 0 and 1
# The input shape/ input_dim = 10 the number of features in the data set
model = Sequential([
    Dense(12, activation='relu', input_shape=( 8 ,)),
    Dense(15, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Loss measuers how well the model did on training , and then tries to improve on it using the optimizer
model.compile(optimizer='sgd', #Stochastic gradient descent optimizer.
              loss='binary_crossentropy', #Used for binary classification
              metrics=['accuracy'])

#Train the model

# Split the data into 20% validation data
hist = model.fit(X_train, y_train,
          batch_size=57, epochs=1000, validation_split=0.2)

#visualize the training loss and the validation loss to see if the model is overfitting
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#visualize the training accuracy and the validation accuracy to see if the model is overfitting
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

#Make a prediction & print the actual values
prediction = model.predict(X_test)
prediction  = [1 if y>=0.5 else 0 for y in prediction] #Threshold
print(prediction)
print(y_test)

#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = model.predict(X_train)
pred  = [1 if y>=0.5 else 0 for y in pred] #Threshold
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))
print()

#Print the predictions
#print('Predicted value: ',model.predict(X_train))

#Print Actual Label
#print('Actual value: ',y_train)

#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = model.predict(X_test)
pred  = [1 if y>=0.5 else 0 for y in pred] #Threshold
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))
print()

#Print the predictions
#print('Predicted value: ',model.predict(X_test))

#Print Actual Label
#print('Actual value: ',y_test)

#Evaluate the test data set

#The reason why we have the index 1 after the model.evaluate function is because
#the function returns the loss as the first element and the accuracy as the 
#second element. To only output the accuracy, simply access the second element 
#(which is indexed by 1, since the first element starts its indexing from 0).
model.evaluate(X_test, y_test)[1]
