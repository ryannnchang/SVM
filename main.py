#Importing FIles
from sensor_preprocessing import read_acc, extract_features

#Importing libraries 
import pickle
import RPi.GPIO as GPIO
from sklearn.svm import SVC
import time

##Button Pin Definition
button = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(button, GPIO.IN)

#Label dictionary
data_labels = {0.0:"Walking", 1.0: "Walking Up", 2.0: "Walking Down", 3.0: "Sitting", 4.0: "Standing", 5.0: "Laying Down"}

#Loading Model In
model = pickle.load(open("SVM_model.pkl", "rb"))

#Collecting data
predictions = []

#Running Sensor 
while GPIO.input(button) != False:
	window = [[],[],[]]
	
	while len(window[0]) < 128:
		ax, ay, az = read_acc()
		window[0].append(ax)
		window[1].append(ay)
		window[2].append(az)
		time.sleep(0.02)
	
	feat = extract_features(window)
	prediction = model.predict(feat)
	print(data_labels[prediction[0]])

print("Stopping Program")
print("Saving data")

	
