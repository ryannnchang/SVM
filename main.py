#Importing FIles
from sensor_preprocessing import read_acc, extract_features

#Importing libraries 
import pickle
import RPi.GPIO as GPIO
from sklearn.svm import SVC
import time
import datetime
import csv

#Run Number
file_name = str(input("Enter the file name (without .csv): "))
file_dict = "experiment_log/" + file_name + ".csv"

##Button Pin Definition
button = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(button, GPIO.IN)

#Label dictionary
data_labels = {0.0:"Walking", 1.0: "Walking Up", 2.0: "Walking Down", 3.0: "Sitting", 4.0: "Standing", 5.0: "Laying Down"}

#Loading Model In
model = pickle.load(open("SVM_model.pkl", "rb"))

#Collecting data
data = []

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
	date = datetime.datetime.now()
	data.append([data_labels[prediction[0]], date])
	
	
	print(f"Prediction: {data_labels[prediction[0]]} \t Time: {date}")

print("Stopping Program")
print("Saving data...")

with open(file_dict, mode="w", newline="") as file:
	for i in range(len(data)):
		writer = csv.writer(file)
		writer.writerow([data[i][0], data[i][1]])

file.close()
print(f"Data saved to {file_dict} successfully")
