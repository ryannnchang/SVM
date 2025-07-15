#Importing FIles
from sensor_preprocessing import read_acc, extract_features, extract_features2, mov_avg
from screen import display, off

#Importing libraries 
import pickle
import RPi.GPIO as GPIO
from sklearn.svm import SVC
import time
import datetime
import csv

#Run Number
#file_name = str(input("Enter the file name (without .csv): "))
file_name = 'Test'
file_dict = "experiment_log/" + file_name + ".csv"

##Button Pin Definition
button = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(button, GPIO.IN)

#Label dictionary
data_labels = {0.0:"Walking", 1.0: "WalkingUp", 2.0: "WalkingDw", 3.0: "Sitting", 4.0: "Standing", 5.0: "Laying"}

#Loading Model In
folder = "models"
model_file = "SVM_modelV2.pkl"
path = folder + "/" + model_file

model = pickle.load(open(path, "rb"))

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
	
	#Predicting based on raw data
	feat = extract_features2(window)
	predict_ind = model.predict(feat)
	prediction = data_labels[predict_ind[0]]
	date = datetime.datetime.now()
	
	#Predicting based on filtered 
	data_fil = mov_avg(window, 10)
	feat_fil = extract_features2(window)
	predict_ind_fil = model.predict(feat_fil)
	prediction_fil = data_labels[predict_ind_fil[0]]

	data.append([date, prediction, prediction_fil])
	
	print(f"Time: {date} \t PredictionRaw: {prediction}")
	display(prediction)

print("Stopping Program")
print("Saving data...")

with open(file_dict, mode="w", newline="") as file:
	for i in range(len(data)):
		writer = csv.writer(file)
		writer.writerow([data[i][0], data[i][1]])

file.close()
print(f"Data saved to {file_dict} successfully")

off()
