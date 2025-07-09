from sensor_preprocessing import read_acc, extract_features
import matplotlib.pyplot as plt
import pickle

data_labels = {0.0:"Walking", 1.0: "Walking Up", 2.0: "Walking Down", 3.0: "Sitting", 4.0: "Standing", 5.0: "Laying"}

model = pickle.load(open("SVM_model.pkl", "rb"))

data = []
window = [[], [], []]

while len(window[0]) < 128:
# Collect accelerometer data
    ax, ay, az = read_acc()
    window[0].append(ax)
    window[1].append(ay)
    window[2].append(az)
    
data.append(window)
feat = extract_features(window)
prediction = model.predict(feat)

print(feat)

#Plotting the collected data
plt.figure(figsize=(10, 4))
plt.plot(window[0], label='X axis')
plt.plot(window[1], label='Y axis')
plt.plot(window[2], label='Z axis')
plt.xlabel('Sample')
plt.ylabel('Acceleration')
plt.title(f'Accelerometer Data (Window {prediction}, {data_labels[prediction[0]]})')
plt.legend()
plt.tight_layout()
plt.show()
  

    
