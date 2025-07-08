from sensor_preprocessing import read_acc
import matplotlib.pyplot as plt

count = 0
data = []
while count < 5:
    window = [[], [], []]

    while len(window[0]) < 128:
       # Collect accelerometer data
      ax, ay, az = read_acc()
      window[0].append(ax)
      window[1].append(ay)
      window[2].append(az)
    
    data.append(window)
    count += 1

#Plotting the collected data
for i, window in enumerate(data):
    plt.figure(figsize=(10, 4))
    plt.plot(window[0], label='X axis')
    plt.plot(window[1], label='Y axis')
    plt.plot(window[2], label='Z axis')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration')
    plt.title(f'Accelerometer Data (Window {i+1})')
    plt.legend()
    plt.tight_layout()
    plt.show()
  

    