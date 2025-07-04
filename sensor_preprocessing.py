from lsm6ds3 import LSM6DS3
import numpy as np
import time

lsm = LSM6DS3()
sent = 0.000061

def read_acc():
	ax, ay, az, gx, gy, gz = lsm.get_readings()
	ax = sent * ax
	ay = sent * ay
	az = sent * az
	
	return ax, ay, az

def extract_features(data): #data has input shape: (3,128)
	#mean
	x_mean = np.mean(data[0])
	y_mean = np.mean(data[1])
	z_mean = np.mean(data[2])
	
	#std 
	std_x = np.std(data[0])
	std_y = np.std(data[1])
	std_z = np.std(data[2])
	
	#max
	max_x = np.max(data[0])
	max_y = np.max(data[1])
	max_z = np.max(data[2])
	
	#min
	min_x = np.min(data[0])
	min_y = np.max(data[1])
	min_z = np.max(data[2])
	
	#average of squares 
	square_x = np.mean(np.square(data[0]))
	square_y = np.mean(np.square(data[1]))
	square_z = np.mean(np.square(data[2]))
	
	#correlation coefficient
	corr_xy = np.corrcoef(data[0], data[1])[0][1]
	corr_xz = np.corrcoef(data[0], data[2])[0][1]
	corr_yz = np.corrcoef(data[1], data[2])[0][1]
	
	feat = [x_mean, y_mean, z_mean, std_x, std_y, std_z, max_x, max_y, max_z, min_x, min_y, min_z, square_x, square_y, square_z, corr_xy, corr_xz, corr_yz]
	
	return feat.reshape(1, -1)
	
	
