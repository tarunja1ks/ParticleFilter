import numpy as np
import math

if __name__ == '__main__':
  dataset = 21
  
  with np.load("./Data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  with np.load("./Data/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]-> one element array
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded) ->array
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans in seconds
  with np.load("./Data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  # with np.load("./Data/Kinect%d.npz"%dataset) as data:
  #   disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
  #   rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

# print(lidar_ranges[0])
# print(len(lidar_stamps))
# print(max(lidar_stamps)-min(lidar_stamps))
# print(len(set(lidar_stamps)))
# print(len(lidar_ranges))
# print(len(imu_angular_velocity[0]))

# print(encoder_counts[0,1000:1010])


reads=[]

# encoder reads for linear velocity
for i in range(1,len(encoder_stamps)):
    t=encoder_stamps[i]-encoder_stamps[i-1]
    d=0.254 # wheel diameter
    n=360 # ticks per revolution
    vl=(encoder_counts[0][i]+encoder_counts[2][i])/(2*n*t)*(math.pi*d)
    vr=(encoder_counts[1][i]+encoder_counts[3][i])/(2*n*t)*(math.pi*d)
    reads.append(["e",float(encoder_stamps[i]),float((vl+vr)/2),t])
    
# imu angular velocity
for i in range(0,len(imu_stamps)):
    reads.append(["i",float(imu_stamps[i]), float(imu_angular_velocity[2][i]),0])

# lidar data
for i in range(0,len(lidar_stamps)):
    reads.append(["l", float(lidar_stamps[i]),i,0]) # acsess through the main lidar read since lidar data big

reads=np.asarray(reads)

reads = reads[reads[:, 1].argsort()]
np.savez("reads.npz",reads_data=reads)



file=open("output.txt","w")
for i in reads:
  file.write(str(i)+"\n")
  





















# dts = []
# for i in range(1, len(imu_stamps)):
#     dts.append(imu_stamps[i] - imu_stamps[i-1])

# average_imu_dt = sum(dts) / len(dts)
# imu_frequency_hz = 1.0 / average_imu_dt
# print(imu_frequency_hz)

# dts = []
# for i in range(1, len(encoder_stamps)):
#     dts.append(encoder_stamps[i] - encoder_stamps[i-1])

# average_imu_dt = sum(dts) / len(dts)
# imu_frequency_hz = 1.0 / average_imu_dt
# print(imu_frequency_hz)




