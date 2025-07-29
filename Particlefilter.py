from utils import utils as util
import matplotlib
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from fractions import Fraction
from Pose import Pose
import math

matplotlib.use('TkAgg')

class ParticleFilter:
    def __init__(self):
        dataset = 20
        with np.load("./Data/Imu%d.npz"%dataset) as data:
            self.imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
            self.imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
            self.imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
        with np.load("./Data/Encoders%d.npz"%dataset) as data:
            self.encoder_counts = data["counts"] # 4 x n encoder counts
            self.encoder_stamps = data["time_stamps"] # encoder time stamps
        
        print(self.encoder_counts[0])
        for i in range(len(self.encoder_stamps)-1):
            print(self.encoder_stamps[i+1]-self.encoder_stamps[i])
        print(self.encoder_counts.shape)
        
    def motion_model():
        thing=True

class Robot:
    def __init__(self):
        self.xt=Pose(0,0,0)
    
    def getPose(self):
        return self.xt
    
    def setPose(self,pose):
        self.xt=pose
    
    def sinc(self,x):
        return math.sin(x)/x

    def motion_model(self,U,Tt): # u is the control input [v,w], and Tt is the time interval for this control input
        xt_as_vector=self.xt.getPoseVector()
        vel_t=U[0]
        ang_t=U[1]
        angle=ang_t*Tt/2
        xt1=xt_as_vector+Tt*np.asarray([vel_t*self.sinc(angle)*math.cos(xt_as_vector[2]+angle), vel_t*self.sinc(angle)*math.sin(xt_as_vector[2]+angle), ang_t ]) #X_T+1
        return xt1
    
    
thing=Robot()
print(thing.motion_model([1,0.5],1))
print()
t=ParticleFilter()



