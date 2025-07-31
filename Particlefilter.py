from OGM import OGM,Trajectory
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
    def __init__(self, initial_pose, numberofparticles=3):
        dataset = 20
        self.numberofparticles=numberofparticles
        with np.load("./Data/Imu%d.npz"%dataset) as data:
            self.imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
            self.imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
            self.imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
        with np.load("./Data/Encoders%d.npz"%dataset) as data:
            self.encoder_counts = data["counts"] # 4 x n encoder counts
            self.encoder_stamps = data["time_stamps"] # encoder time stamps
        
        self.particles=np.asarray([np.asarray([Pose(initial_pose.getPoseVector()[0],initial_pose.getPoseVector()[1],initial_pose.getPoseVector()[2]),1/numberofparticles]) for i in range(numberofparticles)])

        self.sigma_v=0.2 # the stdev for lin vel
        self.sigma_w=0.1 # the stdev for ang vel
        self.covariance=np.asarray([[self.sigma_v**2,0],[0,self.sigma_w**2]])
        self.xt=initial_pose
        
    def getPose(self):
        return self.xt.getPose()
    def getPoseObject(self):
        return self.xt
    
    def setPose(self,pose):
        self.xt=pose
        
    def prediction_step(self,U, Tt): # in the prediction step we create the noise and update the poses
        for i in range(self.numberofparticles):
            noisy_U=U+np.random.multivariate_normal([0,0],self.covariance)
            xt_as_vector=self.particles[i][0].getPoseVector()
            vel_t=noisy_U[0]
            ang_t=noisy_U[1]
            angle=ang_t*Tt/2
            xt1=xt_as_vector+Tt*np.asarray([vel_t*util.sinc(angle)*math.cos(xt_as_vector[2]+angle), vel_t*util.sinc(angle)*math.sin(xt_as_vector[2]+angle), ang_t ]) #X_T+1
            self.particles[i][0]=Pose(xt1[0],xt1[1],xt1[2])
        pf.setPose(self.particles[0][0])
            
    def update_step(self):
        
        return True
            
        




    
    

initial_pose=Pose(0,0,0)
pf=ParticleFilter(initial_pose,3)

reads=np.load("reads.npz")['reads_data']
lin_vel=0
ang_vel=0

ogm=OGM()
last_t=reads[0][1]
ogm.bressenham_mark_Cells(ogm.lidar_ranges[:,0],pf.getPoseObject())
# ogm.showPlots()


# purely localization 

Trajectories=[Trajectory(pf.getPoseObject().getPoseVector()),Trajectory(pf.getPoseObject().getPoseVector()),Trajectory(pf.getPoseObject().getPoseVector())]
# iterating through all of the reads to update models/displays
ind=0
for event in reads:
    dt= float(event[1])-float(last_t)
    if dt>0:
        pf.prediction_step([float(lin_vel), float(ang_vel)], dt)
        # print([j[0].getPoseVector() for j in pf.particles])
        for i in range(pf.numberofparticles):
            # update traj
            current_pose_vector=pf.particles[i][0].getPoseVector()
            Trajectories[i].trajectory_x.append(current_pose_vector[0])
            Trajectories[i].trajectory_y.append(current_pose_vector[1])
            
    if event[0]=="e": # encoder
        lin_vel= event[2]
    elif event[0]=="i": #imu
        ang_vel= event[2]
    elif(event[0]=="l"): # lidar
        ogm.bressenham_mark_Cells(ogm.lidar_ranges[:,int(event[2])],pf.getPoseObject()) # intersecting/marking cells
        ogm.updatePlot()
        ind+=1
        print(ind)
    else:
        continue
    last_t= event[1]
    
    

file=open("ogmMap.txt","w")
for i in ogm.MAP['map']:
    for j in i:
        file.write(str(j)+" ")
    file.write("\n")

ogm.updatePlot()
[i.showPlot() for i in Trajectories] #showing the robots trajectory from encoders/imu
plt.pause(10000)


    



    




