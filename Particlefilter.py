from OGM import OGM,Trajectory
from utils import utils as util
import matplotlib
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import time
from fractions import Fraction
from Pose import Pose
from scipy.special import erf
import math


matplotlib.use('TkAgg')

class ParticleFilter:
    def __init__(self, initial_pose, OGM, numberofparticles=3):
        dataset = 20
        self.numberofparticles=numberofparticles
        with np.load("./Data/Imu%d.npz"%dataset) as data:
            self.imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
            self.imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
            self.imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
        with np.load("./Data/Encoders%d.npz"%dataset) as data:
            self.encoder_counts = data["counts"] # 4 x n encoder counts
            self.encoder_stamps = data["time_stamps"] # encoder time stamps
    
        self.particle_poses= np.tile(initial_pose, (self.numberofparticles, 1)).astype(np.float64)
        self.particle_weights= np.ones(self.numberofparticles)/self.numberofparticles
        
        self.NumberEffective=numberOfParticles
        self.sigma_v=0.02 # the stdev for lin vel
        self.sigma_w=0.03 # the stdev for ang vel 
        self.lidar_stdev=0.05
        
        self.covariance=np.asarray([[self.sigma_v**2,0],[0,self.sigma_w**2]])
        self.xt=initial_pose

        self.robotTosensor= np.array([OGM.sensor_x_r, OGM.sensor_y_r, OGM.sensor_yaw_r])
        

    def normal_pdf(self,x, mu, sigma):
        return np.exp(-0.5*((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
    
    def normal_cdf(self, x, mu, sigma):
        z = (x - mu) / (sigma * np.sqrt(2))
        return 0.5 * (1 + erf(z))


        
    def getPose(self):
        return self.xt.getPose()
    def getPoseObject(self):
        return self.xt
    
    def setPose(self,pose):
        self.xt=pose
    
        
        
    def prediction_step(self,U, Tt): # in the prediction step we create the noise and update the poses
            noise= np.random.multivariate_normal([0,0], self.covariance, size=self.numberofparticles)
            noisy_U= U + noise
            vel= noisy_U[:,0]
            ang= noisy_U[:,1]
            theta= self.particle_poses[:,2]
            
            angle= ang * Tt / 2
            sinc_angle = util.sinc(angle)
            
            dx= Tt * vel * sinc_angle * np.cos(theta + angle)
            dy= Tt * vel * sinc_angle * np.sin(theta + angle)
            dtheta= Tt * ang
            
            self.particle_poses[:,0] += dx
            self.particle_poses[:,1] += dy
            self.particle_poses[:,2] += dtheta
            
    
    def update_step(self,OGM, scan, max_cell_range=600):
        # iterate through each particle and crosscheck with the logodds of the hits from the poses
        new_weights=[]
        j=0
        for particle in self.particle_poses:
            sensor_pose=particle+self.robotTosensor
            
            angles = np.linspace(OGM.lidar_angle_min, OGM.lidar_angle_max, len(scan)) * np.pi / 180.0
            indValid = np.logical_and((scan < OGM.lidar_range_max), (scan > OGM.lidar_range_min))
            ranges = scan[indValid]
            angles = angles[indValid]
            

            
            xs0= (ranges * np.cos(angles))
            ys0= (ranges * np.sin(angles))

            # transform vector-wise into world coord

            xs_scans= sensor_pose[0]+np.cos(sensor_pose[2])*xs0-np.sin(sensor_pose[2])*ys0
            ys_scans= sensor_pose[1]+np.sin(sensor_pose[2])*xs0+np.cos(sensor_pose[2])*ys0
            angles_scans=sensor_pose[2]+angles

            scans=np.stack([xs_scans,ys_scans,angles_scans],axis=1)
            
            ztk=ranges
            
            
            
            # displacements for the angles
            scales=np.linspace(0,1,max_cell_range)
            dx = (np.cos(angles_scans) * max_cell_range)[:, None] * scales
            dy = (np.sin(angles_scans) * max_cell_range)[:, None] * scales
            
            cell_sensor_pose=OGM.meter_to_cell(sensor_pose)
            
            x_cells = np.floor(dx + cell_sensor_pose[0]).astype(int)  # shape (num_rays, max_cell_range)
            y_cells = np.floor(dy + cell_sensor_pose[1]).astype(int)
            
            H, W = OGM.MAP['map'].shape
            x_clamped=np.clip(x_cells, 0, H-1)
            y_clamped=np.clip(y_cells, 0, W-1)
            invalid=(x_cells < 0) | (x_cells >= H) | (y_cells < 0) | (y_cells >= W)
            x_cells=np.where(invalid, x_clamped, x_cells)
            y_cells=np.where(invalid, y_clamped, y_cells)

            occupied=OGM.MAP['map'][x_cells, y_cells ]>0 # shape (num_rays, max_cell_range)
            

            indices=np.argmax(occupied, axis=1)
            
            # print(indices.shape)
            rayrows=np.arange(x_cells.shape[0])
            
            xhits=x_cells[rayrows,indices]
            yhits=y_cells[rayrows,indices]
            
            ztkstar=(((yhits-cell_sensor_pose[1])**2+(xhits-cell_sensor_pose[0])**2)**0.5)/20
            
            
            # nhit and phit vecotrized calcuations
            nhit = 1/(self.normal_cdf(30, ztkstar, self.lidar_stdev) - self.normal_cdf(0, ztkstar, self.lidar_stdev))
            phit = nhit * self.normal_pdf(ztk, ztkstar, self.lidar_stdev)
            
            # print(phit)
            
            

            # nhit=self.normal_cdf(30,zkstars,)
            
            # nshort and pshort vecotrized calcuations
            
            
            
            
            
            
            
            weightI=np.sum(phit)
            self.particle_weights[j]=weightI
            j+=1
            
            
            
    
        
        # normalizing the weights
        total_weight=np.sum(self.particle_weights)
        self.particle_weights/=total_weight
        
        
            
        
        
        weighted_x=np.sum(self.particle_poses[:, 0] * self.particle_weights)
        weighted_y=np.sum(self.particle_poses[:, 1] * self.particle_weights)

        weighted_sin=np.sum(np.sin(self.particle_poses[:, 2]) * self.particle_weights)
        weighted_cos=np.sum(np.cos(self.particle_poses[:, 2]) * self.particle_weights)
        weighted_angle=math.atan2(weighted_sin, weighted_cos)
        
        return np.array([weighted_x,weighted_y,weighted_angle])
    
    def resampling_step(self):
        self.NumberEffective= 1/np.sum(self.particle_weights**2)
        if self.NumberEffective<=self.numberofparticles * 0.5:
            cumsum= np.cumsum(self.particle_weights)
            sample_points= np.random.random() / self.numberofparticles + np.arange(self.numberofparticles) / self.numberofparticles
            indices= np.searchsorted(cumsum, sample_points)
            self.particle_poses= self.particle_poses[indices]
            self.particle_weights= np.full(self.numberofparticles, 1.0 / self.numberofparticles)

    
    

initial_pose=np.array([0,0,0])
numberOfParticles=50


reads=np.load("reads.npz")['reads_data']
lin_vel=0
ang_vel=0

ogm=OGM()
last_t=reads[0][1]

pf=ParticleFilter(initial_pose,ogm,numberOfParticles)

ogm.bressenham_mark_Cells(ogm.lidar_ranges[:,0],pf.particle_poses[0])
ogm.showPlots()

print("here")

# purely localization 

# Trajectories = [Trajectory(initial_pose) for i in range(pf.numberofparticles)]

# iterating through all of the reads to update models/displays
ind=0



for event in reads:
    dt= float(event[1])-float(last_t)
    if dt>0:
        pf.prediction_step([float(lin_vel), float(ang_vel)], dt)
        for i in range(pf.numberofparticles):
            current_pose_vector= pf.particle_poses[i]  # <- use numeric poses
            # Trajectories[i].trajectory_x.append(current_pose_vector[0])
            # Trajectories[i].trajectory_y.append(current_pose_vector[1])
            # Trajectories[i].trajectory_h.append(current_pose_vector[2])
        
    if event[0]=="e": # encoder
        lin_vel= event[2]
    elif event[0]=="i": #imu
        ang_vel= event[2]
    elif(event[0]=="l"): # lidar
        new_Pose=pf.update_step(ogm, ogm.lidar_ranges[:,int(event[2])] )
        ogm.bressenham_mark_Cells(ogm.lidar_ranges[:,int(event[2])],new_Pose)
        ogm.updatePlot()   
        pf.resampling_step()
        ind+=1
        print(ind)

    else:
        continue
    last_t= event[1]
    
    
# [i.showPlot() for i in Trajectories] #showing the robots trajectory from encoders/imu

plt.show() 
plt.pause(10000000)


    



    




