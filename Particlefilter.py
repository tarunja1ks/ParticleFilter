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
from numba import njit

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
        
        self.particles=np.asarray([np.asarray([Pose(initial_pose.getPoseVector()[0],initial_pose.getPoseVector()[1],initial_pose.getPoseVector()[2]),float(1/numberofparticles)]) for i in range(numberofparticles)])
        self.NumberEffective=numberOfParticles
        self.sigma_v=0.2 # the stdev for lin vel
        self.sigma_w=0.10 # the stdev for ang vel
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
    
    
    def update_step(self,OGM, scan):
        # iterate through each particle and crosscheck with the logodds of the hits from the poses
        new_weights=[]
        for particle in self.particles:
            hypothesis=particle[0]
            angles= np.arange(OGM.lidar_angle_min, OGM.lidar_angle_max + OGM.lidar_angle_increment, 
                          OGM.lidar_angle_increment) * np.pi / 180.0
            ranges= scan
            # take valid indices
            indValid= np.logical_and((ranges < OGM.lidar_range_max), (ranges > OGM.lidar_range_min))
            ranges= ranges[indValid]
            angles= angles[indValid]
            
            # xy position in the sensor frame
            xs0= ranges * np.cos(angles)
            ys0= ranges * np.sin(angles)
            
            numberofhits= len(xs0) # number of hits in a scan
            scans= []
            for i in range(numberofhits): 
                scans.append(Pose(xs0[i], ys0[i], angles[i]))
            scans= np.asarray(scans)
            
            # Create sensor pose with offset from robot center
            current_pose_vector= hypothesis.getPoseVector()
            sensor_pose= Pose(current_pose_vector[0] + OGM.sensor_x_r, 
                            current_pose_vector[1] + OGM.sensor_y_r, 
                            current_pose_vector[2] + OGM.sensor_yaw_r)
            
            # Transform scans from sensor frame to world frame
            for i in range(numberofhits):
                scans[i].setPose(np.matmul(sensor_pose.getPose(), scans[i].getPose()))
            
            # Process each scan hit
            matching_probability=0
            for i in scans:
                x, y= OGM.meter_to_cell(i.getPose())
                rx, ry= OGM.meter_to_cell(sensor_pose.getPose())  # Use sensor position, not robot center
                
                scan_intersect= util.bresenham2D(rx, ry, x, y)
                intersection_point_count= len(scan_intersect[0])
                
                matching_probability+=OGM.MAP['map'][x][y] # occupied hitpoint
                # OGM.ogm_plot(x,y, True)
            
            new_weights.append(matching_probability/numberofhits)
        
        # normalizing the new weights to prevent numerical overflow from the ogm logodds plot
        # new_weights=[(i-max(new_weights)) for i in new_weights]
        new_weights=np.exp(new_weights)
        
        total_weight=0
        for i in range(self.numberofparticles):
            self.particles[i][1]*=new_weights[i] # adding the new probabilties in
            total_weight+=self.particles[i][1]
        
        
        for i in range(self.numberofparticles):
            self.particles[i][1]/=total_weight
            
        
        # print([float(i[1]) for i in self.particles],"--------------")
        
        weighted_x=sum([self.particles[i][0].getPoseVector()[0]*self.particles[i][1] for i in range(self.numberofparticles)])
        weighted_y=sum([self.particles[i][0].getPoseVector()[1]*self.particles[i][1] for i in range(self.numberofparticles)])
        
        weighted_sin=sum([math.sin(self.particles[i][0].getPoseVector()[2])*self.particles[i][1] for i in range(self.numberofparticles)])
        weighted_cos=sum([math.cos(self.particles[i][0].getPoseVector()[2])*self.particles[i][1] for i in range(self.numberofparticles)])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        weighted_angle=math.atan2(weighted_sin,weighted_cos)
        
        return Pose(weighted_x,weighted_y,weighted_angle)
    
    def resampling_step(self):
        self.NumberEffective=1/sum([(self.particles[i][1])**2 for i in range(len(self.particles))])
        if(self.NumberEffective<=self.numberofparticles/3):
            weights=[i[1] for i in self.particles]
            cumsum = np.cumsum(weights) 
            
            sample_points = np.random.random() / self.numberofparticles + np.arange(self.numberofparticles) / self.numberofparticles
            indices = np.searchsorted(cumsum, sample_points)
            particle_index=0
            new_particles=[]
            for i in indices:
                particle_index+=1
                new_particle=np.asarray([self.particles[i][0],float(1/self.numberofparticles)])
                new_particles.append(new_particle)
            self.particles=np.asarray(new_particles)
        
        
            
        




    
    

initial_pose=Pose(0,0,0)
numberOfParticles=100
pf=ParticleFilter(initial_pose,numberOfParticles)

reads=np.load("reads.npz")['reads_data']
lin_vel=0
ang_vel=0

ogm=OGM()
last_t=reads[0][1]

ogm.bressenham_mark_Cells(ogm.lidar_ranges[:,0],pf.particles[0][0])
# ogm.showPlots()


# purely localization 
Trajectories=[Trajectory(pf.getPoseObject().getPoseVector())]*numberOfParticles
# iterating through all of the reads to update models/displays
ind=0


for event in reads:
    dt= float(event[1])-float(last_t)
    if dt>0:
        pf.prediction_step([float(lin_vel), float(ang_vel)], dt)
        for i in range(pf.numberofparticles):
            # update traj
            current_pose_vector=pf.particles[i][0].getPoseVector()
            Trajectories[i].trajectory_x.append(current_pose_vector[0])
            Trajectories[i].trajectory_y.append(current_pose_vector[1])
            Trajectories[i].trajectory_h.append(current_pose_vector[2])
            
            
    if event[0]=="e": # encoder
        lin_vel= event[2]
    elif event[0]=="i": #imu
        ang_vel= event[2]
    elif(event[0]=="l"): # lidar
        new_Pose=pf.update_step(ogm, ogm.lidar_ranges[:,int(event[2])] )
        # print(new_Pose)
        ogm.bressenham_mark_Cells(ogm.lidar_ranges[:,int(event[2])],new_Pose)
        ogm.updatePlot()   
        pf.resampling_step()
        ind+=1
        print(ind)
    else:
        continue
    last_t= event[1]
    
    
ogm.updatePlot()
[i.showPlot() for i in Trajectories] #showing the robots trajectory from encoders/imu
plt.pause(10000000)


    



    




