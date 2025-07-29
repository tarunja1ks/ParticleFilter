from OGM import OGM
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
        
        
    def motion_model():
        thing=True

class Robot:
    def __init__(self):
        self.xt=Pose(0,0,0)
    
    def getPose(self):
        return self.xt.getPose()
    def getPoseObject(self):
        return self.xt
    
    def setPose(self,pose):
        self.xt=pose
    
    def sinc(self,x):
        if(x==0.0):
            return 1.0
        return math.sin(x)/x

    def motion_model(self,U,Tt): # u is the control input [v,w], and Tt is the time interval for this control input
        xt_as_vector=self.xt.getPoseVector()
        vel_t=U[0]
        ang_t=U[1]
        angle=ang_t*Tt/2
        xt1=xt_as_vector+Tt*np.asarray([vel_t*self.sinc(angle)*math.cos(xt_as_vector[2]+angle), vel_t*self.sinc(angle)*math.sin(xt_as_vector[2]+angle), ang_t ]) #X_T+1
        return xt1
    
    
robot=Robot()
# print(thing.motion_model([1,0.5],1))
t=ParticleFilter()

reads=np.load("reads.npz")['reads_data']
lin_vel=0
ang_vel=0

ogm=OGM()
last_t=reads[0][1]
ogm.bressenham_mark_Cells(ogm.lidar_ranges[:,0],robot.getPoseObject())
# ogm.showPlots()

# purely localization 
fig_traj, ax_traj = plt.subplots(1, 1, figsize=(8, 8))
ax_traj.set_title("Robot Trajectory")
ax_traj.set_xlabel("X [m]")
ax_traj.set_ylabel("Y [m]")
ax_traj.set_aspect('equal', adjustable='box') # Keep aspect ratio for trajectory
ax_traj.grid(True)

trajectory_x = []
trajectory_y = []
initial_pose_vector = robot.getPoseObject().getPoseVector()
trajectory_x.append(initial_pose_vector[0])
trajectory_y.append(initial_pose_vector[1])

# Plot initial robot position and trajectory on the new trajectory axes
robot_plot_traj, = ax_traj.plot(initial_pose_vector[0], initial_pose_vector[1], 'ro', markersize=5, label='Robot Position')
trajectory_line_traj, = ax_traj.plot(trajectory_x, trajectory_y, 'b-', linewidth=2, label='Trajectory')
ax_traj.legend()


for event in reads:
    dt= float(event[1])-float(last_t)
    if dt>0:
        new_Pose=robot.motion_model([float(lin_vel), float(ang_vel)], dt)
        robot.setPose(Pose(new_Pose[0],new_Pose[1],new_Pose[2]))
        
        current_pose_vector = robot.getPoseObject().getPoseVector()
        trajectory_x.append(current_pose_vector[0])
        trajectory_y.append(current_pose_vector[1])

        # update traj
        robot_plot_traj.set_data([current_pose_vector[0]], [current_pose_vector[1]])
        trajectory_line_traj.set_data(trajectory_x, trajectory_y)
        ax_traj.relim() 
        ax_traj.autoscale_view() 
    if event[0]=="e":
        lin_vel= event[2]
    elif event[0]=="i":
        ang_vel= event[2]
    elif(event[0]=="l"):
        print(event[2],robot.getPoseObject().getPoseVector())
        # ogm.bressenham_mark_Cells(ogm.lidar_ranges[:,int(event[2])],robot.getPoseObject())
        # ogm.updatePlot()
    else:
        continue
    last_t= event[1]
    
fig_traj.canvas.draw_idle()
plt.pause(10000)
    



    




