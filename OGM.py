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

class OGM: 
    def __init__(self):
        dataset=20
        # init MAP
        self.MAP= {}
        self.MAP['res']  = 0.05 #meters
        self.MAP['xmin'] = -25  #meters
        self.MAP['ymin'] = -25
        self.MAP['xmax'] =  25
        self.MAP['ymax'] =  25 
        self.MAP['sizex'] = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
        self.MAP['sizey'] = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.MAP['map']= np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.float32) #DATA TYPE: char or int8
        
        
        fig2= plt.figure(figsize=(10, 10))
        extent= [self.MAP['ymin'], self.MAP['ymax'], self.MAP['xmin'], self.MAP['xmax']]
        self.ogm_map= plt.imshow(self.MAP['map'], cmap="gray", vmin=-5, vmax=5, 
                                 origin='lower', extent=extent)
        plt.title("Occupancy Grid Map (Dynamic)")
        plt.xlabel("Y [meters]")
        plt.ylabel("X [meters]")
        plt.colorbar(label="Log-odds")
        plt.grid(True, alpha=0.3)
        
        
        self.sensor_x_r= 0.30183  
        self.sensor_y_r= 0.0
        self.sensor_yaw_r= 0.0
        with np.load("./Data/Hokuyo%d.npz"%dataset) as data:
            self.lidar_angle_min= data["angle_min"]*180/np.pi # start angle of the scan [rad]
            self.lidar_angle_max= data["angle_max"]*180/np.pi # end angle of the scan [rad]
            self.lidar_angle_increment= data["angle_increment"]*180/np.pi # angular distance between measurements [rad]
            self.lidar_range_min= data["range_min"] # minimum range value [m]
            self.lidar_range_max= data["range_max"] # maximum range value [m]
            self.lidar_ranges= data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
            self.lidar_stamps= data["time_stamps"]  # acquisition times of the lidar scans in seconds
            
    def check_and_expand_map(self, x_world, y_world):
        """Check if coordinates are outside map bounds and expand if necessary"""
        expanded= False
        
        # Check if we need to expand
        if (x_world < self.MAP['xmin'] or x_world > self.MAP['xmax'] or 
            y_world < self.MAP['ymin'] or y_world > self.MAP['ymax']):
            
            # Calculate new bounds with some padding
            padding= 10  # meters
            new_xmin= min(self.MAP['xmin'], x_world - padding)
            new_xmax= max(self.MAP['xmax'], x_world + padding)
            new_ymin= min(self.MAP['ymin'], y_world - padding)
            new_ymax= max(self.MAP['ymax'], y_world + padding)
            
            # Calculate new map size
            new_sizex= int(np.ceil((new_xmax - new_xmin) / self.MAP['res'] + 1))
            new_sizey= int(np.ceil((new_ymax - new_ymin) / self.MAP['res'] + 1))
            
            # Create new larger map
            new_map= np.zeros((new_sizex, new_sizey), dtype=np.float32)
            
            # Calculate offset for copying old map data
            offset_x= int(np.floor((self.MAP['xmin'] - new_xmin) / self.MAP['res']))
            offset_y= int(np.floor((self.MAP['ymin'] - new_ymin) / self.MAP['res']))
            
            # Copy old map data to new map
            new_map[offset_x:offset_x + self.MAP['sizex'], 
                   offset_y:offset_y + self.MAP['sizey']]= self.MAP['map']
            
            # Update map parameters
            self.MAP['xmin']= new_xmin
            self.MAP['xmax']= new_xmax
            self.MAP['ymin']= new_ymin
            self.MAP['ymax']= new_ymax
            self.MAP['sizex']= new_sizex
            self.MAP['sizey']= new_sizey
            self.MAP['map']= new_map
            expanded= True
            
        return expanded
            
    def meter_to_cell(self, pose_matrix):
        x= pose_matrix[0][2]
        y= pose_matrix[1][2]
        
        # Check and expand map if necessary
        self.check_and_expand_map(x, y)
        
        # Convert to cell coordinates using proper floor operation
        cell_x= int(np.floor((x - self.MAP['xmin']) / self.MAP['res']))
        cell_y= int(np.floor((y - self.MAP['ymin']) / self.MAP['res']))
        
        # Clamp to valid range (should rarely be needed now)
        cell_x= max(0, min(cell_x, self.MAP['sizex'] - 1))
        cell_y= max(0, min(cell_y, self.MAP['sizey'] - 1))
        
        return cell_x, cell_y
    
    def plot(self, x, y, value=1):
        # Added bounds checking
        if 0 <= x < self.MAP['sizex'] and 0 <= y < self.MAP['sizey']:
            self.MAP['map'][x][y]= value
    
    def ogm_plot(self, x, y, occupied=False):
        if not (0 <= x < self.MAP['sizex'] and 0 <= y < self.MAP['sizey']):
            return
        confidence= 0.85 # confidence level of the sensor
        if occupied:
            odds= confidence / (1 - confidence)
        else:
            odds= (1 - confidence) / confidence
        self.MAP['map'][x][y] += math.log(odds)
        self.MAP['map'][x][y]= max(-10, min(10, self.MAP['map'][x][y]))
    def logOddstoProbability(self,logOdds):
        return 1 / (1 + math.exp(-logOdds))
    def probabilityToLogOdds(self,probability):
        return math.log(probability/(1-probability))
    
    def bressenham_mark_Cells(self, scan, current_pose):
        angles= np.arange(self.lidar_angle_min, self.lidar_angle_max + self.lidar_angle_increment, 
                          self.lidar_angle_increment) * np.pi / 180.0
        ranges= scan

        # take valid indices
        indValid= np.logical_and((ranges < self.lidar_range_max), (ranges > self.lidar_range_min))
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
        current_pose_vector= current_pose.getPoseVector()
        sensor_pose= Pose(current_pose_vector[0] + self.sensor_x_r, 
                          current_pose_vector[1] + self.sensor_y_r, 
                          current_pose_vector[2] + self.sensor_yaw_r)
        
        # Transform scans from sensor frame to world frame
        for i in range(numberofhits):
            scans[i].setPose(np.matmul(sensor_pose.getPose(), scans[i].getPose()))
        
        # Process each scan hit
        matching_probability=0
        for i in scans:
            x, y= self.meter_to_cell(i.getPose())
            rx, ry= self.meter_to_cell(sensor_pose.getPose())  # Use sensor position, not robot center
            
            scan_intersect= util.bresenham2D(rx, ry, x, y)
            intersection_point_count= len(scan_intersect[0])
            
            # Mark free cells along the ray
            for j in range(intersection_point_count - 1):
                probabilityNotOccupied=1-self.logOddstoProbability(self.MAP['map'][int(scan_intersect[0][j])][int(scan_intersect[1][j])])
                matching_probability+=self.probabilityToLogOdds(probabilityNotOccupied)
                self.ogm_plot(int(scan_intersect[0][j]), int(scan_intersect[1][j]), False)
                
            # Mark occupied cell at the hit
            
            matching_probability+=self.MAP['map'][x][y]
            self.ogm_plot(x, y, True)
        print(matching_probability)

       

    def showPlots(self):
        plt.show()
    
    # def mapCorrelation(): # making it again to understand it more 
        
    def updatePlot(self, robot_pose=None):
        # Check if map was expanded and recreate imshow if needed
        current_extent= [self.MAP['ymin'], self.MAP['ymax'], self.MAP['xmin'], self.MAP['xmax']]
        
        try:
            # Try to update existing plot
            self.ogm_map.set_data(self.MAP['map'])
            self.ogm_map.set_extent(current_extent)
        except:
            # If map size changed, recreate the plot
            plt.clf()  # Clear the figure
            self.ogm_map= plt.imshow(self.MAP['map'], cmap="gray", vmin=-5, vmax=5, 
                                     origin='lower', extent=current_extent)
            plt.title("Occupancy Grid Map (Dynamic)")
            plt.xlabel("Y [meters]")
            plt.ylabel("X [meters]")
            plt.colorbar(label="Log-odds")
            plt.grid(True, alpha=0.3)
            
            # Recreate robot marker
            self.robot_marker= plt.plot(0, 0, 'ro', markersize=8, label='Robot')[0]
            plt.legend()
        
        # Update robot position if provided
        if robot_pose is not None:
            pose_vec= robot_pose.getPoseVector()
            self.robot_marker.set_data([pose_vec[1]], [pose_vec[0]])  # Note: x,y swapped for display
        
        # Update axis limits to show full map
        plt.xlim(self.MAP['ymin'], self.MAP['ymax'])
        plt.ylim(self.MAP['xmin'], self.MAP['xmax'])
        
        plt.pause(0.05)
        
        
class Trajectory:
    def __init__(self, initial_pose_vector):
        self.fig_traj, self.ax_traj= plt.subplots(1, 1, figsize=(8, 8))
        self.ax_traj.set_title("Robot Trajectory")
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.set_aspect('equal', adjustable='box') 
        self.ax_traj.grid(True)
        
        # initializing robot position
        self.trajectory_x= []
        self.trajectory_y= []
        self.trajectory_x.append(initial_pose_vector[0])
        self.trajectory_y.append(initial_pose_vector[1])
        
        self.trajectory_line_traj,= self.ax_traj.plot(self.trajectory_x, self.trajectory_y, 'b-', linewidth=2, label='Trajectory')
        
    def showPlot(self):
        self.trajectory_line_traj.set_data(self.trajectory_x, self.trajectory_y)
        self.ax_traj.relim() 
        self.ax_traj.autoscale_view() 
        self.fig_traj.canvas.draw_idle()