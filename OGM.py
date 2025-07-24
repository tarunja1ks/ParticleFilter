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
        self.MAP = {}
        self.MAP['res']   = 0.05 #meters
        self.MAP['xmin']  = -25  #meters
        self.MAP['ymin']  = -25
        self.MAP['xmax']  =  25
        self.MAP['ymax']  =  25 
        self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
        self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.MAP['map'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.float32) #DATA TYPE: char or int8
        
        
        fig2 = plt.figure()
        self.ogm_map = plt.imshow(self.MAP['map'], cmap="gray", vmin=-5, vmax=5)
        plt.title("Occupancy grid map")
        self.sensor_x_r = 0.30183  
        self.sensor_y_r = 0.0
        self.sensor_yaw_r = 0.0
        with np.load("./Data/Hokuyo%d.npz"%dataset) as data:
            self.lidar_angle_min = data["angle_min"]*180/np.pi # start angle of the scan [rad]
            self.lidar_angle_max = data["angle_max"]*180/np.pi # end angle of the scan [rad]
            self.lidar_angle_increment = data["angle_increment"]*180/np.pi # angular distance between measurements [rad]
            self.lidar_range_min = data["range_min"] # minimum range value [m]
            self.lidar_range_max = data["range_max"] # maximum range value [m]
            self.lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
            self.lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans in seconds
            
            
    def meter_to_cell(self,x,y):
        cell_x, cell_y = (int(np.ceil((x - self.MAP['xmin']) / self.MAP['res']) - 1), \
                  int(np.ceil((y - self.MAP['ymin']) / self.MAP['res']) - 1))
        return cell_x,cell_y
    
    def meter_to_cell(self, pose_matrix):
        x = pose_matrix[0][2]
        y = pose_matrix[1][2]
        cell_x = int(np.ceil((x - self.MAP['xmin']) / self.MAP['res']) - 1)
        cell_y = int(np.ceil((y - self.MAP['ymin']) / self.MAP['res']) - 1)
        return cell_x, cell_y
    
    def plot(self, x, y,value=1):
        self.MAP['map'][x][y]=value
    
    def ogm_plot(self,x,y, occupied=False):
        confidence=0.85 # confidence level of the sensor
        if(occupied):
            odds=confidence/(1-confidence)
        else:
            odds=(1-confidence)/confidence
        self.MAP['map'][x][y]+=math.log(odds)
        
        
        
        
        
        
        
        
    def bressenham_mark_Cells(self, scan, current_pose):
        # x0,y0=self.meter_to_cell(current_pose[0],current_pose[1])
        angles = np.arange(self.lidar_angle_min,self.lidar_angle_max+self.lidar_angle_increment,self.lidar_angle_increment)*np.pi/180.0
        ranges = scan

        # take valid indices
        indValid = np.logical_and((ranges < self.lidar_range_max),(ranges> self.lidar_range_min))
        ranges = ranges[indValid]
        angles = angles[indValid]
        
        # xy position in the sensor frame
        xs0 = ranges*np.cos(angles)
        ys0 = ranges*np.sin(angles)
        
        
        numberofhits=len(xs0) # number of hits in a scan
        scans=[]
        for i in range(numberofhits): 
            scans.append(Pose(xs0[i],ys0[i],angles[i]))
        scans=np.asarray(scans) # converting hits into the numpy array proper pose matrice(this is currently in lidar_position_object form)
        
            
            
        robot_Position_lidar=Pose(3.0183,0,0)
        for i in range(numberofhits): # converting scans in frame relative to robot center
            scans[i].setPose(np.matmul(scans[i].getPose(),robot_Position_lidar.getPose()))
            
        self.MAP['map'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.float32)
        for i in scans:
            x,y=self.meter_to_cell(i.getPose())
            rx,ry=self.meter_to_cell(current_pose.getPose()) # robot x and robot y
            scan_intersect=util.bresenham2D(rx,ry,x,y)
            intersection_point_count=len(scan_intersect[0])
            for j in range(intersection_point_count-1):
                self.ogm_plot(int(scan_intersect[0][j]),int(scan_intersect[1][j]),False)
            self.ogm_plot(x,y,True)
        

    def showPlots(self):
        # plt.show(block=True)
        plt.show()
    def updatePlot(self):
        self.ogm_map.set_data(self.MAP['map'])
        plt.pause(0.05)
        
        
if __name__ == '__main__':
    #   util.show_lidar()
    #   util. test_mapCorrelation()
    ogm=OGM()
    robot_pose=Pose(0,0,0)
    ogm.showPlots()
    for i in range(0,340,2):       # 340 is stagnant limit for ogm testing
        ogm.bressenham_mark_Cells(ogm.lidar_ranges[:,i],robot_pose)
        ogm.updatePlot()
        print(i)
    import sys
    
    sys.stdout=open("output.txt","w")
    m=ogm.MAP['map']
    for i in m:
        output=""
        for j in i:
            output+=str(j)+" "
        print(output)
        
        