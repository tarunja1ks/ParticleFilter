from utils import utils as util
import matplotlib
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from fractions import Fraction


matplotlib.use('TkAgg')

class OGM: 
    def __init__(self):
        dataset=20
        # init MAP
        self.MAP = {}
        self.MAP['res']   = 0.05 #meters
        self.MAP['xmin']  = -20  #meters
        self.MAP['ymin']  = -20
        self.MAP['xmax']  =  20
        self.MAP['ymax']  =  20 
        self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
        self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.MAP['map'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        with np.load("./Data/Hokuyo%d.npz"%dataset) as data:
            self.lidar_angle_min = data["angle_min"]*180/np.pi # start angle of the scan [rad]
            self.lidar_angle_max = data["angle_max"]*180/np.pi # end angle of the scan [rad]
            self.lidar_angle_increment = data["angle_increment"]*180/np.pi # angular distance between measurements [rad]
            self.lidar_range_min = data["range_min"] # minimum range value [m]
            self.lidar_range_max = data["range_max"] # maximum range value [m]
            self.lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
            self.lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans in seconds
    def bressenham_mark_Cells(self, scan, current_pose):
        
        print(scan,len(scan))
        angles = np.arange(self.lidar_angle_min,self.lidar_angle_max+self.lidar_angle_increment,self.lidar_angle_increment)*np.pi/180.0
        ranges = scan

        # take valid indices
        indValid = np.logical_and((ranges < self.lidar_range_max),(ranges> self.lidar_range_min))
        print(indValid,"indvalidOGM")
        print(len(angles), len(indValid))
        ranges = ranges[indValid]
        angles = angles[indValid]
        
        # xy position in the sensor frame
        xs0 = ranges*np.cos(angles)
        ys0 = ranges*np.sin(angles)
        
        # convert position in the map frame here 
        Y = np.stack((xs0,ys0))
        
        # convert from meters to cells
        xis = np.ceil((xs0 - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
        yis = np.ceil((ys0 - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
        
        # build an arbitrary map 
        indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.MAP['sizex'])), (yis < self.MAP['sizey']))
        self.MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1
            
        x_im = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res']) #x-positions of each pixel of the map
        y_im = np.arange(self.MAP['ymin'],self.MAP['ymax']+self.MAP['res'],self.MAP['res']) #y-positions of each pixel of the map

        x_range = np.arange(-0.2,0.2+0.05,0.05)
        y_range = np.arange(-0.2,0.2+0.05,0.05)




        #plot original lidar points
        fig1 = plt.figure()
        plt.plot(xs0,ys0,'.k')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Laser reading")
        plt.axis('equal')
        
        #plot map
        fig2 = plt.figure()
        plt.imshow(self.MAP['map'],cmap="hot")
        plt.title("Occupancy grid map")

        
        
if __name__ == '__main__':
    #   util.show_lidar()
    #   util. test_mapCorrelation()
    ogm=OGM()
    print(ogm.lidar_ranges[0])
    ogm.bressenham_mark_Cells(ogm.lidar_ranges[0],0)
    #   util. test_bresenham2D()
    ogm.showPlots()
