import gtsam
import math
import numpy as np
import matplotlib.pyplot as plt; plt.ion()


class ICP: 
    def getClosest(self,Source,Target):
        diff= Source[:, np.newaxis, :] - Target[np.newaxis, :, :]
        distances= np.linalg.norm(diff, axis=2)
        closest=np.argmin(distances, axis=1)
        return Target[closest]

    def getCentroids(self,Source,Target):
        SourceCentroid=np.mean(Source,axis=0)
        TargetCentroid=np.mean(Target,axis=0)
        return np.array([SourceCentroid,TargetCentroid])





icp=ICP()


data = np.load('icp_dataset.npz')
# print(data['source'])
# print(data['target'])

plt.title("ICP graph")
plt.xlabel("Y [meters]")
plt.ylabel("X [meters]")
plt.grid(True, alpha=0.3)
plt.scatter(data['target'][:, 0], data['target'][:, 1], c='red', label='Target')


source=data['source'].copy()
# plotting the icp data points
errors = []


R_total = np.eye(2)
t_total = np.zeros(2)



for i in range(1000):
    weightedTarget=icp.getClosest(source,data['target'])
    SourceCentroid=icp.getCentroids(source,weightedTarget)[0]
    TargetCentroid=icp.getCentroids(source,weightedTarget)[1]
    err = np.mean(np.linalg.norm(source - weightedTarget, axis=1))
    errors.append(err)
    CenteredSource=source-SourceCentroid    
    CenteredTarget=weightedTarget-TargetCentroid # Nx2 to 2xN

    W=CenteredSource.T@CenteredTarget


    U,S,Vt=np.linalg.svd(W)
    R= Vt.T@U.T
    t=TargetCentroid-R@SourceCentroid

    source = (R @ source.T).T + t

    R_total=R @ R_total          
    t_total=R @ t_total + t  


source=data['source'].copy()
source = (R_total @ source.T).T + t_total
plt.scatter(source[:, 0], source[:, 1], c='blue', label='Source')

print(R_total,t_total)

print(errors[-1])

print("done")
plt.figure()
plt.plot(errors, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Mean Error")
plt.title("ICP Error vs Iteration")
plt.grid(True)

plt.show()
plt.pause(10000)





