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
    def performICP(self,Source,Target):
        errors = []
        R_total = np.eye(2)
        t_total = np.zeros(2)
        for i in range(200):
            weightedTarget=self.getClosest(Source,Target)
            SourceCentroid=self.getCentroids(Source,weightedTarget)[0]
            TargetCentroid=self.getCentroids(Source,weightedTarget)[1]
            err = np.mean(np.linalg.norm(Source - weightedTarget, axis=1))
            errors.append(err)
            CenteredSource=Source-SourceCentroid    
            CenteredTarget=weightedTarget-TargetCentroid # Nx2 to 2xN

            W=CenteredSource.T@CenteredTarget


            U,S,Vt=np.linalg.svd(W)
            R= Vt.T@U.T
            t=TargetCentroid-R@SourceCentroid

            Source = (R @ Source.T).T + t

            R_total=R @ R_total          
            t_total=R @ t_total + t  
        return R_total,t_total, errors[len(errors)-1]




icp=ICP()


data = np.load('icp_dataset.npz')
# print(data['source'])
# print(data['target'])


print(icp.performICP(data['source'],data['target']))














