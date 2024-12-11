
import matplotlib.pyplot as plt
import numpy as np

dat0 = np.load("gt_exe_path0")
dat1 = np.load("gt_exe_path1")
dat2 = np.load("gt_exe_path2")
dat3 = np.load("gt_exe_path3")
dat4 = np.load("gt_exe_path4")
dat5 = np.load("gt_exe_path5")
dat6 = np.load("gt_exe_path6")

plt.subplot(2,3,1)
plt.scatter(dat0[:,0],dat0[:,1])
plt.subplot(2,3,2)
plt.scatter(dat1[:,0],dat1[:,1])
plt.subplot(2,3,3)
plt.scatter(dat2[:,0],dat2[:,1])
plt.subplot(2,3,4)
plt.scatter(dat3[:,0],dat3[:,1])
plt.subplot(2,3,5)
plt.scatter(dat4[:,0],dat4[:,1])
plt.subplot(2,3,6)
plt.scatter(dat5[:,0],dat5[:,1])

plt.show()