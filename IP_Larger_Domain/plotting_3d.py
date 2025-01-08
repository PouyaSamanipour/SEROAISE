# create the figure
import matplotlib.pyplot as plt
import numpy as np
import random
fig = plt.figure()

# add axes
# ax = fig.add_subplot(111,projection='3d')

xx, yy = np.meshgrid([random.uniform(-3, 3) for i in range(100)], [random.uniform(-3, 3) for i in range(100)])
yy1,zz1 = np.meshgrid([random.uniform(-3, 3) for i in range(100)], [random.uniform(-3, 3) for i in range(100)])
# z = (9 - xx - yy) / 2 
z=-(0.89307296*xx+0.8073526*yy-0.7706109881401062)/0.96461904
# array([0.89307296, 0.8073526 , 0.96461904])
# z1=2+0*xx
# z2=-2+0*xx
# x1=2+0*yy1
# x2=-2+0*yy1
# y1=2+0*xx
y2=-2+0*xx
# plot the plane
ax = plt.axes(projection='3d')
# ax.plot_surface(xx, yy, z)
# ax.plot_surface(xx, y2, zz1)
# ax.plot_surface(xx, yy, z2)
# ax.plot_surface(x1, yy1, zz1)
# ax.plot_surface(x2, yy1, zz1)
# ax.plot_surface(xx, y1, zz1)
# ax.plot_surface(xx, y2, zz1)
plt.plot(-0.4011980552974227, -1.7773231558170122, 2.0,'r*')
plt.plot(-0.6881786136897554, -1.6844755451511886, 2.0,'k*')
plt.plot(-0.5869290226413802, -1.5085115350210059, 1.7109952043601253,'g*')
plt.plot(-1.4645218262984707, -1.8721822035271736, 2.0,'y*')
# plt.plot(0.0, -1.4350942689434705, 2.0,'c*')
plt.plot(-1.5113864393549683, -1.8614593479129145, 2.0,'*')
plt.plot(-1.4644896198802606, -1.8269253820366131, 1.9541713594395125,'*')
# plt.plot(0.0, -1.2943799094089816, 1.547169097347295,'*')
# plt.plot(2.0, -2.0, 1.5314044103067495,'*')
# plt.plot(2.0, -1.979845158860625, 2.0,'*')


# plt.plot(2,-1.97,2,'r*')


plt.xlim(-2,2)
plt.ylim(-2,2)
ax.set_zlim(-2,2)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


