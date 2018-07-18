from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import Classes as cs
import Functions as fnc



noise_std = 0.15
path = []
laser = []
estimation = []
time = []

Laser = cs.Laser_Measurement(noise_variance= noise_std)
frame_num = 250

for i in range(0,frame_num):
    dt = 0.1
    t = dt * i
    time.append(t)


    ground_truth = fnc.path_generator((t))
    measurement = Laser.Measure(ground_truth)


    path.append(ground_truth)
    laser.append(measurement)

    if (i == 0):
        # Initiate the Kalman Filter
        kf = cs.kalman_filter
        kf = fnc.Kalman_Filter_Initiator(kf, measurement, dt, noise_std)

        estimation.append([kf.X[0], kf.X[1]])
    else:
        kf.Predict()
        kf.Update(measurement)
        estimation.append([kf.X[0], kf.X[1]])


path = np.array(path)
laser = np.array(laser)
estimation = np.array(estimation)

# Creat the Animation

fig = plt.figure()
ax1 = plt.axes(xlim=(-15, 15), ylim=(-15,15))
line, = ax1.plot([], [], lw=2)
plt.xlabel('X')
plt.ylabel('Y')

lines = []

lobj_path, = ax1.plot([],[],lw=2,color= "black")
lobj_laser, = ax1.plot([],[],'o' ,lw=2,color= "red")
lobj_est, = ax1.plot([],[],'*',lw=2,color= "blue")

lines = [lobj_path,lobj_laser,lobj_est]


def init():
    for line in lines:
        line.set_data([],[])
    return lines

x1,y1 = [],[]
x2,y2 = [],[]
x3,y3 = [],[]


def animate(i):

    x = path[i,0]
    y = path[i,1]

    x1.append(x)
    y1.append(y)

    x = laser[i,0]
    y = laser[i,1]

    x2.append(x)
    y2.append(y)

    x = estimation[i,0]
    y = estimation[i,1]

    x3.append(x)
    y3.append(y)



    xlist = [x1, x2, x3]
    ylist = [y1, y2, y3]

    #for index in range(0,1):
    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.

    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=250, interval=10, blit = True)

plt.show()