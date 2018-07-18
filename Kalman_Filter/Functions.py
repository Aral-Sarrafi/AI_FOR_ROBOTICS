from math import sin, cos
import numpy as np
import Classes as cs

def path_generator(t):
    a = 0.5
    theta = a*t
    r = theta

    x = r*cos(theta)
    y = r*sin(theta)

    xdot = a*cos(a*t) - (a**2)*sin(a*t)
    ydot = a*sin(a*t) + (a**2)*cos(a*t)

    return  [x, y, xdot, ydot]

def Kalman_Filter_Initiator(kalman_filter, measurement, dt, noise_std):
    X = np.ones((4, 1))
    X[0, 0] = measurement[0, 0]
    X[1, 0] = measurement[1, 0]

    # State Transition Matrix for Constant Velocity Motion Model
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # Measurement Matrix
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    # State Covariance Matrix
    P = np.identity(len(X))
    P[0, 0] = 10
    P[1, 1] = 10
    P[2, 2] = 1000
    P[3, 3] = 1000
    # Process Noise Covariance Matrix
    noise_ax = 9
    noise_ay = 9
    dt2 = dt ** 2
    dt3 = dt ** 3
    dt4 = dt ** 4
    Q = np.array([[dt4 * noise_ax / 4, 0, dt3 * noise_ax / 2, 0],
                  [0, dt4 * noise_ay / 4, 0, dt3 * noise_ay / 2],
                  [dt4 * noise_ax / 4, 0, dt2 * noise_ax, 0],
                  [0, dt4 * noise_ay / 4, 0, dt2 * noise_ay]])
    # Measurement Noise Covariance Matrix
    R = noise_std * np.identity(len(measurement))

    # Initiate the Kalman Filter
    kalman_filter = cs.kalman_filter(X_in=X, F_in=F, H_in=H,
                          P_in=P, Q_in=Q, R_in=R)


    return kalman_filter