import numpy as np

# Lase Measurement Class
class Laser_Measurement:

    def __init__(self, noise_variance):
        self.variance = noise_variance

    def Measure(self, ground_thruth):
        x = ground_thruth[0] + self.variance*np.random.randn()
        y = ground_thruth[1] + self.variance*np.random.randn()

        return np.array([[x], [y]])


# Kalman Filter Class

class kalman_filter:

    def __init__(self, X_in, F_in, H_in, P_in, Q_in, R_in):

        self.X = np.array(X_in)
        self.F = np.array(F_in)
        self.H = np.array(H_in)
        self.P = np.array(P_in)
        self.Q = np.array(Q_in)
        self.R = np.array(R_in)

    def Predict(self):

        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.transpose() + self.Q



    def Update(self, Z):

        y = Z - self.H @ self.X
        S = self.H @ self.P @ self.H.transpose() + self.R
        K = self.P @ self.H.transpose() @ np.linalg.inv(S)
        I = np.identity(len(self.X))


        self.X = self.X + K @ y
        self.P = (I - K @ self.H) @ self.P