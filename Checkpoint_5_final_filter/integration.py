'''import numpy as np

'''
'''CALCULATING VELOCITY AND POSITION'''
'''

class MATHS:
    def __init__(self, dt) -> None:
        self._dt = dt
         
    def accelertaion_function(self, t, accel):
        return accel

    def rk4_step(self, f, t, y, dt, acc_linear):
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def integrate_rk4(self, acc_linear):

        velocity = np.zeros((len(acc_linear), 3))
        position = np.zeros((len(acc_linear), 3))

        for i in range(len(acc_linear)):
            acc = acc_linear[i]
            velocity[i] = self.rk4_step(self.accelertaion_function, i*self._dt, velocity[i-1], self._dt, acc)
            position[i] = self.rk4_step(self.accelertaion_function, i*self._dt, position[i-1], self._dt, velocity[i])
        print(velocity)
        return velocity, position'''
        
import numpy as np

class RK4Integrator:
    def __init__(self, dt):
        """
        Initializes the RK4 Integrator with a given time step.
        
        Args:
            dt: Time step for integration (e.g., 0.01 for 10ms)
        """
        self.dt = dt

    def acceleration_function(self, acc_mus_in_array):
        """
        
        Args:
            accel: The current acceleration vector (ax, ay, az).
        
        Returns:
            accel: The acceleration vector.
        """
        accel = acc_mus_in_array
        return accel

    def rk4_step(self, f, y, dt, accel):
        """
        Performs a single Runge-Kutta 4th order (RK4) step.
        
        Args:
            f: The function to integrate (e.g., acceleration or velocity).
            y: The previous value (velocity or position).
            dt: Time step.
            accel: The acceleration vector for this step.
        
        Returns:
            The next integrated value of y.
        """
        
        k1 = f(accel)
        k2 = f(accel + dt / 2 * k1)
        k3 = f(accel + dt / 2 * k2)
        k4 = f(accel + dt * k3)
        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrate(self, acc_mus_in_array):
        """
        Performs RK4 integration over acceleration data to compute velocity and position.
        
        Args:
            acceleration_data: A numpy array of acceleration data (n x 3) where n is 
                               the number of time steps, and 3 corresponds to (ax, ay, az).
        
        Returns:
            velocity: Integrated velocity (n x 3).
            position: Integrated position (n x 3).
        """
        n = len(acc_mus_in_array)

        # Initialize velocity and position arrays (same size as acceleration)
        velocity = np.zeros((n, 3))  # Velocity in 3D space
        position = np.zeros((n, 3))  # Position in 3D space

        # Time integration loop
        for i in range(1, n):
            acc = acc_mus_in_array[i]

            # Update velocity using RK4 (integrating acceleration)
            velocity[i] = self.rk4_step(lambda a: a, velocity[i-1], self.dt, acc)

            vel = np.array(velocity)
            # Update position using RK4 (integrating velocity)
            position[i] = self.rk4_step(lambda v: v, position[i-1], self.dt, velocity[i])

        return velocity, position
