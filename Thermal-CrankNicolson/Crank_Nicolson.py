import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# The AnimationPlotter class allows for the creation of an animated plot of temperature data over
# time.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AnimationPlotter:
    def __init__(self, temperatures):
        self.temperatures = temperatures

    def animate(self, interval=1, save=False, reduce_update=10, dt = None):
        fig, ax = plt.subplots()
        frames_to_show = len(self.temperatures)
        
        def update(frame):
            ax.clear()
            ax.plot(self.temperatures[frame])
            if dt == None:
                ax.set_title(f'Time Step {frame}')
            else:
                ax.set_title(f'Time (seconds) {frame * dt:.2f}')
            ax.set_xlabel('Point')
            ax.set_ylabel('Temperature')
            ax.set_ylim(np.min(self.temperatures[frame]) - 100, np.max(self.temperatures[frame]) + 100)
        
        reduced_frames = range(0, frames_to_show, reduce_update)
        ani = FuncAnimation(fig, update, frames=reduced_frames, interval=interval)
        
        if save:
            ani.save('column_animation.gif', writer='pillow')
        
        plt.show()


class Crank_Nicolson():
    def __init__(self,
                 L : float = 1,
                 alpha : float = 0.1) -> None:
        self.L = L
        self.alpha = alpha
        self.x = []
        self.T = []

    def create_A(self, 
                 n_points : int,
                 dt : float,
                 dx : float):
        
        """
        The function `create_A` creates a matrix `A` of size `n_points x n_points` with specific values
        based on the input parameters `n_points`, `dt`, and `dx`.
        
        :param n_points: The parameter `n_points` represents the number of points in the grid or the size of
        the matrix `A`. It determines the number of rows and columns in the matrix `A`
        :type n_points: int
        :param dt: The parameter `dt` represents the time step size in the numerical method being used. It
        determines the size of the time intervals at which the system is being simulated
        :type dt: float
        :param dx: The parameter `dx` represents the spatial step size or the distance between two adjacent
        points in the spatial domain
        :type dx: float
        :return: a numpy array A, which is a square matrix of size n_points x n_points.
        """
        
        A = np.zeros((n_points, n_points))

        for i in range(n_points):
            A[i][i] = 2 * (dx ** 2 / (dt * self.alpha) + 1)
            
            if i > 0 and i != (n_points - 1):
                A[i][i - 1] = -1
            if i < n_points - 1 and i != 0:
                A[i][i + 1] = -1
        
        A[0][0] = A[n_points - 1][n_points - 1] = 1
        return A
    
    def create_b(self, n_points : int, dx : float, dt : float, T : np.array):
        """
        The function `create_b` calculates the values of the array `b` based on the given parameters
        `n_points`, `dx`, `dt`, and `T`.
        
        :param n_points: The parameter `n_points` represents the number of points in the system. It
        determines the size of the array `b`
        :type n_points: int
        :param dx: The parameter `dx` represents the spatial step size, which is the distance between two
        adjacent points in the spatial domain
        :type dx: float
        :param dt: The parameter `dt` represents the time step size in the numerical simulation. It
        determines how much time elapses between each iteration of the simulation
        :type dt: float
        :param T: T is a numpy array representing the temperature values at each point in the system. It has
        shape (1, n_points), where n_points is the number of points in the system
        :type T: np.array
        """
        b = np.zeros((n_points,))
        for i in range(n_points):
            if i == 0 or i == n_points - 1:
                b[i] = T[i]
            elif 1 <= i <= n_points - 2:
                b[i] = T[i - 1] + 2 * (dx ** 2 / (self.alpha * dt) - 1) * T[i] + T[i + 1]
                
        return b


    def solve(self,
                Tm : float = 300,
                T0 : float = 300,
                Tl : float = 1000,
                n_points : float = 100,
                time_steps = 1000) -> list[float]:
        """
        The function `solve` calculates the temperature distribution over time in a 1D system using the
        finite difference method.
        
        :param Tm: Tm is the temperature at the middle point of the rod, defaults to 300
        :type Tm: float (optional)
        :param T0: T0 is the initial temperature at the left boundary of the system, defaults to 300
        :type T0: float (optional)
        :param Tl: Tl is the temperature at the right boundary of the system, defaults to 1000
        :type Tl: float (optional)
        :param n_points: The parameter `n_points` represents the number of points or grid cells used to
        discretize the domain. It determines the spatial resolution of the problem. A higher value of
        `n_points` will result in a finer grid and more accurate results, but it will also increase the
        computational cost, defaults to 100
        :type n_points: float (optional)
        :param time_steps: The parameter "time_steps" represents the number of time steps to be taken in the
        simulation. It determines how many iterations of the solver will be performed to calculate the
        temperature distribution over time, defaults to 1000 (optional)
        """
        self.dx = self.L/n_points
        self.dt = 0.5 * self.dx ** 2 /(2 * self.alpha)
        initial_T = np.zeros((n_points))

        #calculate the initial temperature
        for point in range(n_points):
            if point == 0:
                initial_T[point] = T0
            if 0 < point < n_points - 1:
                initial_T[point] = 0# Tm*np.sin(np.pi/self.L*dx*point) + (Tl - T0)/self.L*dx*point + T0
            if point == n_points - 1:
                initial_T[point] = Tl
        self.T.append(initial_T)

        #create A matrix, constant
        self.A = self.create_A(n_points, self.dt, self.dx)

        for step in range(time_steps):
            self.b = self.create_b(n_points, self.dx, self.dt, self.T[step])
            self.T.append(np.linalg.solve(a= self.A, b= self.b))

if __name__ == "__main__":
    problema = Crank_Nicolson()
    problema.solve(time_steps= 10000)
    plotter = AnimationPlotter(problema.T)
    plotter.animate(interval=10, save=True, reduce_update=20, dt= problema.dt)