import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

# The AnimationPlotter class allows for the creation of an animated plot of temperature data over
# time.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AnimationPlotter:
    def __init__(self, temperatures, analitic_T):
        self.temperatures = temperatures[1:]
        self.analitic_T = analitic_T

    def animate(self, interval=1, save=False, reduce_update=10, dt = None):
        """
        The `animate` function creates an animation of temperature data over time and optionally saves it as
        a GIF.
        
        :param interval: The `interval` parameter determines the delay between frames in milliseconds. It
        controls the speed of the animation, defaults to 1 (optional)
        :param save: The `save` parameter is a boolean value that determines whether or not to save the
        animation as a GIF file. If `save` is set to `True`, the animation will be saved as
        "column_animation.gif". If `save` is set to `False`, the animation will not be saved, defaults to
        False (optional)
        :param reduce_update: The `reduce_update` parameter determines how many frames to skip between each
        update in the animation. For example, if `reduce_update` is set to 10, then only every 10th frame
        will be shown in the animation. This can be useful if you have a large number of frames and,
        defaults to 10 (optional)
        :param dt: The parameter `dt` represents the time step between each frame in the animation. It is
        used to calculate the time in seconds for each frame when setting the title of the plot. If `dt` is
        not provided (i.e., it is set to `None`), the title will display the
        """
        fig, ax = plt.subplots()
        frames_to_show = len(self.temperatures)
        
        def update(frame):
            ax.clear()
            ax.plot(self.temperatures[frame])
            #ax.plot(self.analitic_T[frame])
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
                 alpha : float = 0.1,
                 Tm : float = 300,
                 T0 : float = 300,
                 Tl : float = 1000,
                 n_points : int = 100) -> None:
        self.L = L
        self.alpha = alpha
        self.x = []
        self.T = []
        self.analitic_T = []
        self.error = []
        self.n_points = n_points
        self.Tm = Tm
        self.T0 = T0
        self.Tl = Tl

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
    
    def analitic_solve(self, t : float, dx : float):
        """
        The `analitic_solve` function calculates the temperature distribution at different points in a 1D
        domain using an analytical solution.
        
        :param t: The parameter "t" represents time
        :type t: float
        :param dx: dx is the spatial step size, which represents the distance between adjacent points in the
        spatial domain
        :type dx: float
        :return: an array `T` which represents the temperature distribution at each point in the system.
        """
        T = np.zeros((self.n_points))
        for step in range(self.n_points):
            T[step] = self.Tm*np.exp(-t*self.alpha*(np.pi/self.L)**2)*np.sin(np.pi/self.L*step*dx) + (self.Tl - self.T0)/self.L*step*dx + self.T0
        return T


    def solve(self,
                simulation_time = 3,
                factor_p : float = 1) -> list[float]:
        
        """
        The function `solve` calculates the temperature distribution over time using a finite difference
        method and compares it to an analytical solution, plotting the error over time.
        
        :param simulation_time: The `simulation_time` parameter represents the total time for which the
        simulation will run. It is measured in arbitrary units, defaults to 3 (optional)
        :param factor_p: The `factor_p` parameter is a float that is used to adjust the time step size
        (`dt`) in the simulation. It is multiplied by the default value of `0.5 * self.dx ** 2 /(2 *
        self.alpha)` to calculate the actual value of `dt`. By adjusting, defaults to 1
        :type factor_p: float (optional)
        """
        
        self.dx = self.L/self.n_points 
        self.dt = 0.5 * self.dx ** 2 /(2 * self.alpha) * factor_p
        time_steps = int(simulation_time/self.dt)
        self.error = {"time" : [], "error" : []}

        initial_T = np.zeros((self.n_points))

        #calculate the initial temperature
        for point in range(self.n_points):
            if point == 0:
                initial_T[point] = self.T0
            if 0 < point < self.n_points - 1:
                initial_T[point] = 0 #self.Tm*np.sin(np.pi/self.L*self.dx*point) + (self.Tl - self.T0)/self.L*self.dx*point + self.T0
            if point == self.n_points - 1:
                initial_T[point] = self.Tl
        self.T.append(initial_T)

        #create A matrix, constant
        self.A = self.create_A(self.n_points, self.dt, self.dx)

        for step in range(time_steps):
            self.b = self.create_b(self.n_points, self.dx, self.dt, self.T[step])

            self.T.append(np.linalg.solve(a= self.A, b= self.b))
            self.analitic_T.append(self.analitic_solve(t = step*self.dt, dx= self.dx))

            self.error['error'].append(abs(np.sum(self.T[step] - self.analitic_T[step])/time_steps))
            self.error['time'].append(step*self.dt)

        plt.plot(self.error['time'], self.error['error'])
        plt.ylabel("Error")
        plt.xlabel("Time (seconds)")


        # with open("dados.json", "w") as arquivo_json:
        #     json.dump(self.error, arquivo_json, indent=4)


        

if __name__ == "__main__":
    problema = Crank_Nicolson(n_points= 50)
    problema.solve(simulation_time= 3, factor_p= 1)
    print(problema.dt, problema.dt)
    plotter = AnimationPlotter(problema.T, problema.analitic_T)
    plotter.animate(interval=10, save=True, reduce_update=20, dt= problema.dt)