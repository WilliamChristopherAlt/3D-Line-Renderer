# PREREQUISITE: 
# + Quaternion to represent 3D rotation
# + Bresenham Algorithm
# + Projection from 3D to 2D plane

import numpy as np
from matplotlib import pyplot as plt
from math import floor, sin, cos, sqrt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import pickle
import os

frames = 1000

def animate(i, graph):
    print(f'{frames-i} frames remained')
    graph.rotate()
    plt.cla()
    graph.draw()
    
def create_animation(graph, dt):
    fig_animation = plt.figure()
    ax_animation = plt.axes()
    fig_animation.patch.set_facecolor('xkcd:black')
    ax_animation.set_facecolor([0.1, 0.15, 0.15])
    ax_animation.tick_params(axis='x', which='both', bottom=False,
                             top=False, labelbottom=False)
    ax_animation.tick_params(axis='y', which='both', right=False,
                             left=False, labelleft=False)
    plt.tight_layout()

    ani = FuncAnimation(fig_animation, animate, fargs=(graph, ), frames=frames , interval=dt)

    return ani

LINE_CHR = bytes((219,)).decode('cp437')

def plotLineLow(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0

    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    D = 2 * dy - dx
    y = y0

    noPointsDrawn = dx + 1
    xs = np.empty(noPointsDrawn, dtype=int)
    ys = np.empty(noPointsDrawn, dtype=int)

    for i, x in enumerate(range(x0, x1 + 1)):
        xs[i] = x
        ys[i] = y
        if D > 0:
            y += yi
            D = D + (2 * (dy - dx))
        else:
            D += 2 * dy

    return xs, ys

def plotLineHigh(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0

    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx

    D = 2 * dx - dy
    x = x0

    noPointsDrawn = dy + 1
    xs = np.empty(noPointsDrawn, dtype=int)
    ys = np.empty(noPointsDrawn, dtype=int)

    for i, y in enumerate(range(y0, y1 + 1)):
        xs[i] = x
        ys[i] = y
        if D > 0:
            x += xi
            D = D + (2 * (dx - dy))
        else:
            D += 2 * dx

    return xs, ys

def plotLine(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plotLineLow(x1, y1, x0, y0)
        else:
            return plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return plotLineHigh(x1, y1, x0, y0)
        else:
            return plotLineHigh(x0, y0, x1, y1)

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"
    
    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            # Quaternion multiplication
            return Quaternion(
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            )
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Unsupported operand type for multiplication")
        
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def normSquare(self):
        return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
    def norm(self):
        return sqrt(self.normSquare());

    def inverse(self):
        normSquare = self.normSquare()
        if normSquare == 0:
            raise ZeroDivisionError("Cannot invert a quaternion with zero norm")
        return self.conjugate() * (1.0 / normSquare)


class Graph:
    def __init__(self, minX, minY, minZ, maxX, maxY, maxZ, gridXSize, noPoints, axisList, rotateBy, points=None, connections=None):
        if minX >= maxX or minY >= maxY:
            raise ValueError

        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY

        self.lengthX = maxX - minX
        self.lengthY = maxY - minY

        self.gridXSize = gridXSize
        self.gridYSize = floor(gridXSize * self.lengthY / self.lengthX)

        self.scaleX = 1.0 / self.lengthX * self.gridXSize
        self.scaleY = 1.0 / self.lengthY * self.gridYSize

        self.noPoints = noPoints 
        self.axisList = axisList
        self.rotateBy = rotateBy

        if points is None and connections is None:
            randXs = np.random.uniform(minX, maxX, (noPoints))
            randYs = np.random.uniform(minY, maxY, (noPoints))
            randZs = np.random.uniform(minZ, maxZ, (noPoints))
            self.points = np.column_stack((randXs, randYs, randZs))

            self.connections = np.random.randint(0, 2, (noPoints, noPoints), dtype=bool)
            self.connections[np.tril_indices(noPoints, k=0)] = 0
        else:
            self.points = points
            self.connections = connections

        self.grid = np.zeros((self.gridYSize, self.gridXSize), dtype=bool)

    def rotate(self):
        for i in range(self.noPoints):
            for axis in self.axisList:
                rotate(self.points[i], axis, self.rotateBy)

    def draw(self, console=False):
        pointsProjected = self.points[:, :2]

        self.grid.fill(0)
        for i in range(self.noPoints):
            for j in range (self.noPoints):
                if i < j:
                    if self.connections[i, j]:
                        x0 = floor((pointsProjected[i][0] - self.minX) * self.scaleX)
                        y0 = floor((-pointsProjected[i][1] - self.minY) * self.scaleY)
                        x1 = floor((pointsProjected[j][0] - self.minX) * self.scaleX)
                        y1 = floor((-pointsProjected[j][1] - self.minY) * self.scaleY)
                        xs, ys = plotLine(x0, y0, x1, y1)

                        valid_indices = (xs >= 0) & (xs < self.gridXSize) & (ys >= 0) & (ys < self.gridYSize)
                        xs = xs[valid_indices]
                        ys = ys[valid_indices]

                        self.grid[ys, xs] = True
        
        if console:
            self.printGrid()
        else:
            plt.imshow(self.grid, cmap='grey')

    def printGrid(self):
        os.system('cls')
        for i in range(self.gridYSize):
            for j in range(self.gridXSize):
                print(LINE_CHR if self.grid[i, j] else ' ', end='')
            print()

# Rotate p around axis(normalize) by angle
def rotate(p, axis, angle):
    quaternion = Quaternion(0, p[0], p[1], p[2])

    # Construct the quaternion for rotation
    sinVal = sin(angle / 2.0)
    q = Quaternion(cos(angle / 2.0), sinVal * axis[0], sinVal * axis[1], sinVal * axis[2])

    # q = Quaternion(cos(angle / 2.0), 0.0, 0.0, 0.0) + axis * sin(angle / 2.0)
    rotated = q * quaternion * q.inverse()
    p[0] = rotated.x
    p[1] = rotated.y
    p[2] = rotated.z

# Project p to plane with normal vector n and passes through p0
def project(p, n, p0):
    return p - n * (n @ (p0 - p) / (-(n**2).sum()))

def main(): 
    noPoints = 12
    minX = -2
    minY = -2
    minZ = -2
    maxX = 2
    maxY = 2 
    maxZ = 2
    gridXSize = 500
    rotateBy = 0.005

    dt = 50 # Milliseconds

    # with open(r'C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\icosahedron.pkl', 'wb') as f:
    #     pickle.dump({'points': points, 'connections': connections}, f)

    with open(r'C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\icosahedron.pkl', 'rb') as f:
        objInfo = pickle.load(f)
    points = objInfo['points']
    connections = objInfo['connections']

    axisList = np.array([
        [1, 0, 0], [0, 1, 0], [0, 1, 1]
    ], dtype=float)

    for axis in axisList:
        axis = axis / np.linalg.norm(axis)

    graph = Graph(minX, minY, minZ, maxX, maxY, maxZ, gridXSize, noPoints, axisList, rotateBy, points, connections)

    ani = create_animation(graph, dt)
    # plt.show()
    FFwriter = animation.FFMpegWriter(fps=40)
    ani.save(r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\icosahedron.mp4", writer = FFwriter, dpi=200)

main()