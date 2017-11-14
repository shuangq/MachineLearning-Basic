from scipy import misc
import glob
import numpy as np
from matplotlib import pyplot as plt

# Set parameters
N = 10
theta1 = 1
theta2 = 5
theta3 = 10
Y = np.array([211, 301, 271, 121, 31, 341, 401, 241, 181, 301])

images = []

for filename in glob.glob('./Webcam/MontBlanc*.png'):
    print(filename)
    im = misc.imread(filename)
    images.append(im)

green = []

for image in images:
    n_greenness = np.sum(image[:,:,1]) / (image.shape[0] * image.shape[1])
    green.append(n_greenness)


def calKernel(x, xi, theta):
    return np.exp((-0.5) * ((x - xi) / theta) ** 2 )


def calNormKernel(x, xi, theta):
    K1 = calKernel(x, xi, theta)
    K2 = 0
    for xl in green:
        K2 = K2 + calKernel(x, xl, theta)
    return K1/K2


# Define range of input values at which to approximate the function
xs = range(70, 131, 1)
predictY1 = []
predictY2 = []
predictY3 = []

for x in xs:
    pre_y1 = 0
    pre_y2 = 0
    pre_y3 = 0
    for i in range(0, N):
        xi = green[i]
        yi = Y[i]
        pre_y1 = pre_y1 + Y[i] * calNormKernel(x, xi, theta1)
        pre_y2 = pre_y2 + Y[i] * calNormKernel(x, xi, theta2)
        pre_y3 = pre_y3 + Y[i] * calNormKernel(x, xi, theta3)

    predictY1.append(pre_y1)
    predictY2.append(pre_y2)
    predictY3.append(pre_y3)


# Plot
ax = plt.subplot(111)
ax.plot(xs, predictY1, label='σ=1')
ax.plot(xs, predictY2, label='σ=5')
ax.plot(xs, predictY3, label='σ=10')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.scatter(green, Y)
plt.xlabel('Normalized Greenness')
plt.ylabel('Daytime (duration in minutes from 7am)')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)

plt.show()
