from scipy import misc
import numpy as np
import glob
from matplotlib import pyplot as plt

N = 10
Y = np.array([211, 301, 271, 121, 31, 341, 401, 241, 181, 301])
# Force it to be a column vector
Y = np.reshape(Y, (10, 1))
images = []

for filename in glob.glob('./Webcam/MontBlanc*.png'):
    print(filename)
    im = misc.imread(filename)
    images.append(im)

green = []

for image in images:
    width = image.shape[0]
    height = image.shape[1]
    pixelAmount = width * height
    greenness = np.sum(image[:, :, 1]) / pixelAmount
    green.append(greenness)

# With dummy feature
Xg = np.array(green)
Xg = Xg.reshape((10,1))
Xg = np.concatenate((np.ones(10)[:, np.newaxis], Xg), axis=1)
XgT = np.transpose(Xg)
print(Xg)

dot = np.dot(XgT, Xg)
print(dot)

lam2 = 2
lam5 = 5
I = np.ones(2)

e1 = dot + lam2 * I
e2 = dot + lam5 * I
e = np.dot(XgT, Y)

Wopt2 = np.dot(np.linalg.inv(e1), e)
Wopt5 = np.dot(np.linalg.inv(e2), e)

Hy1 = np.dot(Xg, Wopt2)
Hy2 = np.dot(Xg, Wopt5)


# Standardize feature vector
stdGreen = (green - np.mean(green, axis=0)) / np.std(green, axis=0)

# Plot
ax = plt.subplot(111)
plt.scatter(stdGreen, Y)
ax.plot(stdGreen, Hy1, label='λ=2')
ax.plot(stdGreen, Hy2, label='λ=5')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.xlabel('Normalized Greenness')
plt.ylabel('Daytime (duration in minutes from 7am)')
labels = ['{0}'.format(i) for i in range(1, N + 1)]
for label, x, y in zip(labels, stdGreen, Y):
    plt.annotate(
        label,
        xy=(x, y), xytext=(5, 5),
        textcoords='offset points', ha='right', va='bottom', color="grey")
plt.show()