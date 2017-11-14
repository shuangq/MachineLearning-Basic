from scipy import misc
import numpy as np
import glob
from matplotlib import pyplot as plt

N = 10
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


# With dummy feature

# ones = np.ones(N, dtype=np.int)
# XgT = np.array(green)
# XgT = np.vstack((XgT, ones))
# Xg = np.transpose(XgT)
#
# e1 = np.dot(XgT, Xg)
# inv1 = np.linalg.inv(e1)
# e2 = np.dot(XgT, Y)
# Wopt = np.dot(inv1, e2)
# Hy = np.dot(Xg, Wopt)


# Without dummy feature

XgT = np.array(green)
Xg = np.transpose(XgT)
e1 = np.dot(XgT, Xg)
e2 = np.dot(XgT, Y)
Wopt = e2 / e1      # Since wopt only has one value, without the dummy feature it is non invertible.
print(Wopt)
Hy = Wopt * Xg

print(Hy)


# Plot
plt.scatter(green, Y, marker="o")
plt.plot(green, Hy, color="green")
plt.xlabel('Normalized Greenness')
plt.ylabel('Daytime (duration in minutes from 7am)')

labels = ['{0}'.format(i) for i in range(1, N + 1)]
for label, x, y in zip(labels, green, Y):
    plt.annotate(
        label,
        xy=(x, y), xytext=(5, 5),
        textcoords='offset points', ha='right', va='bottom', color="grey")

plt.show()