from scipy import misc
import glob
from matplotlib import pyplot as plt
import numpy as np

###########################################
#
# 要死了a=4*10e-9, 跑了200k得到的ER约等于为112
#
###########################################


N = 10
y = np.array([211, 301, 271, 121, 31, 341, 401, 241, 181, 301])
# Force y to be column vector
y = np.reshape(y, (10, 1))

# Read input
images = []

for filename in glob.glob('./Webcam/MontBlanc*.png'):
    im = misc.imread(filename)
    images.append(im)


gVectors = []

for image in images:
    gIndensity = image[0:100, 0:100, 1]
    gVector = np.reshape(gIndensity, 10000)
    gVectors.append(gVector)

# @TODO: Standardize feature vector? --> hmmm probably not
# stdgVec = (gVectors - np.mean(gVectors, axis=0)) / np.std(gVectors, axis=0)

# Set parameters
alpha1 = 1 * 10 ** (-10)
alpha2 = 1 * 10 ** (-9)
alpha3 = 4 * 10 ** (-9)

# Initialize
w = np.zeros((10000, 1))
w2 = np.zeros((10000, 1))
w3 = np.zeros((10000, 1))

Xg = np.array(gVectors)
XgT = np.transpose(Xg)
XTX = np.dot(XgT, Xg)
XTy = np.dot(XgT, y)

# NOTE: Instead of doing XTX•w, do XT•Xw, since computing XTX takes centuries
Xw = np.dot(Xg, w)
Xw2 = np.dot(Xg, w)
Xw3 = np.dot(Xg, w)


def calGradient(xw):
    # return 2 / N * (np.dot(XTX, w) - XTy)
    return 2 / N * (np.dot(XgT, xw) - XTy)


def calEmpiricalRisk(xw):
    risk = np.dot(np.transpose(xw - y), (xw - y)) / N
    return risk[0]


risks1 = []
risks2 = []
risks3 = []

# Run iterations
i = 0
while i < 200000:
    print(i)
    # Calculate gradient
    gradient = calGradient(Xw)
    gradient2 = calGradient(Xw2)
    gradient3 = calGradient(Xw3)

    # Update w
    w = w - alpha1 * gradient
    w2 = w2 - alpha2 * gradient2
    w3 = w3 - alpha3 * gradient3
    Xw = np.dot(Xg, w)
    Xw2 = np.dot(Xg, w2)
    Xw3 = np.dot(Xg, w3)

    # Calculate empirical risk
    risk1 = calEmpiricalRisk(Xw)
    risk2 = calEmpiricalRisk(Xw2)
    risk3 = calEmpiricalRisk(Xw3)

    # Store risk value in array
    risks1.append(risk1)
    risks2.append(risk2)
    risks3.append(risk3)

    # Update iteration
    i = i + 1



# Plot
iteration = range(1, 200001)

ax = plt.subplot(111)
ax.plot(iteration, risks1, label='a=1×10e-10')
ax.plot(iteration, risks2, label='a=1×10e-9')
ax.plot(iteration, risks3, label='a=4×10e-9')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.xlabel('Numbers of Iterations')
plt.ylabel('Empirical Risk')
plt.margins(y=0)

plt.show()


