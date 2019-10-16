import pandas as pd
import numpy.polynomial.polynomial as poly
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# preprocessing requires the raw data file and calibration filename in the same folder as the script
filename = '2019-8-12 5X1 Delay0 XYZTF T460 E1 T1'
filename = '2019-8-13 5X1 Delay0 XYZTF T460 E1 T2'
filename = '2019-8-13 5X1 Delay0 XYZTF T460 E1 T3'
filename = '2019-8-29 5X1 Delay2 XYZTF T460 B1 T1'
filename = '2019-8-30 5X1 Delay2 XYZTF T460 B1 T1'
filename = '2019-9-1 5X1 Delay2 XYZTF T460 B1 T1'
filename = '2019-9-2 5X1 Delay2 XYZTF T460 B1 T1'
filename = '2019-9-15 5X1 Delay5 XYZTF T460 B1 E2'
calibration_file = '2019-08-12 21-25 E1 5X1 Board Calibration.npz'

data=pd.read_csv(filename + ".txt", sep=", ", header=None, engine='python')
data.columns = ["X", "Y", "Z", "Bx0", "By0", "Bz0", "Bt0", "Bx1", "By1", "Bz1", "Bt1", "Bx2", "By2", "Bz2", "Bt2", "Bx3", "By3", "Bz3", "Bt3","Bx4", "By4", "Bz4", "Bt4", "Bx5", "By5", "Bz5", "Bt5", "F"]

data['X'] = data['X'].apply(lambda x: x.replace('[', '').replace('X', '').replace('\'', '')).astype('float')
data['Y'] = data['Y'].apply(lambda x: x.replace('Y', '').replace('\'', '')).astype('float')
data['Z'] = data['Z'].apply(lambda x: x.replace('Z', '').replace('\'', '')).astype('float')
data['F'] = data['F'].apply(lambda x: x.replace(']', '')).astype('float')

print(data.head())
#print(data.dtypes)

# Save raw data to pandas dataframe
data.to_pickle("./" + str(filename) + " Raw.pkl")

# load calibration transforms from file
calibration_data = np.load(calibration_file)
A = calibration_data['affine']
offsets = calibration_data['offsets']
scales = calibration_data['scales']
# original_caldata = calibration_data['data']

# get Bx By Bz coords only
coords = np.asarray(data[["Bx0", "By0", "Bz0", 'Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', "Bx3", "By3", "Bz3", "Bx4", "By4", "Bz4", "Bx5", "By5", "Bz5"]])
#print("coords shape is: " ,coords.shape)

# scale magnetometers separately to spheres
scaled = np.multiply(scales,(coords-offsets))

scaled_data = pd.DataFrame({'Bx0': scaled[:, 0], 'By0': scaled[:, 1], 'Bz0': scaled[:, 2], 'Bx1': scaled[:, 3], 'By1': scaled[:, 4], 'Bz1': scaled[:, 5], 'Bx2': scaled[:, 6], 'By2': scaled[:, 7], 'Bz2': scaled[:, 8], 'Bx3': scaled[:, 9], 'By3': scaled[:, 10], 'Bz3': scaled[:, 11], 'Bx4': scaled[:, 12], 'By4': scaled[:, 13], 'Bz4': scaled[:, 14], 'Bx5': scaled[:, 15], 'By5': scaled[:, 16], 'Bz5': scaled[:, 17]})
scaled_data = pd.concat([data[["X", "Y", "Z"]], scaled_data, data[["F"]]], axis=1)
scaled_data.to_pickle("./" + str(filename) + " Scaled.pkl")
print(scaled_data.head())

# make coords homogeneous
scaled = np.insert(scaled, 3, 1, axis=1)
scaled = np.insert(scaled, 7, 1, axis=1)
scaled = np.insert(scaled, 11, 1, axis=1)
scaled = np.insert(scaled, 15, 1, axis=1)
scaled = np.insert(scaled, 19, 1, axis=1)
scaled = np.insert(scaled, 23, 1, axis=1)

# apply A to reference, subtract from signal
print(np.size(scaled,0))
print(np.size(scaled,1))
print(A)
transform_sig = np.zeros(shape=(np.size(scaled,0),20))
for i in range(1, 6):
    transform_sig[:, 4*(i-1):4*i] = scaled[:,0:4].dot(A[i-1,:,:])
    plt.subplot(121)
    plt.plot(scaled[:,4*i]-transform_sig[:,4*i-4], c="red")
    plt.plot(scaled[:,4*i+1]-transform_sig[:,4*i-3], c="green")
    plt.plot(scaled[:,4*i+2]-transform_sig[:,4*i-2], c="blue")
    plt.title('Scaled')
    plt.subplot(122)
    plt.plot(scaled[:,4*i]-transform_sig[:,4*i-4], c="red")
    plt.plot(scaled[:,4*i+1]-transform_sig[:,4*i-3], c="green")
    plt.plot(scaled[:,4*i+2]-transform_sig[:,4*i-2], c="blue")
    plt.title('Filtered')
    plt.show()
transform_sig = scaled[:,4:]-transform_sig

transform_data = pd.DataFrame({'Bx1': transform_sig[:, 0], 'By1': transform_sig[:, 1], 'Bz1': transform_sig[:, 2], 'Bx2': transform_sig[:, 4], 'By2': transform_sig[:, 5], 'Bz2': transform_sig[:, 6], 'Bx3': transform_sig[:, 8], 'By3': transform_sig[:, 9], 'Bz3': transform_sig[:, 10], 'Bx4': transform_sig[:, 12], 'By4': transform_sig[:, 13], 'Bz4': transform_sig[:, 14], 'Bx5': transform_sig[:, 16], 'By5': transform_sig[:, 17], 'Bz5': transform_sig[:, 18]})
transform_data = pd.concat([data[["X", "Y", "Z"]], transform_data, data[["F"]]], axis=1)
transform_data.to_pickle("./" + str(filename) + " SVD.pkl")
print(transform_data.head())