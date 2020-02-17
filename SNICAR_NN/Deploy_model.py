import

# create arrays of random values for each param
cz = np.random.uniform(low=0.01, high=0.99, size=(1000,1000))
al = np.random.uniform(low = 0, high = 1000000, size=(1000,1000))
du = np.random.uniform(low=0, high = 1000000, size = (1000,1000))
rd = np.random.uniform(low=300, high=10000, size=(1000,1000))
rh = np.random.uniform(low=200, high=900, size=(1000,1000))
so = np.random.uniform(low=0, high=3000, size=(1000,1000))

output = np.zeros(shapoe=(1000,1000))

for i in range(1000):
    for j in range(1000):
        output[i,j] = model.predict(data[:,i,j])
        