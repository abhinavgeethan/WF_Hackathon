# Imports
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Loading Data into Array
with open("training_data.csv") as f:
  train_data=np.loadtxt(f, delimiter=",",dtype=str)

print(train_data.ndim)

# Converting Tenor to Days
def tenorToDays(T)->int:
  identifier=T[-1]
  amt=int(T[:-1])
  return amt*30 if identifier=='M' else amt*12*30

for i in range(1,train_data.shape[0]):
  tenor=tenorToDays(train_data[i][1])
  train_data[i][1]=tenor

# Visualising Surface
fig = plt.figure()
ax = plt.axes(projection='3d')
# tenors=np.array([float(train_data[i][1]) for i in range(1,train_data.shape[0])])
tenors=train_data[1:,1].astype(float)
# print(tenors)
# moneyness=np.array([float(train_data[0][i]) for i in range(2,21)])
moneyness=train_data[0,2:21].astype(float)
# print(moneyness)
# dates=[dt.strptime(train_data[i][0],"%m/%d/%Y") for i in range(1,train_data.shape[0])]
# dates=[x.timestamp() for x in dates]
# dates=np.array([int(round(x)) for x in dates])
iv=train_data[1:, 2:21].astype(float)
# print(iv)
X,Y=np.meshgrid(moneyness,tenors)
print(X.shape,Y.shape,iv.shape)
# ax.plot_wireframe(X,Y,iv,color="red")
ax.plot_surface(X,Y,iv,cmap="viridis",rstride=1,cstride=1,edgecolor="none")
plt.show()