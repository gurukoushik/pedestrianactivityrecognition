import matplotlib.pyplot as plt

resnet18_accuracy = [0.852,0.875,0.873,0.854,0.856,0.869]
resnet18_localization = [0.0308,0.0371,0.0414,0.0476,0.0454,0.0461]
resnet18_frozen3D_accuracy=[]
resnet18_frozen3D_localization=[]
resnet50_accuracy = []
resnet50_localization = []
resnext100_accuracy = []
resnext100_localization = []

plt.figure(1)
plt.plot(resnet18_accuracy,'o-')
plt.legend(['resnet18'])

plt.figure(2)
plt.plot(resnet18_localization,'o-')
plt.legend(['resnet18'])

plt.show()