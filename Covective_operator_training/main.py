from model import Model
import matplotlib.pyplot as plt


m = Model()
m.plot_2D_parameter_space()
m.train_NN()
plt.show()