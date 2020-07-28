# TDE - Clase practica del 20/07/2020, problema de la dimensionalidad. 
# Obtencion del vector y de muestras de dimension reducida 
# partir de una muestra D con 8 muestras de dimension 2

import numpy as np
import matplotlib.pyplot as plt


# 8 muestras de 2 features, 2 dimensiones
D = np.array([ [[1,2]],
               [[2,3]],
               [[3,2]],
               [[4,4]],
               [[5,4]],
               [[6,7]],
               [[7,6]],
               [[9,7]] ])

#### Metodo de reduccion de dimensionalidad ####
####
# Paso 1: calculo la media muestral
media = D.mean(axis=0)

# Paso 2: genero nueva muestra D'=D-media
D_prima = D - media

# Paso 3: calculo la matriz S
S = np.zeros([2,2])
for i in range(len(D)):
    S = S + D_prima[i]*D_prima[i].T
    
# Paso 4: calculo autovectores de S
eigvals, eigvecs = np.linalg.eig(S)

# Paso 5: busco el maximo autovalor
idx_max = np.argmax(eigvals)


# Paso 6: y_i = eigenvect_max*x_i
y = np.zeros(len(D))
for i in range(len(D)):
    y[i] = np.dot(eigvecs[idx_max],D[i,0])
    
    
# Para ver como estan las muestras
fig_1 = plt.figure(1)
plt.plot(D[:,:,0],D[:,:,1],'o',linewidth=1)
plt.xlabel('Feature x_1')
plt.ylabel('Feature x_2')
plt.title('Set de muestras en 2 dimensiones')
plt.grid()
plt.legend()
plt.show()
#fig_1.savefig('fig_set_muestras.png', bbox_inches='tight')

fig_2 = plt.figure(2)
plt.plot(len(y) * [0],y,'x',linewidth=1.3)
plt.title('Set de muestras luego de reducir la dimensionalidad')
plt.grid()
plt.show
#fig_2.savefig('fig_reduccion_dim.png', bbox_inches='tight')