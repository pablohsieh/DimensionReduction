# TDE - Clase practica del 27/07/2020, FISHER. 

import numpy as np
import matplotlib.pyplot as plt

# Set de muestras provenientes de las clases W1 y W2
W1 = np.array([ [1,2],
                [2,3],
                [3,3],
                [4,5],
                [5,5] ])
             
W2 = np.array([ [1,0],
                [2,1],
                [3,1],
                [3,2],
                [5,3],
                [6,5] ])
                              
n1 = len(W1)
n2 = len(W2)

#### Metodo de reduccion de dimensionalidad FISHER ###
####
# Paso 1: calculo la media muestral
media_1 = W1.mean(axis=0)
media_2 = W2.mean(axis=0)

# Paso 2: calculo de s1 y s2
cov_1 = np.cov(W1.T)
cov_2 = np.cov(W2.T)

S1 = cov_1 * (n1-1)
S2 = cov_2 * (n2-1)

# Paso 3: calculo de Sw
Sw = S1 + S2

# Paso 4: inversion de Sw
Sw_inv = np.linalg.inv(Sw)

# Paso 5: calculo de v
v = np.dot(Sw_inv,(media_1 - media_2))

#normalizo v
suma = 0
for i in range(len(v)):
    suma = suma + v[i]**2    
norma = np.sqrt(suma)

v_norm = v / norma

# Paso 6: calculo los y_i = v^T * w_i * V
Y1 = np.dot(v_norm,W1.T)
Y2 = np.dot(v_norm,W2.T)

# Para ver como estan las muestras
fig_1 = plt.figure(1)
plt.plot(W1[:,0],W1[:,1],'o',linewidth=1.2, label='Clase W1')
plt.plot(W2[:,0],W2[:,1],'x',linewidth=1.2, label='Clase W2')
plt.xlabel('Feature x_1')
plt.ylabel('Feature x_2')
plt.title('Set de muestras en 2 dimensiones')
plt.grid()
plt.legend()
plt.show()
#fig_1.savefig('fig_set_muestras.png', bbox_inches='tight')

fig_2 = plt.figure(2)
plt.plot(len(Y1) * [0],Y1,'o',linewidth=1.2, label='Clase W1')
plt.plot(len(Y2) * [0],Y2,'x',linewidth=1.2, label='Clase W2')
plt.title('Set de muestras luego de reducir la dimensionalidad')
plt.grid()
plt.legend()
plt.show
#fig_2.savefig('fig_reduccion_dim.png', bbox_inches='tight')