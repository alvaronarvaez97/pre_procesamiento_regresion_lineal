#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import math 
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[2]:


datos=pd.read_csv('entrenamiento.csv', sep=',',names=["1","ID","2","data_time","3","latitud","4","longitud","5","frecuencia","6","presionatm"])


# In[3]:


datos_nuevos=datos[["ID","data_time","latitud","longitud","frecuencia"]]


# In[4]:


print(datos_nuevos)


# In[5]:


datos_nuevos["data_time"] = pd.to_datetime(datos_nuevos["data_time"], format="%d/%m/%y %H:%M:%S")
datos_nuevos=datos_nuevos.dropna()
datos_nuevos=datos_nuevos[(datos_nuevos.frecuencia != 0)]
datos_nuevos.to_csv("pre_procesado.csv", index=False)
print(datos_nuevos)


# In[9]:


def obtencion_velocidad(lat1,lon1,lat2,lon2):
    R = 6371 # Radius of the earth in km
    dLat = radians(lat2-lat1)
    dLon = radians(lon2-lon1)
    rLat1 = radians(lat1)
    rLat2 = radians(lat2)
    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * cos(rLat2) * sin(dLon/2) * sin(dLon/2) 
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    return d

def calculo_velocidad(distancia_Km, time_start, time_end):
    """Return 0 if time_start == time_end, avoid dividing by 0"""
    return distancia_Km/ (time_end - time_start).seconds if time_end > time_start else 0


# In[10]:


# First sort by ID and timestamp:
datos_nuevos = datos_nuevos.sort_values(by=['ID', 'data_time'])

# Group the sorted dataframe by ID, and grab the initial value for lat, lon, and time.
datos_nuevos['lat0'] = datos_nuevos.groupby('ID')['latitud'].transform(lambda x: x.iat[0])
datos_nuevos['lon0'] =datos_nuevos.groupby('ID')['longitud'].transform(lambda x: x.iat[0])
datos_nuevos['t0'] = datos_nuevos.groupby('ID')['data_time'].transform(lambda x: x.iat[0])


# In[15]:


# create a new column for distance
datos_nuevos['distancia_Km'] = datos_nuevos.apply(
    lambda row: obtencion_velocidad(
        lat1=row['latitud'],
        lon1=row['longitud'],
        lat2=row['lat0'],
        lon2=row['lon0']
    ),
    axis=1
)

# create a new column for velocity
datos_nuevos['velocidad_Kmps'] = datos_nuevos.apply(
    lambda row: calculo_velocidad(
        distancia_Km=row['distancia_Km'],
        time_start=row['t0'],
        time_end=row['data_time']
    ),
    axis=1
)


# In[20]:


print(datos_nuevos[['ID', 'data_time', 'latitud', 'longitud','frecuencia', 'distancia_Km', 'velocidad_Kmps']])


# In[30]:


df = pd.read_csv("pre_procesado.csv")
my_plot = df.plot("frecuencia", "velocidad_Kmps", kind="scatter")
plt.show() # no necesariamente en Jupyter Notebooks


# In[25]:


#df = pd.read_csv("nombre archivo.csv")

filtered_datos_nuevos = datos_nuevos[(datos_nuevos['frecuencia'] <= 3500) & (datos_nuevos["velocidad_Kmps"] <= 80000)]

dataX =filtered_datos_nuevos[["frecuencia"]]
X_train = np.array(dataX)
y_train = filtered_datos_nuevos["velocidad_Kmps"].values
 
# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()
 
# Entrenamos nuestro modelo
regr.fit(X_train, y_train)
 
# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, y_pred, color='black', linewidth=3)


plt.show()


# In[ ]:




