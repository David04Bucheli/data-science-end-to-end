import pandas as pd
import os
import matplotlib.pyplot as plt

os.makedirs('./starter/figuras', exist_ok=True) #carpeta que se crea con el programa, para que donde sea que se ejecute se puedan guardar las imágenes de forma ordenada (si existe, que no la trate de crear)

#cargo la data
data = pd.read_csv('./starter/viajes_taxi.csv')

print('Analisis exploratorio descriptivo')
#es importante saber como se esta haciendo parsing de los datos, para eso se puede usar el metodo info() de pandas
data.info() #muestra el tipo de dato de cada columna, si hay valores nulos y la cantidad de registros
print(data.dtypes) #muestra el tipo de dato de cada columna (dtypes es un atributo, no una función)
print(data.isna().sum()) #muestra la cantidad de valores nulos por columna
print(data.duplicated().sum()) #muestra la cantidad de registros duplicados

print(data.describe()) #muestra estadisticas descriptivas de las columnas numericas

#cada grafico de matplotlib es una "figura", abajo esta la estructura de un grafico
plt.figure(figsize=(10, 6)) #todo lo que esté abajo es una sola figura, hasta .close()
plt.hist(data['tarifa'], bins=50) #histograma de la columna trip_distance
plt.title('Histograma de tarifa') #titulo del grafico
plt.xlabel('Tarifa USD') #etiqueta del eje x
plt.ylabel('Numero de viajes') #etiqueta del eje y
plt.savefig('./starter/figuras/histograma_tarifa.png') #guarda la figura en la carpeta 'figuras'
plt.close() #muestra la figura

#grafico diagrama de caja
plt.figure(figsize=(10, 6)) #todo lo que esté abajo es una sola figura, hasta .close()
plt.boxplot(data['tarifa']) #histograma de la columna trip_distance
plt.title('Boxplot de tarifa') #titulo del grafico
plt.savefig('./starter/figuras/boxplot_tarifa.png') #guarda la figura en la carpeta 'figuras'
plt.close() #muestra la figura

#extracción de valores atipicos

q1=data['tarifa'].quantile(0.25) #cuartil 1
q3=data['tarifa'].quantile(0.75) #cuartil 3
iqr=q3-q1 #rango intercuartilico

limite_superior=q3+1.5*iqr #limite superior para valores atipicos
limite_inferior=q1-1.5*iqr #limite inferior para valores atipicos

atipicos = data.query(f'tarifa > {limite_superior} or tarifa < {limite_inferior}') # query() permite hacer consultas como un WHERE en SQL
sin_atipicos = data.query(f'tarifa <= {limite_superior} and tarifa >= {limite_inferior}') #la negación de la condición anterior, para obtener los datos sin valores atipicos
print(atipicos) #muestra los valores atipicos

#grafico de dispersión
plt.figure(figsize=(10, 6)) #todo lo que esté abajo es una sola figura, hasta .close()
plt.scatter(data['tarifa'], data['distancia_km']) #gráfico de dispersión de las columnas tarifa y distancia
plt.title('Dispersión de tarifa vs Distancia') #titulo del grafico
plt.xlabel('Tarifa USD') #etiqueta del eje x
plt.ylabel('Distancia') #etiqueta del eje y
plt.savefig('./starter/figuras/dispersión_tarifa_distancia.png') #guarda la figura en la carpeta 'figuras'
plt.close() #muestra la figura

#print(data.corr()) #muestra la correlación entre las columnas numéricas (matriz de correlación, por defecto pearson que se cambia con el parámetro method)

#grafico de dispersión
plt.figure() #todo lo que esté abajo es una sola figura, hasta .close()
plt.scatter(sin_atipicos['tarifa'], sin_atipicos['distancia_km']) #gráfico de dispersión de las columnas tarifa y distancia
plt.title('Dispersión de tarifa vs Distancia') #titulo del grafico
plt.xlabel('Tarifa USD') #etiqueta del eje x
plt.ylabel('Distancia') #etiqueta del eje y
plt.savefig('./starter/figuras/dispersión_tarifa_distancia_sin_atipicos.png') #guarda la figura en la carpeta 'figuras'
plt.close() #muestra la figura
