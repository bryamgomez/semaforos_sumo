# Importar las bibliotecas necesarias
import pandas as pd  # Para trabajar con los datos en formato CSV y manipulaciones de DataFrame
import matplotlib.pyplot as plt  # Para crear gráficos de barras
import os  # Para manejar las rutas de los archivos y directorios
import numpy as np  # Para trabajar con arrays y realizar cálculos numéricos

# Ruta base actual
# Definir la ruta base donde se encuentran los archivos y subdirectorios relacionados con el proyecto
carpeta_base = r"C:\Users\braya\Desktop\proyecto vehiculares"

# Directorios específicos para los resultados de semáforo estático e IA
carpeta_estatico = os.path.join(carpeta_base, "semaforo_estatico")  # Directorio con los datos estáticos
carpeta_ia = os.path.join(carpeta_base, "resultados_finales")  # Directorio con los resultados de IA

# Lista de semáforos
# Nombres de los semáforos que se utilizarán en el análisis y gráficos
semaforos = ["semaforo1", "semaforo2", "semaforo3", "semaforo4"]

# Función para graficar las barras comparando los resultados de IA y Estático
def graficar_barras(nombre, df_ia, df_estatico):
    tiempo = df_ia["tiempo"]  # Obtener la columna de tiempo
    # Filtrar las columnas que contienen datos de cola (vehículos esperando)
    columnas_cola = [col for col in df_ia.columns if "_cola" in col]
    # Filtrar las columnas que contienen datos de espera (tiempo de espera de vehículos)
    columnas_espera = [col for col in df_ia.columns if "_espera" in col]

    # Graficar las barras para las colas
    for col in columnas_cola:
        indices = np.arange(len(tiempo))  # Índices para las barras (de acuerdo al tiempo)
        ancho = 0.4  # Ancho de las barras

        plt.figure(figsize=(12, 5))  # Definir el tamaño de la figura del gráfico
        # Graficar las barras para el semáforo estático
        plt.bar(indices - ancho/2, df_estatico[col], width=ancho, label="Estático", color="orange", alpha=0.7)
        # Graficar las barras para el semáforo con IA
        plt.bar(indices + ancho/2, df_ia[col], width=ancho, label="IA", color="blue", alpha=0.7)

        # Personalizar el gráfico
        plt.title(f"{nombre.upper()} - Colas ({col.replace('_cola','')})")  # Título del gráfico
        plt.xlabel("Tiempo (s)")  # Etiqueta para el eje x (tiempo)
        plt.ylabel("Cantidad de Vehículos")  # Etiqueta para el eje y (cantidad de vehículos)
        plt.xticks(indices[::max(1, len(indices)//10)], tiempo[::max(1, len(tiempo)//10)], rotation=45)  # Configurar las etiquetas del eje x con tiempo
        plt.legend()  # Mostrar leyenda
        plt.grid(True, linestyle="--", linewidth=0.5)  # Añadir una cuadrícula al gráfico
        plt.tight_layout()  # Ajustar el diseño para evitar que se recorten las etiquetas
        plt.show()  # Mostrar el gráfico

    # Graficar las barras para los tiempos de espera
    for col in columnas_espera:
        indices = np.arange(len(tiempo))  # Índices para las barras (de acuerdo al tiempo)
        ancho = 0.4  # Ancho de las barras

        plt.figure(figsize=(12, 5))  # Definir el tamaño de la figura del gráfico
        # Graficar las barras para el semáforo estático
        plt.bar(indices - ancho/2, df_estatico[col], width=ancho, label="Estático", color="orange", alpha=0.7)
        # Graficar las barras para el semáforo con IA
        plt.bar(indices + ancho/2, df_ia[col], width=ancho, label="IA", color="blue", alpha=0.7)

        # Personalizar el gráfico
        plt.title(f"{nombre.upper()} - Espera ({col.replace('_espera','')})")  # Título del gráfico
        plt.xlabel("Tiempo (s)")  # Etiqueta para el eje x (tiempo)
        plt.ylabel("Tiempo de Espera (s)")  # Etiqueta para el eje y (tiempo de espera)
        plt.xticks(indices[::max(1, len(indices)//10)], tiempo[::max(1, len(tiempo)//10)], rotation=45)  # Configurar las etiquetas del eje x con tiempo
        plt.legend()  # Mostrar leyenda
        plt.grid(True, linestyle="--", linewidth=0.5)  # Añadir una cuadrícula al gráfico
        plt.tight_layout()  # Ajustar el diseño para evitar que se recorten las etiquetas
        plt.show()  # Mostrar el gráfico

# Función principal que procesa los archivos y genera los gráficos
def main():
    for nombre in semaforos:
        # Construir las rutas completas a los archivos CSV de resultados para cada semáforo
        path_ia = os.path.join(carpeta_ia, f"{nombre}_resultados_ia.csv")
        path_estatico = os.path.join(carpeta_estatico, f"{nombre}_estatico.csv")

        # Verificar si ambos archivos existen antes de procesarlos
        if os.path.exists(path_ia) and os.path.exists(path_estatico):
            # Leer los archivos CSV en DataFrames de pandas
            df_ia = pd.read_csv(path_ia)
            df_estatico = pd.read_csv(path_estatico)
            # Imprimir mensaje indicando que se están comparando los datos
            print(f"📊 Comparando (IA vs Estático): {nombre}")
            # Llamar a la función para graficar las barras para este semáforo
            graficar_barras(nombre, df_ia, df_estatico)
        else:
            # Imprimir mensaje si faltan los archivos
            print(f"❌ Archivos faltantes para {nombre}")

# Ejecutar la función principal
if __name__ == "__main__":
    main()
