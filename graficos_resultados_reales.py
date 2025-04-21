import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Ruta base actual
carpeta_base = r"C:\Users\braya\Desktop\proyecto vehiculares"

# Directorios específicos
carpeta_estatico = os.path.join(carpeta_base, "semaforo_estatico")
carpeta_ia = os.path.join(carpeta_base, "resultados_finales")

# Lista de semáforos
semaforos = ["semaforo1", "semaforo2", "semaforo3", "semaforo4"]

def graficar_barras(nombre, df_ia, df_estatico):
    tiempo = df_ia["tiempo"]
    columnas_cola = [col for col in df_ia.columns if "_cola" in col]
    columnas_espera = [col for col in df_ia.columns if "_espera" in col]

    for col in columnas_cola:
        indices = np.arange(len(tiempo))
        ancho = 0.4

        plt.figure(figsize=(12, 5))
        plt.bar(indices - ancho/2, df_estatico[col], width=ancho, label="Estático", color="orange", alpha=0.7)
        plt.bar(indices + ancho/2, df_ia[col], width=ancho, label="IA", color="blue", alpha=0.7)

        plt.title(f"{nombre.upper()} - Colas ({col.replace('_cola','')})")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Cantidad de Vehículos")
        plt.xticks(indices[::max(1, len(indices)//10)], tiempo[::max(1, len(tiempo)//10)], rotation=45)
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    for col in columnas_espera:
        indices = np.arange(len(tiempo))
        ancho = 0.4

        plt.figure(figsize=(12, 5))
        plt.bar(indices - ancho/2, df_estatico[col], width=ancho, label="Estático", color="orange", alpha=0.7)
        plt.bar(indices + ancho/2, df_ia[col], width=ancho, label="IA", color="blue", alpha=0.7)

        plt.title(f"{nombre.upper()} - Espera ({col.replace('_espera','')})")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Tiempo de Espera (s)")
        plt.xticks(indices[::max(1, len(indices)//10)], tiempo[::max(1, len(tiempo)//10)], rotation=45)
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

def main():
    for nombre in semaforos:
        path_ia = os.path.join(carpeta_ia, f"{nombre}_resultados_ia.csv")
        path_estatico = os.path.join(carpeta_estatico, f"{nombre}_estatico.csv")

        if os.path.exists(path_ia) and os.path.exists(path_estatico):
            df_ia = pd.read_csv(path_ia)
            df_estatico = pd.read_csv(path_estatico)
            print(f"📊 Comparando (IA vs Estático): {nombre}")
            graficar_barras(nombre, df_ia, df_estatico)
        else:
            print(f"❌ Archivos faltantes para {nombre}")

if __name__ == "__main__":
    main()
