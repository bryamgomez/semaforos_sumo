import pandas as pd
import matplotlib.pyplot as plt

# Diccionario con los archivos CSV
archivos = {
    "Semáforo 1": "semaforo1_resultados_ia.csv",
    "Semáforo 2": "semaforo2_resultados_ia.csv",
    "Semáforo 3": "semaforo3_resultados_ia.csv",
    "Semáforo 4": "semaforo4_resultados_ia.csv"
}

def graficar_semaforo(nombre, df):
    tiempo = df['tiempo']
    
    # Gráfica de fases: predicción vs fase aplicada
    plt.figure()
    plt.plot(tiempo, df['prediccion'], label='Predicción IA')
    plt.plot(tiempo, df['fase_aplicada'], label='Fase Aplicada', linestyle='--')
    plt.title(f"{nombre} - Fase: Predicción vs Aplicada")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Fase")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Gráfica de colas
    columnas_cola = [col for col in df.columns if "_cola" in col]
    if columnas_cola:
        plt.figure()
        for col in columnas_cola:
            plt.plot(tiempo, df[col], label=col.replace("_cola", ""))
        plt.title(f"{nombre} - Colas de Vehículos")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Cantidad de Vehículos")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    # Gráfica de esperas
    columnas_espera = [col for col in df.columns if "_espera" in col]
    if columnas_espera:
        plt.figure()
        for col in columnas_espera:
            plt.plot(tiempo, df[col], label=col.replace("_espera", ""))
        plt.title(f"{nombre} - Tiempo de Espera")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Tiempo (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

# Cargar y graficar todos los CSV
for nombre, ruta in archivos.items():
    df = pd.read_csv(ruta)
    graficar_semaforo(nombre, df)

plt.show()

