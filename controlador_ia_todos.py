# === Importación de librerías necesarias ===
import os
import torch
import torch.nn as nn
import joblib
import traci
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

# === Configuración general ===
CONFIG_FILE = "simulacion2.sumocfg"  # Archivo de configuración de SUMO
INTERVALO_DECISION = 5  # Intervalo de tiempo para tomar decisiones en segundos
DURACION_VERDE_MIN = 10  # Duración mínima de luz verde en segundos
DURACION_VERDE_MAX = 50  # Duración máxima de luz verde en segundos
CARPETA_RESULTADOS = "resultados_finales"  # Carpeta donde se guardarán los resultados
ios.makedirs(CARPETA_RESULTADOS, exist_ok=True)  # Crea la carpeta si no existe

# === Definición de la red neuronal convolucional para controlar semáforos ===
class SemaforoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2),  # Capa convolucional con 16 filtros
            nn.ReLU(),  # Función de activación
            nn.Flatten()  # Aplana la salida para conectarse con la capa totalmente conectada
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 9, 64),  # Capa totalmente conectada con 64 neuronas
            nn.ReLU(),
            nn.Linear(64, 2)  # Capa de salida con 2 neuronas (2 posibles decisiones)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# === Definición de los semáforos y sus carriles asociados ===
SEMAFOROS = {
    "semaforo1": {
        "calles": {
            "BenignoMalo_llegada_Bolivar": ["malo1_0"],
            "Bolivar_entre_Cordero_Malo": [
                "337277970#0_0", "337277970#0_1", ":12013799527_0_0", ":12013799527_0_1",
                "bolivar9_0", "bolivar9_1", ":12013799529_0_0", ":12013799529_0_1",
                "337277970#2_0", "337277970#2_1"
            ]
        }
    },
    "semaforo2": {
        "calles": {
            "BenignoMalo_entre_Bolivar_Sucre": [
                "337277951#0_0", "337277951#0_1", ":12013799530_0_0", ":12013799530_0_1",
                "malo2_1", "malo2_0", ":620530034_0_0", ":620530034_0_1",
                "malo3_1", "malo3_0", ":12013799524_0_0", ":12013799524_0_1",
                "malo4_1", "malo4_0", ":12013799523_0_0", ":12013799523_0_1",
                "malo5_1", "malo5_0"
            ],
            "Sucre_llegada_BenignoMalo": ["sucre1_0"]
        }
    },
    "semaforo3": {
        "calles": {
            "LuisCordero_llegada_Sucre": ["cordero4_0"],
            "Sucre_entre_Malo_y_Cordero": [
                "sucre2_1", "sucre2_0", ":12013799485_0_0", ":12013799485_0_1",
                "sucre3_0", "sucre3_1", ":12013799525_0_1", ":12013799525_0_0",
                "sucre4_0", "sucre4_1"
            ]
        }
    },
    "semaforo4": {
        "calles": {
            "Bolivar_llegada_Cordero": ["bolivar10_0"],
            "LuisCordero_entre_Bolivar_Sucre": [
                "cordero1_0", "cordero1_1", ":12013799528_0_0", ":12013799528_0_1",
                "central4#3_1", "central4#3_0", ":12013799521_0_0", ":12013799521_0_1",
                ":12013799526_0_0", ":12013799526_0_1", "cordero3_0", "cordero3_1"
            ]
        }
    }
}

# === Carga de modelos entrenados y escaladores por semáforo ===
for s_id in SEMAFOROS:
    modelo = SemaforoCNN()  # Se instancia la red neuronal
    modelo_path = f"modelos/modelo_{s_id}.pt"  # Ruta al modelo
    scaler_path = f"modelos/scaler_{s_id}.pkl"  # Ruta al escalador
    modelo.load_state_dict(torch.load(modelo_path))  # Carga del modelo
    modelo.eval()  # Se pone el modelo en modo evaluación
    scaler = joblib.load(scaler_path)  # Carga del escalador
    # Se guardan en el diccionario de cada semáforo
    SEMAFOROS[s_id]["modelo"] = modelo
    SEMAFOROS[s_id]["scaler"] = scaler
    SEMAFOROS[s_id]["fase"] = 0  # Fase inicial
    SEMAFOROS[s_id]["t_ultima"] = 0  # Última vez que se cambió de estado
    SEMAFOROS[s_id]["proximo"] = 0  # Tiempo para el próximo cambio
    SEMAFOROS[s_id]["csv"] = []  # Datos recopilados

# === Variables para monitorear los semáforos ===
semaphore_ids = list(SEMAFOROS.keys())
monitor_data = {
    sem_id: {
        "tiempos_por_estado": defaultdict(float),
        "tiempos_por_color": {"Rojo": 0.0, "Amarillo": 0.0, "Verde": 0.0},
        "transiciones": []  # Transiciones de colores con tiempo y duración
    }
    for sem_id in semaphore_ids
}

# === Variables auxiliares ===
prev_states = {sem_id: None for sem_id in semaphore_ids}  # Estado anterior de cada semáforo
change_times = {sem_id: 0.0 for sem_id in semaphore_ids}  # Tiempo del último cambio

# === Para el conteo total de vehículos ===
time_steps = []
vehicle_counts = []

# === Recolección de datos por carril de cada calle ===
def recolectar_datos(calles):
    datos_modelo = []
    datos_visuales = {}
    for nombre, carriles in calles.items():
        total_veh, total_vel, total_cola, total_espera, total_ocup = 0, 0, 0, 0.0, 0
        for lane in carriles:
            vehs = traci.lane.getLastStepVehicleIDs(lane)
            n = len(vehs)
            total_veh += n
            total_vel += traci.lane.getLastStepMeanSpeed(lane) * n
            total_cola += traci.lane.getLastStepHaltingNumber(lane)
            total_ocup += traci.lane.getLastStepOccupancy(lane)
            for v in vehs:
                total_espera += traci.vehicle.getWaitingTime(v)
        vel_prom = total_vel / total_veh if total_veh > 0 else 0
        espera_prom = total_espera / total_veh if total_veh > 0 else 0
        ocup = total_ocup / len(carriles)
        datos_modelo.extend([total_veh, vel_prom, total_cola, espera_prom, ocup])
        datos_visuales[nombre] = {
            "espera": espera_prom,
            "cola": total_cola
        }
    return datos_modelo, datos_visuales
