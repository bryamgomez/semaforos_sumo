import os
import torch
import torch.nn as nn
import joblib
import traci
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

# === Configuración ===
CONFIG_FILE = "simulacion2.sumocfg"
INTERVALO_DECISION = 5
DURACION_VERDE_MIN = 10
DURACION_VERDE_MAX = 50
CARPETA_RESULTADOS = "resultados_finales"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

# === CNN para los semáforos ===
class SemaforoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# === Configuración por semáforo ===
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

# === Cargar modelos y escaladores ===
for s_id in SEMAFOROS:
    modelo = SemaforoCNN()
    modelo_path = f"modelos/modelo_{s_id}.pt"
    scaler_path = f"modelos/scaler_{s_id}.pkl"
    modelo.load_state_dict(torch.load(modelo_path))
    modelo.eval()
    scaler = joblib.load(scaler_path)
    SEMAFOROS[s_id]["modelo"] = modelo
    SEMAFOROS[s_id]["scaler"] = scaler
    SEMAFOROS[s_id]["fase"] = 0
    SEMAFOROS[s_id]["t_ultima"] = 0
    SEMAFOROS[s_id]["proximo"] = 0
    SEMAFOROS[s_id]["csv"] = []

# === Variables para monitoreo de semáforos ===
semaphore_ids = list(SEMAFOROS.keys())
monitor_data = {
    sem_id: {
        "tiempos_por_estado": defaultdict(float),
        "tiempos_por_color": {"Rojo": 0.0, "Amarillo": 0.0, "Verde": 0.0},
        "transiciones": []
    }
    for sem_id in semaphore_ids
}

# Variables para rastrear cambios de estado
prev_states = {sem_id: None for sem_id in semaphore_ids}
change_times = {sem_id: 0.0 for sem_id in semaphore_ids}

# Variables para el conteo de vehículos
time_steps = []
vehicle_counts = []

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

# === Funciones de graficación ===
def plot_time_by_color(data, semaphore_ids):
    colors = ['Rojo', 'Amarillo', 'Verde']
    color_values = ['red', 'yellow', 'green']
    fig, ax = plt.subplots(figsize=(12, 6))
    
    width = 0.25
    x = range(len(semaphore_ids))
    
    for i, (color, cvalue) in enumerate(zip(colors, color_values)):
        values = [data[sem_id]["tiempos_por_color"][color] for sem_id in semaphore_ids]
        ax.bar([pos + width*i for pos in x], values, width, label=color, color=cvalue)
    
    ax.set_xticks([pos + width for pos in x])
    ax.set_xticklabels(semaphore_ids)
    ax.set_ylabel('Tiempo (s)')
    ax.set_title('Tiempo total por color de semáforo')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, 'tiempos_por_color.png'), dpi=300)
    plt.close()

def plot_transitions(data, semaphore_ids):
    fig, axs = plt.subplots(len(semaphore_ids), 1, figsize=(14, 8))
    if len(semaphore_ids) == 1:
        axs = [axs]
    
    color_map = {
        'r': 'red',
        'R': 'red',
        'y': 'yellow',
        'Y': 'yellow',
        'G': 'green',
        'g': 'green'
    }
    
    for i, sem_id in enumerate(semaphore_ids):
        for trans in data[sem_id]["transiciones"]:
            color = trans["color"][0]
            axs[i].barh(sem_id, trans["duracion"], left=trans["inicio"], 
                       color=color_map.get(color, 'gray'), edgecolor='black', height=0.5)
        
        axs[i].set_xlabel('Tiempo (s)')
        axs[i].set_ylabel('Semáforo')
        axs[i].set_title(f'Transiciones de {sem_id}')
        axs[i].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, 'transiciones_semaforos.png'), dpi=300)
    plt.close()

def plot_pie_charts(data, semaphore_ids):
    fig, axs = plt.subplots(1, len(semaphore_ids), figsize=(16, 5))
    if len(semaphore_ids) == 1:
        axs = [axs]
    
    for i, sem_id in enumerate(semaphore_ids):
        sizes = [data[sem_id]["tiempos_por_color"]["Rojo"],
                data[sem_id]["tiempos_por_color"]["Amarillo"],
                data[sem_id]["tiempos_por_color"]["Verde"]]
        labels = ['Rojo', 'Amarillo', 'Verde']
        colors = ['red', 'yellow', 'green']
        
        axs[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                  startangle=90, explode=(0.05, 0.05, 0.05), shadow=True)
        axs[i].set_title(f'Distribución de tiempos - {sem_id}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, 'proporciones_semaforos.png'), dpi=300)
    plt.close()

def plot_vehicle_count(time_steps, vehicle_counts):
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, vehicle_counts, 'b-', linewidth=2)
    plt.title('Cantidad de vehículos en la red vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Número de vehículos')
    plt.grid(True)
    plt.savefig(os.path.join(CARPETA_RESULTADOS, 'conteo_vehiculos.png'), dpi=300)
    plt.close()

# === Ejecutar simulación ===
traci.start(["sumo-gui", "-c", CONFIG_FILE, "--step-length", "0.10", "--lateral-resolution", "0"])
print("🚦 Simulación iniciada con control inteligente y monitoreo...")

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    tiempo = traci.simulation.getTime()
    
    # Registrar datos de vehículos
    vehicle_count = traci.vehicle.getIDCount()
    time_steps.append(tiempo)
    vehicle_counts.append(vehicle_count)
    
    # Monitorear estados de semáforos
    for sem_id in semaphore_ids:
        estado_actual = traci.trafficlight.getRedYellowGreenState(sem_id)
        
        if prev_states[sem_id] != estado_actual:
            if prev_states[sem_id] is not None:
                tiempo_estado = tiempo - change_times[sem_id]
                monitor_data[sem_id]["transiciones"].append({
                    "color": prev_states[sem_id],
                    "duracion": tiempo_estado,
                    "inicio": change_times[sem_id],
                    "fin": tiempo
                })
            change_times[sem_id] = tiempo
            prev_states[sem_id] = estado_actual
        
        monitor_data[sem_id]["tiempos_por_estado"][estado_actual] += traci.simulation.getDeltaT()
        
        for luz in estado_actual:
            if luz in ['r', 'R']:
                monitor_data[sem_id]["tiempos_por_color"]["Rojo"] += traci.simulation.getDeltaT() / len(estado_actual)
            elif luz in ['y', 'Y']:
                monitor_data[sem_id]["tiempos_por_color"]["Amarillo"] += traci.simulation.getDeltaT() / len(estado_actual)
            elif luz in ['g', 'G']:
                monitor_data[sem_id]["tiempos_por_color"]["Verde"] += traci.simulation.getDeltaT() / len(estado_actual)
    
    # Control inteligente de semáforos
    for sem_id, info in SEMAFOROS.items():
        entrada, visual = recolectar_datos(info["calles"])
        entrada_n = info["scaler"].transform([entrada])
        entrada_tensor = torch.tensor(entrada_n, dtype=torch.float32).unsqueeze(1)
        pred = info["modelo"](entrada_tensor).argmax().item()

        if tiempo >= info["t_ultima"] + INTERVALO_DECISION and tiempo >= info["proximo"]:
            if pred != info["fase"]:
                traci.trafficlight.setPhase(sem_id, pred)
                info["fase"] = pred
                info["proximo"] = tiempo + max(DURACION_VERDE_MIN, min(DURACION_VERDE_MAX, INTERVALO_DECISION * 2))
            info["t_ultima"] = tiempo

        # Guardar datos
        fila = {
            "tiempo": tiempo,
            "prediccion": pred,
            "fase_aplicada": info["fase"]
        }
        for nombre, datos in visual.items():
            fila[f"{nombre}_cola"] = datos["cola"]
            fila[f"{nombre}_espera"] = datos["espera"]
        info["csv"].append(fila)

# Registrar el último estado de cada semáforo
for sem_id in semaphore_ids:
    if prev_states[sem_id] is not None:
        tiempo_estado = tiempo - change_times[sem_id]
        monitor_data[sem_id]["transiciones"].append({
            "color": prev_states[sem_id],
            "duracion": tiempo_estado,
            "inicio": change_times[sem_id],
            "fin": tiempo
        })

traci.close()

# === Exportar datos y generar gráficos ===
print("Procesando datos y generando gráficos...")

# Exportar CSVs de control inteligente
for sem_id, info in SEMAFOROS.items():
    df = pd.DataFrame(info["csv"])
    nombre_archivo = os.path.join(CARPETA_RESULTADOS, f"{sem_id}_resultados_ia.csv")
    df.to_csv(nombre_archivo, index=False)
    print(f"✅ Datos de control exportados: {nombre_archivo}")

# Guardar datos de monitoreo de semáforos en CSV
with open(os.path.join(CARPETA_RESULTADOS, 'periodos_semaforos.csv'), 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Semáforo', 'Color', 'Duración (s)', 'Inicio (s)', 'Fin (s)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for sem_id in semaphore_ids:
        for transicion in monitor_data[sem_id]["transiciones"]:
            writer.writerow({
                'Semáforo': sem_id,
                'Color': transicion["color"],
                'Duración (s)': f"{transicion['duracion']:.2f}",
                'Inicio (s)': f"{transicion['inicio']:.2f}",
                'Fin (s)': f"{transicion['fin']:.2f}"
            })

# Guardar datos de vehículos en CSV
with open(os.path.join(CARPETA_RESULTADOS, 'conteo_vehiculos.csv'), 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Tiempo (s)', 'Número de vehículos'])
    for t, v in zip(time_steps, vehicle_counts):
        writer.writerow([f"{t:.2f}", v])

# Generar gráficos
plot_time_by_color(monitor_data, semaphore_ids)
plot_transitions(monitor_data, semaphore_ids)
plot_pie_charts(monitor_data, semaphore_ids)
plot_vehicle_count(time_steps, vehicle_counts)

print("Análisis completado:")
print("- Datos de semáforos guardados en 'periodos_semaforos.csv'")
print("- Datos de vehículos guardados en 'conteo_vehiculos.csv'")
print("- Gráficas guardadas en la carpeta resultados_finales:")
print("  * tiempos_por_color.png")
print("  * transiciones_semaforos.png")
print("  * proporciones_semaforos.png")
print("  * conteo_vehiculos.png")