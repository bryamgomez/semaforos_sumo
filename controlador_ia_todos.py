import os
import torch
import torch.nn as nn
import joblib
import traci
import pandas as pd

CONFIG_FILE = "simulacion2.sumocfg"
INTERVALO_DECISION = 5
DURACION_VERDE_MIN = 10
DURACION_VERDE_MAX = 50
CARPETA_RESULTADOS = "resultados_finales"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

# === CNN utilizada para todos los semáforos ===
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

# === Ejecutar simulación
traci.start(["sumo", "-c", CONFIG_FILE])
print("🚦 Simulación iniciada con control inteligente...")

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    tiempo = traci.simulation.getTime()

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

traci.close()

# === Exportar CSVs
for sem_id, info in SEMAFOROS.items():
    df = pd.DataFrame(info["csv"])
    nombre_archivo = os.path.join(CARPETA_RESULTADOS, f"{sem_id}_resultados_ia.csv")
    df.to_csv(nombre_archivo, index=False)
    print(f"✅ Datos exportados: {nombre_archivo}")

