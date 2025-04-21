import os
import traci
import pandas as pd

CONFIG_FILE = "simulacion2.sumocfg"  # actualiza si tiene otro nombre
CSV_SALIDA = "dataset_semaforo1_benigno.csv"

CALLES_SEMAFORO1 = {
    "BenignoMalo_llegada_Bolivar": ["malo1_0"],
    "Bolivar_entre_Cordero_Malo": [
        "337277970#0_0", "337277970#0_1", ":12013799527_0_0", ":12013799527_0_1",
        "bolivar9_0", "bolivar9_1", ":12013799529_0_0", ":12013799529_0_1",
        "337277970#2_0", "337277970#2_1"
    ]
}

def decidir_etiqueta(stats):
    # Puedes ajustar esta lógica de decisión si quieres priorizar otro criterio
    return 0 if stats["BenignoMalo_llegada_Bolivar"]["espera"] > stats["Bolivar_entre_Cordero_Malo"]["espera"] else 1

datos_finales = []

traci.start(["sumo", "-c", CONFIG_FILE])
for _ in range(7000):
    traci.simulationStep()
    tiempo = traci.simulation.getTime()

    stats = {}
    for calle, carriles in CALLES_SEMAFORO1.items():
        total_veh = total_vel = total_cola = total_espera = total_occup = 0
        for lane in carriles:
            vehs = traci.lane.getLastStepVehicleIDs(lane)
            n = len(vehs)
            total_veh += n
            total_vel += traci.lane.getLastStepMeanSpeed(lane) * n
            total_cola += traci.lane.getLastStepHaltingNumber(lane)
            total_occup += traci.lane.getLastStepOccupancy(lane)
            for v in vehs:
                total_espera += traci.vehicle.getWaitingTime(v)
        vel_prom = total_vel / total_veh if total_veh > 0 else 0
        espera_prom = total_espera / total_veh if total_veh > 0 else 0
        stats[calle] = {
            "veh": total_veh,
            "vel": vel_prom,
            "cola": total_cola,
            "espera": espera_prom,
            "occup": total_occup / len(carriles)
        }

    etiqueta = decidir_etiqueta(stats)
    fila = [
        tiempo,
        stats["BenignoMalo_llegada_Bolivar"]["veh"],
        stats["BenignoMalo_llegada_Bolivar"]["vel"],
        stats["BenignoMalo_llegada_Bolivar"]["cola"],
        stats["BenignoMalo_llegada_Bolivar"]["espera"],
        stats["BenignoMalo_llegada_Bolivar"]["occup"],
        stats["Bolivar_entre_Cordero_Malo"]["veh"],
        stats["Bolivar_entre_Cordero_Malo"]["vel"],
        stats["Bolivar_entre_Cordero_Malo"]["cola"],
        stats["Bolivar_entre_Cordero_Malo"]["espera"],
        stats["Bolivar_entre_Cordero_Malo"]["occup"],
        etiqueta
    ]
    datos_finales.append(fila)

traci.close()

df = pd.DataFrame(datos_finales, columns=[
    "tiempo",
    "malo_veh", "malo_vel", "malo_cola", "malo_espera", "malo_occup",
    "bolivar_veh", "bolivar_vel", "bolivar_cola", "bolivar_espera", "bolivar_occup",
    "etiqueta"
])
df.to_csv(CSV_SALIDA, index=False)
print(f"✅ Dataset guardado como: {CSV_SALIDA}")

