import traci
import pandas as pd

CONFIG_FILE = "simulacion2.sumocfg"
CSV_BASE = "resultados_semaforo{}_estatico.csv"
INTERVALO_DECISION = 5  # igual que en IA

# Diccionario de semáforos y sus calles observadas
SEMAFOROS = {
    "semaforo1": ["BenignoMalo_llegada_Bolivar", "Bolivar_entre_Cordero_Malo"],
    "semaforo2": ["BenignoMalo_entre_Bolivar_Sucre", "Sucre_llegada_BenignoMalo"],
    "semaforo3": ["LuisCordero_llegada_Sucre", "Sucre_entre_Malo_y_Cordero"],
    "semaforo4": ["Bolivar_llegada_Cordero", "LuisCordero_entre_Bolivar_Sucre"],
}

def recolectar_datos():
    traci.start(["sumo", "-c", CONFIG_FILE])
    tiempo = 0
    datos_por_semaforo = {k: [] for k in SEMAFOROS}

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        tiempo += 1

        if tiempo % INTERVALO_DECISION == 0:
            for semaforo, calles in SEMAFOROS.items():
                fila = {"tiempo": tiempo, "prediccion": None, "fase_aplicada": traci.trafficlight.getPhase(semaforo)}
                for calle in calles:
                    cola = traci.edge.getLastStepHaltingNumber(calle)
                    espera = traci.edge.getWaitingTime(calle)
                    fila[f"{calle}_cola"] = cola
                    fila[f"{calle}_espera"] = espera
                datos_por_semaforo[semaforo].append(fila)

    traci.close()

    for semaforo, datos in datos_por_semaforo.items():
        df = pd.DataFrame(datos)
        df.to_csv(CSV_BASE.format(semaforo[-1]), index=False)

if __name__ == "__main__":
    recolectar_datos()

