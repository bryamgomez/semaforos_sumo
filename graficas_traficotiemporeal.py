import traci
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

# Configuración SUMO (sin --end)
sumo_config = [
    'sumo-gui',
    '-c', 'C:/Users/braya/Desktop/Nueva carpeta (2)/configuracion.sumocfg',
    '--step-length', '0.10',
    '--lateral-resolution', '0'
]

# Iniciar la conexión con SUMO
traci.start(sumo_config)

# Semáforos a monitorear
semaphore_ids = ["semaforo1", "semaforo2", "semaforo3", "semaforo4"]
data = {
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

try:
    tiempo_total = 0.0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        duracion_paso = traci.simulation.getDeltaT()
        
        # Registrar datos de vehículos
        vehicle_count = traci.vehicle.getIDCount()
        time_steps.append(tiempo_total)
        vehicle_counts.append(vehicle_count)
        
        # Monitorear semáforos
        for sem_id in semaphore_ids:
            estado_actual = traci.trafficlight.getRedYellowGreenState(sem_id)
            
            if prev_states[sem_id] != estado_actual:
                if prev_states[sem_id] is not None:
                    tiempo_estado = tiempo_total - change_times[sem_id]
                    data[sem_id]["transiciones"].append({
                        "color": prev_states[sem_id],
                        "duracion": tiempo_estado,
                        "inicio": change_times[sem_id],
                        "fin": tiempo_total
                    })
                change_times[sem_id] = tiempo_total
                prev_states[sem_id] = estado_actual
            
            data[sem_id]["tiempos_por_estado"][estado_actual] += duracion_paso
            
            for luz in estado_actual:
                if luz in ['r', 'R']:
                    data[sem_id]["tiempos_por_color"]["Rojo"] += duracion_paso / len(estado_actual)
                elif luz in ['y', 'Y']:
                    data[sem_id]["tiempos_por_color"]["Amarillo"] += duracion_paso / len(estado_actual)
                elif luz in ['g', 'G']:
                    data[sem_id]["tiempos_por_color"]["Verde"] += duracion_paso / len(estado_actual)
        
        tiempo_total += duracion_paso

    # Registrar el último estado de cada semáforo
    for sem_id in semaphore_ids:
        if prev_states[sem_id] is not None:
            tiempo_estado = tiempo_total - change_times[sem_id]
            data[sem_id]["transiciones"].append({
                "color": prev_states[sem_id],
                "duracion": tiempo_estado,
                "inicio": change_times[sem_id],
                "fin": tiempo_total
            })

finally:
    traci.close()

# Funciones de graficación (sin cambios)
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
    plt.savefig('tiempos_por_color.png', dpi=300)
    plt.show()

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
    plt.savefig('transiciones_semaforos.png', dpi=300)
    plt.show()

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
    plt.savefig('proporciones_semaforos.png', dpi=300)
    plt.show()

def plot_vehicle_count(time_steps, vehicle_counts):
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, vehicle_counts, 'b-', linewidth=2)
    plt.title('Cantidad de vehículos en la red vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Número de vehículos')
    plt.grid(True)
    plt.savefig('conteo_vehiculos.png', dpi=300)
    plt.show()

# Generar gráficas y CSV
print("Procesando datos y generando gráficos...")

# Gráficos de semáforos
plot_time_by_color(data, semaphore_ids)
plot_transitions(data, semaphore_ids)
plot_pie_charts(data, semaphore_ids)

# Gráfico de vehículos
plot_vehicle_count(time_steps, vehicle_counts)

# Guardar datos de semáforos en CSV
with open('periodos_semaforos.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Semáforo', 'Color', 'Duración (s)', 'Inicio (s)', 'Fin (s)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for sem_id in semaphore_ids:
        for transicion in data[sem_id]["transiciones"]:
            writer.writerow({
                'Semáforo': sem_id,
                'Color': transicion["color"],
                'Duración (s)': f"{transicion['duracion']:.2f}",
                'Inicio (s)': f"{transicion['inicio']:.2f}",
                'Fin (s)': f"{transicion['fin']:.2f}"
            })

# Guardar datos de vehículos en CSV
with open('conteo_vehiculos.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Tiempo (s)', 'Número de vehículos'])
    for t, v in zip(time_steps, vehicle_counts):
        writer.writerow([f"{t:.2f}", v])

print("Análisis completado:")
print("- Datos de semáforos guardados en 'periodos_semaforos.csv'")
print("- Datos de vehículos guardados en 'conteo_vehiculos.csv'")
print("- Gráficas guardadas como:")
print("  * tiempos_por_color.png")
print("  * transiciones_semaforos.png")
print("  * proporciones_semaforos.png")
print("  * conteo_vehiculos.png")