import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Parsear el archivo XML
tree = ET.parse('C:/Users/braya/Desktop/proyecto vehiculares/trafico2.rou.xml')
root = tree.getroot()

# Definir los intervalos de tiempo (en segundos)
time_periods = [
    {"begin": 0.0, "end": 7200.0, "name": "6:00-8:00"},
    {"begin": 7200.0, "end": 23400.0, "name": "8:00-12:30"},
    {"begin": 23400.0, "end": 27000.0, "name": "12:30-13:30"},
    {"begin": 27000.0, "end": 41400.0, "name": "13:30-17:30"},
    {"begin": 41400.0, "end": 48600.0, "name": "17:30-19:30"}
]

# Función para calcular vehículos en un intervalo
def calculate_vehicles(begin, end, period):
    if float(period) <= 0:
        return 0
    return int((float(end) - float(begin)) / float(period))

# Procesar todos los flujos (flow) del XML
total_vehicles = [0] * len(time_periods)

for flow in root.findall('flow'):
    begin = flow.get('begin')
    end = flow.get('end')
    period = flow.get('period')
    
    for i, period_data in enumerate(time_periods):
        if float(begin) == period_data["begin"] and float(end) == period_data["end"]:
            total_vehicles[i] += calculate_vehicles(begin, end, period)

# Crear el gráfico
plt.figure(figsize=(12, 6))
bars = plt.bar(
    [period["name"] for period in time_periods],
    total_vehicles,
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
)

# Añadir etiquetas con el número de vehículos
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom'
    )

# Personalizar el gráfico
plt.xlabel('Intervalo de Tiempo', fontsize=12)
plt.ylabel('Cantidad de Vehículos', fontsize=12)
plt.title('Distribución de Vehículos por Intervalo Horario', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajustar márgenes
plt.tight_layout()

# Mostrar el gráfico
plt.show()