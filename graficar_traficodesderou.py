# Importar las bibliotecas necesarias
import xml.etree.ElementTree as ET  # Para analizar el archivo XML
import matplotlib.pyplot as plt  # Para crear el gráfico de barras

# Parsear el archivo XML
# Se carga y analiza el archivo XML que contiene los datos del flujo de tráfico.
tree = ET.parse('C:/Users/braya/Desktop/proyecto vehiculares/trafico2.rou.xml')
root = tree.getroot()

# Definir los intervalos de tiempo (en segundos)
# Los intervalos de tiempo se definen con un tiempo de inicio (begin) y fin (end),
# y un nombre que los representa para la visualización en el gráfico.
time_periods = [
    {"begin": 0.0, "end": 7200.0, "name": "6:00-8:00"},
    {"begin": 7200.0, "end": 23400.0, "name": "8:00-12:30"},
    {"begin": 23400.0, "end": 27000.0, "name": "12:30-13:30"},
    {"begin": 27000.0, "end": 41400.0, "name": "13:30-17:30"},
    {"begin": 41400.0, "end": 48600.0, "name": "17:30-19:30"}
]

# Función para calcular la cantidad de vehículos en un intervalo
# Calcula el número de vehículos en función de la duración del flujo y el periodo de tiempo.
def calculate_vehicles(begin, end, period):
    # Si el periodo es 0 o menor, no hay vehículos, se devuelve 0.
    if float(period) <= 0:
        return 0
    # El número de vehículos se calcula como la diferencia entre el tiempo de fin y comienzo dividido por el periodo.
    return int((float(end) - float(begin)) / float(period))

# Inicializar una lista para almacenar el total de vehículos por cada intervalo
total_vehicles = [0] * len(time_periods)  # Se crea una lista con ceros, uno por cada intervalo de tiempo

# Procesar todos los flujos (flow) del XML
# Se recorren todos los elementos 'flow' en el archivo XML para extraer los datos relevantes.
for flow in root.findall('flow'):
    begin = flow.get('begin')  # Obtiene el tiempo de inicio del flujo
    end = flow.get('end')  # Obtiene el tiempo de fin del flujo
    period = flow.get('period')  # Obtiene el periodo de tiempo de cada flujo
    
    # Verificar en qué intervalo se encuentra el flujo y sumar los vehículos en ese intervalo
    for i, period_data in enumerate(time_periods):
        # Si el flujo se encuentra dentro del intervalo actual, se calcula la cantidad de vehículos
        if float(begin) == period_data["begin"] and float(end) == period_data["end"]:
            total_vehicles[i] += calculate_vehicles(begin, end, period)

# Crear el gráfico de barras
# Usamos matplotlib para crear un gráfico de barras que muestre la cantidad de vehículos por intervalo.
plt.figure(figsize=(12, 6))  # Definir el tamaño del gráfico
bars = plt.bar(
    [period["name"] for period in time_periods],  # Nombres de los intervalos en el eje x
    total_vehicles,  # Cantidad de vehículos en cada intervalo
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Colores para cada barra
)

# Añadir etiquetas con el número de vehículos sobre cada barra
for bar in bars:
    height = bar.get_height()  # Obtiene la altura de la barra (número de vehículos)
    # Añadir el valor sobre la barra, centrado en la parte superior de la misma
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # Coordenada x (centrado)
        height,  # Coordenada y (en la parte superior de la barra)
        f'{int(height)}',  # Texto con el número de vehículos
        ha='center',  # Alineación horizontal
        va='bottom'  # Alineación vertical
    )

# Personalizar el gráfico
plt.xlabel('Intervalo de Tiempo', fontsize=12)  # Etiqueta del eje x
plt.ylabel('Cantidad de Vehículos', fontsize=12)  # Etiqueta del eje y
plt.title('Distribución de Vehículos por Intervalo Horario', fontsize=14, pad=20)  # Título del gráfico
plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x para mayor claridad
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añadir una cuadrícula horizontal en el eje y

# Ajustar los márgenes para que no se recorten los elementos
plt.tight_layout()

# Mostrar el gráfico
plt.show()
