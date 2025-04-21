Descripción de los Archivos

controlador_ia_todos.py
Este archivo controla el semáforo utilizando un modelo de Inteligencia Artificial (IA). Específicamente, implementa un sistema basado en redes neuronales convolucionales (CNN) para gestionar los semáforos de manera inteligente y optimizada, mejorando el flujo de tráfico.

graficar_traficodesderou.py
Este archivo se encarga de graficar el tráfico generado a partir del archivo trafico2.rou. Utiliza herramientas de visualización para mostrar el comportamiento del tráfico en diferentes momentos y condiciones.

graficas_semaforoytrafico.py
Este archivo realiza dos funciones clave:

Grafica el tráfico en tiempo real utilizando la interfaz Traci.

Genera gráficos detallados sobre el comportamiento de los semáforos y los tiempos de cambio de luces, proporcionando información visual para el análisis del rendimiento de los semáforos.

graficos_resultados_reales.py
Este archivo crea gráficos que muestran las comparaciones entre los resultados obtenidos en la simulación y los datos reales. Los gráficos generados se almacenan en la carpeta imágenes, facilitando el análisis y la evaluación del rendimiento del sistema.

recolectar_csv_semaforo1_manual.py
Este archivo recolecta los datos de los semáforos sin IA, los cuales se utilizarán para entrenar el modelo de IA en el archivo controlador_ia_todos.py. Los datos obtenidos en este proceso son esenciales para el entrenamiento del modelo de redes neuronales convolucionales (CNN), optimizando el rendimiento del sistema de semáforos controlado por IA.

