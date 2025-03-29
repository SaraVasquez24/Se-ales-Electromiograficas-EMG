# Se-ales-Electromiograficas-EMG

### Adquisición de Datos con NI-DAQmx 

#### Importante
Para la realización del código, hay que tener la siguiente libreria instalada en python.

```
pip install nidaqmx numpy matplotlib
```
#### Librerias
```
import nidaqmx
import numpy as np
import time
import matplotlib.pyplot as plt
```
`nidaqmx` Para interactuar con el hardware de National Instruments.

`numpy` Para almacenar y manipular los datos adquiridos.

`time` Para medir tiempos de ejecución.

`matplotlib.pyplot` Para graficar la señal adquirida.

#### 2. Configuración
```
DEVICE_NAME = "Dev1"      # Nombre del dispositivo DAQ
CHANNEL = "ai1"          # Canal de entrada analógica
SAMPLE_RATE = 1000       # Frecuencia de muestreo en Hz
DURATION = 120           # Duración de la adquisición en segundos (2 minutos)
FILENAME_NPY = "datos_señal.npy"  # Archivo para guardar en formato NumPy
```
`DEVICE_NAME` Nombre del dispositivo DAQ configurado en NI MAX.

`CHANNEL` Canal de entrada analógica donde se conectará la señal.

`SAMPLE_RATE` Frecuencia de muestreo en Hertzios (Hz).

`DURATION` Duración total de la adquisición en segundos.

`FILENAME_NPY` Nombre del archivo donde se guardarán los datos.

#### 3. Adquirir datos
```
def adquirir_datos():
    num_samples = SAMPLE_RATE * DURATION  # Número total de muestras
    data = np.zeros(num_samples)  # Inicializar array vacío
    
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(f"{DEVICE_NAME}/{CHANNEL}")
        task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=num_samples)

        print(f"Adquiriendo datos durante {DURATION // 60} minutos...")
        start_time = time.time()

        for i in range(DURATION):
            chunk = task.read(number_of_samples_per_channel=SAMPLE_RATE)  
            data[i * SAMPLE_RATE : (i + 1) * SAMPLE_RATE] = chunk  # Almacenar datos adquiridos
            elapsed = time.time() - start_time
            print(f"Progreso: {i+1}/{DURATION} segundos ({elapsed:.1f} s transcurridos)", end="\r")
    
    print("\n Adquisición completada.")
    return data
```
- Se configura y abre una tarea NI-DAQmx para adquirir datos de un canal analógico.

- Se establece una frecuencia de muestreo definida.

- Se adquiere la señal en segmentos de 1 segundo para optimizar la captura.

- Se almacena en un array de NumPy para su posterior análisis.

#### 4. Verificación de datos
```
def verificar_datos(data):
    if np.all(data == 0):
        print("Advertencia: No se detecta señal, verifica la conexión.")
        return False
    if np.std(data) < 0.001:
        print(" Advertencia: La señal parece demasiado estable, ¿es correcto?")
        return False
    print(" Señal detectada correctamente.")
    return True
```

- Se revisa si los datos adquiridos contienen solo ceros lo que posible fallo de conexión.

- Se evalúa si la señal es demasiado estable, lo que podría indicar una medición incorrecta.

- Si los datos son válidos, se confirma la correcta adquisición.

#### 5. Guardar datos

```
def guardar_datos_npy(data, filename):
    np.save(filename, data)
    print(f" Datos guardados en {filename}")
```
Guarda los datos adquiridos en un archivo `.npy` para facilitar su carga en Python.

#### 6. Visualización de la señal
```
def graficar_datos(data):
    tiempo = np.arange(0, len(data)) / SAMPLE_RATE  # Crear vector de tiempo

    plt.figure(figsize=(12, 5))
    
    if len(data) > 50000:
        step = len(data) // 50000
        plt.plot(tiempo[::step], data[::step], label="Señal adquirida (muestra reducida)", color='b', linewidth=0.5)
    else:
        plt.plot(tiempo, data, label="Señal adquirida", color='b', linewidth=0.5)

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Voltaje (V)")
    plt.title("Señal Adquirida desde DAQ")
    plt.legend()
    plt.grid(True)
    plt.show()
```
- Genera una gráfica de la señal adquirida.

- Si hay demasiados datos, selecciona una muestra representativa para evitar sobrecargar la visualización.

#### Ejecición

```
datos = adquirir_datos()
if verificar_datos(datos):
    guardar_datos_npy(datos, FILENAME_NPY)
    graficar_datos(datos)
```



### Análisis de Fatiga Muscular a partir de Señales EMG

Este código analiza señales de electromiografía (EMG) para evaluar la fatiga muscular mediante procesamiento de señales y análisis espectral. Se hace una división temporal de la señal, comparando el espectro de potencia de la primera y la segunda mitad para identificar cambios en la frecuencia dominante, los cuales pueden estar asociados con la fatiga muscular.

#### Importante
Para la realización del código, hay que tener la siguiente libreria instalada en python.

```
pip install numpy matplotlib scipy
```
Y también se necesita el archivo con los datos de la señal previamente obtenidos. ` datos_señal.npy`

#### 1. Importación de librerias

Se importan las librerias para el uso de datos, gráficos y el análisis estadístico.

```
import numpy as np  # Manejo de datos numéricos
import matplotlib.pyplot as plt  # Visualización de datos
from scipy.stats import ttest_rel  # Prueba estadística t pareada
from scipy.signal import welch  # Método de Welch para análisis espectral
from scipy.signal.windows import hann  # Ventana de Hanning para suavizar datos
```

#### 2. Configuración de parámetros

Se define la frecuencia de muestreo y el archivo que tiene la señal.

```
FS = 1000  # Frecuencia de muestreo en Hz
ARCHIVO_DATOS = "datos_señal.npy"  # Archivo con la señal EMG
```

#### 3.  Carga de datos

Se carga la señal desde el archivo `npy` y se almacena en un arreglo de NumPy.

```
datos_emg = np.load(ARCHIVO_DATOS)
```
#### 4. Visualización de la señal 

Se define una función para graficar la señal en funcion del tiempo.

```
def mostrar_senal(raw_signal, sampling_rate):
    t_axis = np.arange(len(raw_signal)) / sampling_rate
    plt.figure(figsize=(12, 5))
    plt.plot(t_axis, raw_signal, label="Señal EMG Bruta", color='purple', linewidth=0.6)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Voltaje (V)")
    plt.title("Señal EMG Adquirida (Original)")
    plt.legend()
    plt.grid(True)
    plt.show()
```
#### 5. Análisis espectral

Se define la función para calcular el espectro y evaluar los cambios de la frecuencia mediana.

Se aplica la ventana de Hanning para suavizar la señal y reducir efectos de discontinuidad
```
def evaluar_fatiga(emg_signal, sampling_rate):
    window_function = hann(256)
```
 Se calcula la densidad espectral de potencia con el método de Welch
 ```
    freqs, power_spectrum = welch(emg_signal, fs=sampling_rate, window=window_function, nperseg=256)
```
 Se divide la señal en dos segmentos: primera y segunda mitad
``` 
    primera_mitad = emg_signal[:len(emg_signal) // 2]
    segunda_mitad = emg_signal[len(emg_signal) // 2:]
```
 Se calcula el espectro de cada segmento para comparar cambios en el tiempo

```
    f_1, p_1 = welch(primera_mitad, fs=sampling_rate, window=window_function, nperseg=256)
    f_2, p_2 = welch(segunda_mitad, fs=sampling_rate, window=window_function, nperseg=256)
```
 Se realiza una prueba estadística t pareada para detectar diferencias significativas entre los espectros
```
tamano_min = min(len(p_1), len(p_2))
    t_stat, p_valor = ttest_rel(p_1[:tamano_min], p_2[:tamano_min])
```
Se imprimen los resultados de la frecuencia mediana y el análisis estadístico
```
print(f"Frecuencia mediana (inicio): {np.median(f_1):.2f} Hz")
    print(f"Frecuencia mediana (final): {np.median(f_2):.2f} Hz")
    print(f"Prueba de hipótesis (valor p): {p_valor:.4f}")
```
Se grafica la densidad espectral de potencia de toda la señal
```
    plt.figure(figsize=(12, 6))
    plt.fill_between(freqs, power_spectrum, color="orange", alpha=0.6, label="Densidad Espectral")
    plt.axvline(np.median(f_1), color="blue", linestyle="--", label="Mediana Inicial")
    plt.axvline(np.median(f_2), color="red", linestyle="--", label="Mediana Final")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Densidad Espectral de Potencia (V²/Hz)")
    plt.title("Análisis Espectral de la Señal EMG")
    plt.legend()
    plt.grid()
    plt.show()
```
 Se comparan los espectros de la primera y segunda mitad de la señal
```
    plt.figure(figsize=(12, 6))
    plt.plot(f_1, p_1, label="Espectro Inicial", color="blue")
    plt.plot(f_2, p_2, label="Espectro Final", color="red", linestyle="dashed")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Densidad Espectral de Potencia (V²/Hz)")
    plt.title("Comparación de Espectros EMG en el Tiempo")
    plt.legend()
    plt.grid()
    plt.show()
```
Este código nos permite analizar señales EMG mediante el método de Welch, evaluando cambios en la frecuencia mediana para determinar la presencia de fatiga muscular. Una disminución en la frecuencia mediana a lo largo del tiempo puede indicar un proceso de fatiga en el músculo analizado.

#### Señal Original
![image](https://github.com/user-attachments/assets/97aa9763-8032-4348-8919-39d6ddd94c93)

#### Resultado
![image](https://github.com/user-attachments/assets/1b18948d-6c9d-4c91-b6fe-717d69b923cc)

![image](https://github.com/user-attachments/assets/3872ede9-bd4b-485b-93a8-d730013e450e)

#### Análisis Espectral de la Señal EMG:

- Muestra el espectro de potencia con las medianas de frecuencia inicial y final marcadas.

- La frecuencia mediana inicial y final son ambas 250 Hz, lo que sugiere que no hay un desplazamiento significativo en la frecuencia dominante.

#### Comparación de Espectros Inicial y Final:

Se observa la evolución de la distribución espectral en el tiempo.

Un valor p de 0.0086 indica que hay una diferencia estadísticamente significativa en la distribución espectral entre el inicio y el final.

### Hipotesis

Después de ejecutar `ttest_rel(p_1[:tamano_min], p_2[:tamano_min])`, obtenemos:

- Valor de t: Indica cuán diferente es la potencia espectral entre los dos momentos.

- Valor p: Indica la probabilidad de obtener esos datos si la hipótesis nula fuera cierto


Después de ejecutar `ttest_rel(p_1[:tamano_min], p_2[:tamano_min])`, obtenemos:

- Valor de t: Indica cuán diferente es la potencia espectral entre los dos momentos.

- Valor p: Indica la probabilidad de obtener esos datos si la hipótesis nula fuera cier

Si p < 0.05, rechazamos 𝐻0 , lo que sugiere que la potencia espectral cambió significativamente (indicio de fatiga muscular).

Si p > 0.05, no hay suficiente evidencia para rechazar 𝐻0 , por lo que no podemos afirmar que haya fatiga

Este análisis nos ayuda a determinar si la señal  muestra signos de fatiga muscular con el tiempo. Si el valor p es bajo, podemos concluir que el espectro de potencia cambia significativamente, lo que podría indicar fatiga.
