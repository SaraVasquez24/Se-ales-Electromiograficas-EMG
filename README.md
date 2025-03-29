# Se-ales-Electromiograficas-EMG

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

