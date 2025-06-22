import time
import os
import numpy as np
import gc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import sys
import pywt
from scipy.fft import fft
from itertools import product

def create_results_directory():
    """Crea el directorio de resultados si no existe."""
    results_dir = "resultados_tiempos_memoria"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def fourier(sequences, is_str=True):
    """Aplica transformada de Fourier a las secuencias."""
    templist = []
    for seq in sequences:
        if is_str:
            num_seq = [ord(char) for char in seq]
        else:
            num_seq = seq
        fft_seq = fft(num_seq)
        fft_seq = np.abs(fft_seq)
        templist.append(fft_seq[1:len(fft_seq)//2])
    return templist

def generate_kmers_dict(k, unique_chars=set('ACGNT')):
    """Genera diccionario de k-mers."""
    kmers = product(unique_chars, repeat=k)
    return {''.join(kmer): i for i, kmer in enumerate(kmers)}

def k_mers(sequencias, k=3, unique_chars=set('ACGNT')):
    """Codificación usando k-mers."""
    kmers_map = generate_kmers_dict(k, unique_chars)
    templist = []
    for seq in sequencias:
        temp = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        templist.append([kmers_map[i] for i in temp])
    return templist

def one_hot(sequences, max_len, unique_chars=set('ACGNT'), reshape=True):
    """Codificación one-hot."""
    mapping = {j: i for i, j in enumerate(unique_chars)}
    sequencias_procesadas = []
    
    for s in sequences:
        temp = np.zeros((max_len, len(unique_chars)))
        for c in zip(s, temp):
            c[1][mapping[c[0]]] = 1
        if reshape:
            sequencias_procesadas.append(temp.reshape(-1))
        else:
            sequencias_procesadas.append(temp)
    return sequencias_procesadas

def wavelet_transform(sequences, numeric=False, wavelet_name='db1', level=5):
    """Transformada wavelet."""
    templist = []
    for seq in sequences:
        if not numeric:
            num_seq = [ord(char) for char in seq]
        else:
            num_seq = seq
        coeffs = pywt.wavedec(num_seq, wavelet_name, level)
        templist.append(np.concatenate(coeffs))
    return templist

def kmers_fft(sequences, k=3, unique_chars=set('ACGNT')):
    """Combinación k-mers + FFT."""
    kmers_result = k_mers(sequences, k, unique_chars)
    return fourier(kmers_result, is_str=False)

def onehot_fft(sequences, max_len, unique_chars=set('ACGTN')):
    """Combinación one-hot + FFT."""
    onehot_result = one_hot(sequences, max_len, unique_chars, reshape=True)
    return fourier(onehot_result, is_str=False)

def kmers_wavelet(sequences, k=3, unique_chars=set('ACGNT'), wavelet_name='db1', level=5):
    """Combinación k-mers + wavelet."""
    kmers_result = k_mers(sequences, k, unique_chars)
    return wavelet_transform(kmers_result, numeric=True, wavelet_name=wavelet_name, level=level)

def onehot_wavelet(sequences, max_len, unique_chars=set('ACGNT'), wavelet_name='db1', level=5):
    """Combinación one-hot + wavelet."""
    onehot_result = one_hot(sequences, max_len, unique_chars, reshape=True)
    return wavelet_transform(onehot_result, numeric=True, wavelet_name=wavelet_name, level=level)

def generate_dna_sequence(length, nucleotides='ACGTN'):
    """Genera secuencia aleatoria de ADN."""
    return ''.join(random.choice(nucleotides) for _ in range(length))

def generate_sequences(n, length, nucleotides='ACGTN', seed=None):
    """Genera n secuencias de ADN."""
    if seed is not None:
        random.seed(seed)
    return [generate_dna_sequence(length, nucleotides) for _ in range(n)]

def measure_time(func, args=(), kwargs={}):
    """Mide tiempo de ejecución."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def get_deep_size(obj, seen=None):
    """Calcula tamaño real de objetos anidados."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_deep_size(item, seen) for item in obj)
    elif isinstance(obj, np.ndarray):
        size = obj.nbytes
    
    return size

def measure_direct_memory(result):
    """Mide directamente el tamaño en memoria del resultado."""
    if isinstance(result, list):
        tamano_lista = sys.getsizeof(result)
        if all(isinstance(item, np.ndarray) for item in result):
            tamano_elementos = sum(arr.nbytes for arr in result)
        else:
            tamano_elementos = sum(sys.getsizeof(elemento) for elemento in result)
            for elemento in result:
                if isinstance(elemento, (list, np.ndarray)):
                    if isinstance(elemento, list):
                        tamano_elementos += sum(sys.getsizeof(subelem) for subelem in elemento)
                    elif isinstance(elemento, np.ndarray):
                        tamano_elementos += elemento.nbytes - sys.getsizeof(elemento)
        
        tamano_total = tamano_lista + tamano_elementos
        return tamano_total / (1024 * 1024)
    elif isinstance(result, np.ndarray):
        return result.nbytes / (1024 * 1024)
    else:
        return sys.getsizeof(result) / (1024 * 1024)

def benchmark_function(func, args=(), kwargs={}, n_runs=5):
    """Mide tiempo y memoria de una función."""
    time_results = []
    result_sizes = []
    result = None
    
    for i in range(n_runs):
        gc.collect()
        result, time_used = measure_time(func, args, kwargs)
        time_results.append(time_used)
        result_size = measure_direct_memory(result)
        result_sizes.append(result_size)
        gc.collect()
    
    return {
        'function': func.__name__,
        'avg_time': np.mean(time_results),
        'std_time': np.std(time_results),
        'avg_result_size': np.mean(result_sizes),
        'std_result_size': np.std(result_sizes),
        'all_times': time_results,
        'all_result_sizes': result_sizes
    }

def run_benchmarks(sequences, n_runs=5):
    """Ejecuta benchmarks para todas las funciones."""
    max_length = max(len(seq) for seq in sequences)
    
    functions = [
        fourier, k_mers, one_hot, wavelet_transform,
        kmers_fft, onehot_fft, kmers_wavelet, onehot_wavelet
    ]
    
    functions_args = [
        {'is_str': True},
        {'k': 3, 'unique_chars': set('ACGTN')},
        {'max_len': max_length, 'unique_chars': set('ACGTN'), 'reshape': True},
        {'numeric': False, 'wavelet_name': 'db1', 'level': 5},
        {'k': 3, 'unique_chars': set('ACGTN')},
        {'max_len': max_length, 'unique_chars': set('ACGTN')},
        {'k': 3, 'unique_chars': set('ACGTN'), 'wavelet_name': 'db1', 'level': 5},
        {'max_len': max_length, 'unique_chars': set('ACGTN'), 'wavelet_name': 'db1', 'level': 5}
    ]
    
    results = []
    for func, kwargs in zip(functions, functions_args):
        print(f"Benchmarking {func.__name__}...")
        result = benchmark_function(func, args=(sequences,), kwargs=kwargs, n_runs=n_runs)
        results.append(result)
    
    return pd.DataFrame(results)

def save_plot(fig, filename, results_dir, scale=5):
    """Guarda un gráfico en el directorio de resultados."""
    filepath = os.path.join(results_dir, f"{filename}.png")
    fig.write_image(filepath, scale=scale)
    print(f"Gráfico guardado: {filepath}")

def plot_benchmark_results(df_results, results_dir, title="Comparación de Rendimiento"):
    """Genera y guarda gráfico de rendimiento."""
    df_sorted = df_results.sort_values('avg_time')
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Tiempo de Ejecución Promedio", "Tamaño del Resultado en Memoria"),
                        vertical_spacing=0.15)
    
    fig.add_trace(
        go.Bar(
            x=df_sorted['function'],
            y=df_sorted['avg_time'],
            error_y=dict(type='data', array=df_sorted['std_time']),
            name='Tiempo de Ejecución (s)',
            hovertemplate='<b>%{x}</b><br>Tiempo: %{y:.3f} s ± %{error_y.array:.3f}<br>',
            marker_color='rgb(55, 83, 109)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=df_sorted['function'],
            y=df_sorted['avg_result_size'],
            error_y=dict(type='data', array=df_sorted['std_result_size']),
            name='Tamaño del Resultado (MB)',
            hovertemplate='<b>%{x}</b><br>Memoria: %{y:.2f} MB ± %{error_y.array:.2f}<br>',
            marker_color='rgb(26, 118, 255)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title_text=title,
        showlegend=False,
        height=700,
        width=900,
        title={'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Tiempo (segundos)", row=1, col=1)
    fig.update_yaxes(title_text="Memoria (MB)", row=2, col=1)
    
    save_plot(fig, "01_performance_comparison", results_dir)
    return fig

def plot_scatter_efficiency(df_results, results_dir):
    """Genera y guarda gráfico de dispersión eficiencia."""
    fig = go.Figure()
    
    df_results['compression_ratio'] = df_results['avg_result_size'] / df_results['avg_result_size'].min()
    
    fig.add_trace(go.Scatter(
        x=df_results['avg_time'],
        y=df_results['avg_result_size'],
        mode='markers+text',
        marker=dict(
            size=15,
            color=df_results['compression_ratio'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Ratio de Tamaño<br>(relativo al mínimo)'),
            line=dict(width=1, color='black')
        ),
        text=df_results['function'],
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>Tiempo: %{x:.3f}s<br>Memoria: %{y:.2f}MB<extra></extra>'
    ))
    
    median_time = df_results['avg_time'].median()
    median_memory = df_results['avg_result_size'].median()
    
    fig.add_shape(type="line", x0=median_time, y0=0, x1=median_time, 
                  y1=df_results['avg_result_size'].max() * 1.1,
                  line=dict(color="gray", width=1, dash="dash"))
    
    fig.add_shape(type="line", x0=0, y0=median_memory, 
                  x1=df_results['avg_time'].max() * 1.1, y1=median_memory,
                  line=dict(color="gray", width=1, dash="dash"))
    
    fig.update_layout(
        title='Relación Tiempo-Memoria por Método de Codificación',
        xaxis_title='Tiempo (segundos)',
        yaxis_title='Tamaño del Resultado (MB)',
        height=700,
        width=900
    )
    
    save_plot(fig, "02_scatter_efficiency", results_dir)
    return fig

def plot_combined_heatmap(df_results, results_dir):
    """Genera y guarda mapa de calor combinado."""
    df = df_results.copy()
    
    # Normalizar métricas
    min_time = df['avg_time'].min()
    max_time = df['avg_time'].max()
    if max_time > min_time:
        df['norm_time'] = 1 - ((df['avg_time'] - min_time) / (max_time - min_time))
    else:
        df['norm_time'] = 1.0
        
    min_mem = df['avg_result_size'].min()
    max_mem = df['avg_result_size'].max()
    if max_mem > min_mem:
        df['norm_memory'] = 1 - ((df['avg_result_size'] - min_mem) / (max_mem - min_mem))
    else:
        df['norm_memory'] = 1.0
    
    df['time_stability'] = 1 - (df['std_time'] / df['avg_time']).fillna(0)
    df['mem_stability'] = 1 - (df['std_result_size'] / df['avg_result_size']).fillna(0)
    df['time_stability'] = df['time_stability'].clip(0, 1)
    df['mem_stability'] = df['mem_stability'].clip(0, 1)
    df['efficiency'] = (df['norm_time'] + df['norm_memory']) / 2
    
    # Crear datos para heatmap
    heatmap_data = []
    for _, row in df.iterrows():
        heatmap_data.extend([
            [row['function'], 'Tiempo', row['norm_time']],
            [row['function'], 'Memoria', row['norm_memory']],
            [row['function'], 'Estabilidad Tiempo', row['time_stability']],
            [row['function'], 'Estabilidad Memoria', row['mem_stability']],
            [row['function'], 'Eficiencia Global', row['efficiency']]
        ])
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=['Método', 'Métrica', 'Valor'])
    pivot_df = heatmap_df.pivot(index='Método', columns='Métrica', values='Valor')
    pivot_df = pivot_df.sort_values('Eficiencia Global', ascending=False)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        text=np.round(pivot_df.values, 2),
        texttemplate="%{text:.2f}",
        textfont={"size":10},
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Mapa de Calor: Rendimiento Relativo de Métodos de Codificación',
        height=700,
        width=1000,
        xaxis_title='Métrica',
        yaxis_title='Método de Codificación'
    )
    
    save_plot(fig, "03_heatmap_performance", results_dir)
    return fig

def run_complete_benchmark(sequences=None, n=500, length=100, max_sequences=None, 
                          max_length=None, n_runs=10, seed=None):
    """Ejecuta benchmark completo y guarda resultados."""
    
    # Crear directorio de resultados
    results_dir = create_results_directory()
    
    # Generar o procesar secuencias
    if sequences is None:
        print(f"Generando {n} secuencias de longitud {length}...")
        sequences = generate_sequences(n, length, 'ACGTN', seed)
    
    if max_sequences is not None and max_sequences < len(sequences):
        sequences = sequences[:max_sequences]
    
    if max_length is not None:
        sequences = [seq[:max_length] for seq in sequences]
    
    print(f"Ejecutando benchmark con {len(sequences)} secuencias...")
    
    # Ejecutar benchmarks
    results = run_benchmarks(sequences, n_runs=n_runs)
    
    # Generar y guardar gráficos
    print("Generando gráficos...")
    title = f"Comparación de Rendimiento (N={len(sequences)}, L={len(sequences[0])})"
    
    plot_benchmark_results(results, results_dir, title)
    plot_scatter_efficiency(results, results_dir)
    plot_combined_heatmap(results, results_dir)
    
    # Guardar resultados en CSV
    results_file = os.path.join(results_dir, "benchmark_results.csv")
    results.to_csv(results_file, index=False)
    print(f"Resultados guardados en: {results_file}")
    
    # Mostrar resumen
    print("\nResumen de resultados:")
    display_results = results[['function', 'avg_time', 'std_time', 'avg_result_size', 'std_result_size']]
    display_results = display_results.sort_values('avg_time')
    print(display_results.to_string(index=False))
    
    return results

def main():
    """Función principal del script."""
    try:
        # Intentar cargar datos desde archivo CSV
        if os.path.exists('secuencias_bench.txt'):
            print("Cargando secuencias desde secuencias_bench.txt...")
            df = pd.read_csv('secuencias_bench.txt')
            sequences = df['secuencia'].tolist()
            print(f"Cargadas {len(sequences)} secuencias de longitud {len(sequences[0])}")
        else:
            print("Archivo secuencias_bench.txt no encontrado. Generando secuencias...")
            sequences = None
        
        # Ejecutar benchmark completo
        results = run_complete_benchmark(
            sequences=sequences,
            n=1000,  # Solo se usa si sequences es None
            length=100,  # Solo se usa si sequences es None
            n_runs=40,
            seed=42
        )
        
        print("\n¡Benchmark completado! Revisa la carpeta 'resultados_tiempos_memoria' para ver los gráficos.")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
