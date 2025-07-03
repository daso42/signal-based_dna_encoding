import time
import os
import numpy as np
import pandas as pd
import sys
import gc
from itertools import product
import pywt
from scipy.fft import fft
import plotly.graph_objs as go
from plotly.subplots import make_subplots

pd.options.mode.chained_assignment = None

def create_results_directory():
    """Creates results directory if it doesn't exist."""
    results_dir = "dna_encoding_benchmark_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def fourier(sequences, is_str=True):
    """Applies Fourier transform to sequences."""
    if is_str:
        templist = []
        for seq in sequences:
            num_seq = [ord(char) for char in seq]
            fft_seq = fft(num_seq)
            fft_seq = np.abs(fft_seq)
            templist.append(fft_seq[1:len(fft_seq)//2])
        return templist
    else:
        templist = []
        for seq in sequences:
            fft_seq = fft(seq)
            fft_seq = np.abs(fft_seq)
            templist.append(fft_seq[1:len(fft_seq)//2])
        return templist

def generate_kmers_dict(k, unique_chars=set('ACGNT')):
    """Generates k-mers dictionary."""
    kmers = product(unique_chars, repeat=k)
    return {''.join(kmer): i for i, kmer in enumerate(kmers)}

def k_mers(sequencias, k=3, unique_chars=set('ACGNT')):
    """K-mers encoding."""
    kmers_map = generate_kmers_dict(k, unique_chars)
    templist = []
    for seq in sequencias:
        temp = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        templist.append([kmers_map[i] for i in temp])
    return templist

def one_hot(sequences, max_len, unique_chars=set('ACGNT'), reshape=True):
    """One-hot encoding."""
    mapping = {j: i for i, j in enumerate(unique_chars)}
    sequencias_procesadas = []
    if reshape == True:
        for s in sequences:
            temp = np.zeros((max_len, len(unique_chars)))
            for c in zip(s, temp):
                c[1][mapping[c[0]]] = 1
            sequencias_procesadas.append(temp.reshape(-1))
        return sequencias_procesadas
    elif reshape == False:
        for s in sequences:
            temp = np.zeros((max_len, len(unique_chars)))
            for c in zip(s, temp):
                c[1][mapping[c[0]]] = 1
            sequencias_procesadas.append(temp)
        return sequencias_procesadas

def wavelet(sequences, numeric=False, wavelet='db1', level=5):
    """Wavelet transform."""
    templist = []
    if numeric == False:
        for seq in sequences:
            num_seq = [ord(char) for char in seq]
            coeffs = pywt.wavedec(num_seq, wavelet, level)
            templist.append(np.concatenate(coeffs))
        return templist
    elif numeric == True:
        for seq in sequences:
            coeffs = pywt.wavedec(seq, wavelet, level)
            templist.append(np.concatenate(coeffs))
        return templist

def pad_sequences(sequences, maxlen):
    """Pads sequences to equal length."""
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            seq += 'N' * (maxlen - len(seq))
        else:
            seq = seq[:maxlen]
        padded_sequences.append(seq)
    return padded_sequences

def measure_time(func, *args, **kwargs):
    """Measures execution time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def measure_memory(result):
    """Measures memory usage of result."""
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
        return tamano_total / (1024 * 1024)  # MB
    elif isinstance(result, np.ndarray):
        return result.nbytes / (1024 * 1024)  # MB
    else:
        return sys.getsizeof(result) / (1024 * 1024)  # MB

def benchmark_encoding(func, sequences, func_name, n_runs=5, **kwargs):
    """Executes benchmark for an encoding function."""
    print(f"Benchmarking {func_name}...")
    
    time_results = []
    memory_results = []
    
    for i in range(n_runs):
        gc.collect()
        result, exec_time = measure_time(func, sequences, **kwargs)
        memory_size = measure_memory(result)
        
        time_results.append(exec_time)
        memory_results.append(memory_size)
        gc.collect()
    
    return {
        'encoding': func_name,
        'avg_time': np.mean(time_results),
        'std_time': np.std(time_results),
        'avg_memory': np.mean(memory_results),
        'std_memory': np.std(memory_results),
        'all_times': time_results,
        'all_memories': memory_results
    }

def run_combined_benchmarks(df, n_runs=5):
    """Runs benchmarks for both AS and PS sequences."""
    results = []
    
    # AS sequences
    print("\n=== BENCHMARKING AS SEQUENCES ===")
    as_sequences = df['as'].values
    as_max_len = len(as_sequences[0])
    
    # AS - One Hot
    result = benchmark_encoding(one_hot, as_sequences, 'AS-One Hot', 
                              n_runs, max_len=as_max_len)
    results.append(result)
    
    # AS - K-mers
    result = benchmark_encoding(k_mers, as_sequences, 'AS-K-mers', n_runs)
    results.append(result)
    
    # AS - FFT
    result = benchmark_encoding(fourier, as_sequences, 'AS-FFT', n_runs)
    results.append(result)
    
    # AS - Wavelet
    result = benchmark_encoding(wavelet, as_sequences, 'AS-Wavelet', n_runs)
    results.append(result)
    
    # AS combinations
    as_kmers = k_mers(as_sequences)
    as_onehot = one_hot(as_sequences, as_max_len)
    
    result = benchmark_encoding(fourier, as_kmers, 'AS-K-mers + FFT', 
                              n_runs, is_str=False)
    results.append(result)
    
    result = benchmark_encoding(fourier, as_onehot, 'AS-One Hot + FFT', 
                              n_runs, is_str=False)
    results.append(result)
    
    result = benchmark_encoding(wavelet, as_kmers, 'AS-K-mers + Wavelet', 
                              n_runs, numeric=True)
    results.append(result)
    
    result = benchmark_encoding(wavelet, as_onehot, 'AS-One Hot + Wavelet', 
                              n_runs, numeric=True)
    results.append(result)
    
    # PS sequences
    print("\n=== BENCHMARKING PS SEQUENCES ===")
    ps_sequences = df['ps'].values
    ps_max_len = len(ps_sequences[0])
    
    # PS - One Hot
    result = benchmark_encoding(one_hot, ps_sequences, 'PS-One Hot', 
                              n_runs, max_len=ps_max_len)
    results.append(result)
    
    # PS - K-mers
    result = benchmark_encoding(k_mers, ps_sequences, 'PS-K-mers', n_runs)
    results.append(result)
    
    # PS - FFT
    result = benchmark_encoding(fourier, ps_sequences, 'PS-FFT', n_runs)
    results.append(result)
    
    # PS - Wavelet
    result = benchmark_encoding(wavelet, ps_sequences, 'PS-Wavelet', n_runs)
    results.append(result)
    
    # PS combinations
    ps_kmers = k_mers(ps_sequences)
    ps_onehot = one_hot(ps_sequences, ps_max_len)
    
    result = benchmark_encoding(fourier, ps_kmers, 'PS-K-mers + FFT', 
                              n_runs, is_str=False)
    results.append(result)
    
    result = benchmark_encoding(fourier, ps_onehot, 'PS-One Hot + FFT', 
                              n_runs, is_str=False)
    results.append(result)
    
    result = benchmark_encoding(wavelet, ps_kmers, 'PS-K-mers + Wavelet', 
                              n_runs, numeric=True)
    results.append(result)
    
    result = benchmark_encoding(wavelet, ps_onehot, 'PS-One Hot + Wavelet', 
                              n_runs, numeric=True)
    results.append(result)
    
    return pd.DataFrame(results)

def save_plot(fig, filename, results_dir):
    """Saves plot to results directory."""
    filepath = os.path.join(results_dir, f"{filename}.png")
    fig.write_image(filepath, scale=3)
    print(f"Plot saved: {filepath}")

def plot_combined_benchmark(df_results, results_dir):
    """Generates combined benchmark comparison plot."""
    df_sorted = df_results.sort_values('avg_time')
    
    # Separate AS and PS for color coding
    df_sorted['sequence_type'] = df_sorted['encoding'].str.split('-').str[0]
    colors = ['rgb(55, 83, 109)' if x == 'AS' else 'rgb(26, 118, 255)' 
              for x in df_sorted['sequence_type']]
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Execution Time", "Memory Usage"),
                        vertical_spacing=0.12)
    
    # Time plot
    fig.add_trace(
        go.Bar(
            x=df_sorted['encoding'],
            y=df_sorted['avg_time'],
            error_y=dict(type='data', array=df_sorted['std_time']),
            name='Execution Time (s)',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Time: %{y:.4f} s ± %{error_y.array:.4f}<br>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Memory plot
    fig.add_trace(
        go.Bar(
            x=df_sorted['encoding'],
            y=df_sorted['avg_memory'],
            error_y=dict(type='data', array=df_sorted['std_memory']),
            name='Memory Usage (MB)',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Memory: %{y:.2f} MB ± %{error_y.array:.2f}<br>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title_text='DNA Sequence Encoding Benchmark: AS vs PS Sequences',
        height=1000,
        width=1400
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
    
    # Add legend manually
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                            marker=dict(size=10, color='rgb(55, 83, 109)'),
                            legendgroup='AS', showlegend=True, name='AS Sequences'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                            marker=dict(size=10, color='rgb(26, 118, 255)'),
                            legendgroup='PS', showlegend=True, name='PS Sequences'))
    
    save_plot(fig, "combined_benchmark", results_dir)
    return fig

def plot_efficiency_comparison(df_results, results_dir):
    """Generates efficiency comparison scatter plot."""
    # Separate AS and PS
    as_results = df_results[df_results['encoding'].str.startswith('AS-')]
    ps_results = df_results[df_results['encoding'].str.startswith('PS-')]
    
    fig = go.Figure()
    
    # AS sequences
    fig.add_trace(go.Scatter(
        x=as_results['avg_time'],
        y=as_results['avg_memory'],
        mode='markers+text',
        marker=dict(
            size=12,
            color='rgb(55, 83, 109)',
            line=dict(width=2, color='white')
        ),
        text=[enc.replace('AS-', '') for enc in as_results['encoding']],
        textposition="top center",
        name='AS Sequences',
        hovertemplate='<b>%{text}</b><br>Time: %{x:.4f}s<br>Memory: %{y:.2f}MB<extra></extra>'
    ))
    
    # PS sequences
    fig.add_trace(go.Scatter(
        x=ps_results['avg_time'],
        y=ps_results['avg_memory'],
        mode='markers+text',
        marker=dict(
            size=12,
            color='rgb(26, 118, 255)',
            line=dict(width=2, color='white')
        ),
        text=[enc.replace('PS-', '') for enc in ps_results['encoding']],
        textposition="bottom center",
        name='PS Sequences',
        hovertemplate='<b>%{text}</b><br>Time: %{x:.4f}s<br>Memory: %{y:.2f}MB<extra></extra>'
    ))
    
    fig.update_layout(
        title='Encoding Efficiency: Time vs Memory Usage',
        xaxis_title='Execution Time (seconds)',
        yaxis_title='Memory Usage (MB)',
        height=700,
        width=1200,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    save_plot(fig, "efficiency_comparison", results_dir)
    return fig

def plot_encoding_heatmap(df_results, results_dir):
    """Generates performance heatmap."""
    # Normalize metrics
    df = df_results.copy()
    df['norm_time'] = 1 - ((df['avg_time'] - df['avg_time'].min()) / 
                          (df['avg_time'].max() - df['avg_time'].min()))
    df['norm_memory'] = 1 - ((df['avg_memory'] - df['avg_memory'].min()) / 
                            (df['avg_memory'].max() - df['avg_memory'].min()))
    df['efficiency'] = (df['norm_time'] + df['norm_memory']) / 2
    
    # Create heatmap data
    metrics = ['Time Efficiency', 'Memory Efficiency', 'Overall Efficiency']
    heatmap_data = []
    
    for _, row in df.iterrows():
        heatmap_data.append([row['encoding'], 'Time Efficiency', row['norm_time']])
        heatmap_data.append([row['encoding'], 'Memory Efficiency', row['norm_memory']])
        heatmap_data.append([row['encoding'], 'Overall Efficiency', row['efficiency']])
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=['Encoding', 'Metric', 'Value'])
    pivot_df = heatmap_df.pivot(index='Encoding', columns='Metric', values='Value')
    pivot_df = pivot_df.sort_values('Overall Efficiency', ascending=False)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        text=np.round(pivot_df.values, 3),
        texttemplate="%{text:.3f}",
        textfont={"size": 10},
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Encoding Performance Heatmap (Higher is Better)',
        height=800,
        width=800,
        xaxis_title='Performance Metric',
        yaxis_title='Encoding Method'
    )
    
    save_plot(fig, "performance_heatmap", results_dir)
    return fig

def load_and_process_data():
    """Loads and processes the dataset."""
    print("Loading dataset...")
    df = pd.read_csv('dmfne.csv')  # File in same folder
    df = df[['genus', 'se', 'sequence', 'gc_content']]
    df = df.rename(columns={'se': 'as'})
    
    print("Processing sequences...")
    # Sequence padding
    maxlen = max([len(i) for i in df['sequence']])
    df['ps'] = pad_sequences(df['sequence'], maxlen)
    
    # Add length information
    df['len_sequence'] = [len(i) for i in df['sequence']]
    df['len_ps'] = [len(i) for i in df['ps']]
    df['len_as'] = [len(i) for i in df['as']]
    
    # Class mapping
    map_genus = {j: i for i, j in enumerate(df['genus'].unique())}
    df['clases_modelos'] = df['genus'].map(map_genus)
    
    print(f"Dataset loaded: {len(df)} sequences")
    print(f"Max original sequence length: {max(df['len_sequence'])}")
    print(f"Padded sequence length: {df['len_ps'][0]}")
    print(f"AS sequence length: {df['len_as'][0]}")
    
    return df

def main():
    """Main function."""
    try:
        # Create results directory
        results_dir = create_results_directory()
        
        # Load and process data
        df = load_and_process_data()
        
        # Run combined benchmarks
        print("\n=== RUNNING COMBINED BENCHMARK ===")
        results = run_combined_benchmarks(df, n_runs=40)
        
        # Generate plots
        print("\nGenerating plots...")
        plot_combined_benchmark(results, results_dir)
        plot_efficiency_comparison(results, results_dir)
        plot_encoding_heatmap(results, results_dir)
        
        # Save results
        results.to_csv(os.path.join(results_dir, 'combined_benchmark_results.csv'), index=False)
        
        # Show summary
        print("\n=== BENCHMARK RESULTS SUMMARY ===")
        display_results = results[['encoding', 'avg_time', 'std_time', 'avg_memory', 'std_memory']]
        display_results = display_results.sort_values('avg_time')
        print(display_results.to_string(index=False))
        
        # Best performers
        print("\n=== FASTEST ENCODINGS ===")
        fastest = results.nsmallest(5, 'avg_time')[['encoding', 'avg_time']]
        print(fastest.to_string(index=False))
        
        print("\n=== MOST MEMORY EFFICIENT ===")
        most_efficient = results.nsmallest(5, 'avg_memory')[['encoding', 'avg_memory']]
        print(most_efficient.to_string(index=False))
        
        print(f"\nBenchmark completed! Check '{results_dir}' folder for detailed results.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())