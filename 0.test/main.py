import time
import os
import numpy as np
import gc
import pandas as pd
import random
import sys
import pywt
from scipy.fft import fft
from itertools import product

# Disable pandas warnings
pd.options.mode.chained_assignment = None

# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================

def fourier(sequences, is_str=True):
    """Apply Fourier transform to sequences."""
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
    """Generate k-mers dictionary."""
    kmers = product(unique_chars, repeat=k)
    kmer_dict = {''.join(kmer): i for i, kmer in enumerate(kmers)}
    return kmer_dict

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

# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def create_results_directory():
    """Create results directory if it doesn't exist."""
    results_dir = "benchmark_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def measure_time(func, args=(), kwargs={}):
    """Measure execution time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def measure_direct_memory(result):
    """Measure direct memory size of result."""
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

def benchmark_encoding_method(sequences, encoding_func, kwargs, method_name, n_runs=40, dataset_info=None):
    """Benchmark a single encoding method."""
    print(f"    Benchmarking {method_name}...")
    
    time_results = []
    memory_results = []
    result = None
    
    # Record start time for total execution tracking
    total_start_time = time.time()
    
    for i in range(n_runs):
        gc.collect()
        result, time_used = measure_time(encoding_func, args=(sequences,), kwargs=kwargs)
        time_results.append(time_used)
        memory_used = measure_direct_memory(result)
        memory_results.append(memory_used)
        gc.collect()
    
    total_execution_time = time.time() - total_start_time
    
    # Calculate encoded length and compression ratio
    if result and len(result) > 0:
        if isinstance(result[0], np.ndarray):
            encoded_length = result[0].shape[0] if result[0].ndim == 1 else np.prod(result[0].shape)
        elif isinstance(result[0], list):
            encoded_length = len(result[0])
        else:
            encoded_length = 1
    else:
        encoded_length = 0
    
    original_length = len(sequences[0])
    compression_ratio = encoded_length / original_length if original_length > 0 else 0
    
    # Calculate efficiency metrics
    time_per_sequence = np.mean(time_results) / len(sequences) * 1000  # ms per sequence
    memory_per_sequence = np.mean(memory_results) / len(sequences)  # MB per sequence
    
    # Performance stability metrics
    time_cv = np.std(time_results) / np.mean(time_results) if np.mean(time_results) > 0 else 0  # Coefficient of variation
    memory_cv = np.std(memory_results) / np.mean(memory_results) if np.mean(memory_results) > 0 else 0
    
    # Determine sequence type and encoding category
    sequence_type = 'AS' if method_name.startswith('AS-') else 'PS'
    encoding_category = 'Basic' if '+' not in method_name else 'Hybrid'
    base_method = method_name.split('-')[1].split(' +')[0]  # Extract base method (One Hot, K-mers, etc.)
    
    result_dict = {
        # Basic identification
        'encoding_method': method_name,
        'sequence_type': sequence_type,
        'encoding_category': encoding_category,
        'base_method': base_method,
        
        # Execution parameters
        'num_sequences': len(sequences),
        'n_runs': n_runs,
        'total_execution_time': total_execution_time,
        
        # Sequence characteristics
        'sequence_length_original': original_length,
        'sequence_length_encoded': encoded_length,
        'compression_ratio': compression_ratio,
        
        # Time metrics
        'avg_time': np.mean(time_results),
        'std_time': np.std(time_results),
        'min_time': np.min(time_results),
        'max_time': np.max(time_results),
        'median_time': np.median(time_results),
        'time_cv': time_cv,
        'time_per_sequence_ms': time_per_sequence,
        
        # Memory metrics
        'avg_memory_mb': np.mean(memory_results),
        'std_memory_mb': np.std(memory_results),
        'min_memory_mb': np.min(memory_results),
        'max_memory_mb': np.max(memory_results),
        'median_memory_mb': np.median(memory_results),
        'memory_cv': memory_cv,
        'memory_per_sequence_mb': memory_per_sequence,
        
        # Performance efficiency
        'sequences_per_second': len(sequences) / np.mean(time_results),
        'mb_per_second': np.mean(memory_results) / np.mean(time_results),
        
        # Raw data for detailed analysis
        'all_times': time_results,
        'all_memories': memory_results
    }
    
    # Add dataset information if provided
    if dataset_info:
        result_dict.update(dataset_info)
    
    return result_dict

def run_benchmark_for_sample_size(df, sample_size, n_runs=40):
    """Run benchmark for a specific sample size."""
    print(f"\nProcessing sample size: {sample_size}")
    
    # Sample the dataframe
    if sample_size >= len(df):
        df_subset = df.copy()
        print(f"Using complete dataset ({len(df)} sequences)")
    else:
        df_subset = df.sample(n=sample_size, random_state=42).copy()
        print(f"Sampled {sample_size} sequences from dataset")
    
    # Extract sequences
    as_sequences = df_subset['aligned_sequence'].values
    ps_sequences = df_subset['padded_sequences'].values
    
    # Calculate max lengths for one-hot encoding
    max_len_as = max([len(seq) for seq in as_sequences])
    max_len_ps = max([len(seq) for seq in ps_sequences])
    
    print(f"  AS sequence length: {len(as_sequences[0])} bp")
    print(f"  PS sequence length: {len(ps_sequences[0])} bp")
    print(f"  Processing {len(encodings)} encoding methods with {n_runs} runs each...")
    
    # Create dataset info dictionary for this sample size
    dataset_info = {
        'sample_size': sample_size,
        'total_dataset_size': len(df),
        'as_max_length': max_len_as,
        'ps_max_length': max_len_ps,
        'sampling_ratio': sample_size / len(df),
        'benchmark_timestamp': pd.Timestamp.now().isoformat(),
        'random_seed': 42
    }
    
    # Define encoding configurations
    encoding_configs = [
        # AS encodings
        (as_sequences, one_hot, {'max_len': max_len_as, 'unique_chars': set('ACGNT'), 'reshape': True}, 'AS-One Hot'),
        (as_sequences, k_mers, {'k': 3, 'unique_chars': set('ACGNT')}, 'AS-K-mers'),
        (as_sequences, fourier, {'is_str': True}, 'AS-FFT'),
        (as_sequences, wavelet, {'numeric': False, 'wavelet': 'db1', 'level': 5}, 'AS-Wavelet'),
        
        # PS encodings
        (ps_sequences, one_hot, {'max_len': max_len_ps, 'unique_chars': set('ACGNT'), 'reshape': True}, 'PS-One Hot'),
        (ps_sequences, k_mers, {'k': 3, 'unique_chars': set('ACGNT')}, 'PS-K-mers'),
        (ps_sequences, fourier, {'is_str': True}, 'PS-FFT'),
        (ps_sequences, wavelet, {'numeric': False, 'wavelet': 'db1', 'level': 5}, 'PS-Wavelet'),
    ]
    
    results = []
    
    # Benchmark basic encodings first
    for sequences, func, kwargs, name in encoding_configs:
        result = benchmark_encoding_method(sequences, func, kwargs, name, n_runs, dataset_info)
        results.append(result)
    
    # Now benchmark hybrid encodings that require pre-computed results
    print("    Computing hybrid encodings...")
    
    # Compute intermediate results for hybrid encodings
    as_kmers = k_mers(as_sequences, k=3, unique_chars=set('ACGNT'))
    as_onehot = one_hot(as_sequences, max_len_as, unique_chars=set('ACGNT'), reshape=True)
    ps_kmers = k_mers(ps_sequences, k=3, unique_chars=set('ACGNT'))
    ps_onehot = one_hot(ps_sequences, max_len_ps, unique_chars=set('ACGNT'), reshape=True)
    
    # Hybrid encoding configurations
    hybrid_configs = [
        # AS hybrid encodings
        (as_kmers, fourier, {'is_str': False}, 'AS-K-mers + FFT'),
        (as_onehot, fourier, {'is_str': False}, 'AS-One Hot + FFT'),
        (as_kmers, wavelet, {'numeric': True, 'wavelet': 'db1', 'level': 5}, 'AS-K-mers + Wavelet'),
        (as_onehot, wavelet, {'numeric': True, 'wavelet': 'db1', 'level': 5}, 'AS-One Hot + Wavelet'),
        
        # PS hybrid encodings
        (ps_kmers, fourier, {'is_str': False}, 'PS-K-mers + FFT'),
        (ps_onehot, fourier, {'is_str': False}, 'PS-One Hot + FFT'),
        (ps_kmers, wavelet, {'numeric': True, 'wavelet': 'db1', 'level': 5}, 'PS-K-mers + Wavelet'),
        (ps_onehot, wavelet, {'numeric': True, 'wavelet': 'db1', 'level': 5}, 'PS-One Hot + Wavelet'),
    ]
    
    # Benchmark hybrid encodings
    for sequences, func, kwargs, name in hybrid_configs:
        result = benchmark_encoding_method(sequences, func, kwargs, name, n_runs, dataset_info)
        results.append(result)
    
    return pd.DataFrame(results)

def run_complete_benchmark():
    """Run complete benchmark with multiple sample sizes."""
    print("Starting Sequence Encoding Benchmark System")
    print("=" * 60)
    
    # Load the dataset
    print("Loading dataset from csvtest.csv...")
    try:
        df = pd.read_csv('csvtest.csv')
        print(f"Dataset loaded successfully: {len(df)} sequences")
        print(f"Columns: {list(df.columns)}")
        
        # Verify required columns
        required_cols = ['aligned_sequence', 'padded_sequences']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Show sequence length info
        as_lengths = [len(seq) for seq in df['aligned_sequence']]
        ps_lengths = [len(seq) for seq in df['padded_sequences']]
        
        print(f"AS (Aligned) sequences: min={min(as_lengths)}, max={max(as_lengths)}, avg={np.mean(as_lengths):.1f}")
        print(f"PS (Padded) sequences: min={min(ps_lengths)}, max={max(ps_lengths)}, avg={np.mean(ps_lengths):.1f}")
            
    except FileNotFoundError:
        print("ERROR: csvtest.csv not found!")
        print("Please ensure the file exists and contains the required columns:")
        print("  - aligned_sequence, padded_sequences")
        return None
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None
    
    # Create results directory
    results_dir = create_results_directory()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define sample sizes as requested
    sample_sizes = [5000, 10000, 15000, 20000, 25000]
    
    # Filter sample sizes based on dataset size
    max_size = len(df)
    sample_sizes = [size for size in sample_sizes if size <= max_size]
    if max_size not in sample_sizes:
        sample_sizes.append(max_size)  # Add complete dataset
    sample_sizes = sorted(list(set(sample_sizes)))  # Remove duplicates and sort
    
    print(f"Will test with sample sizes: {sample_sizes}")
    
    all_results = []
    
    # Run benchmarks for each sample size
    for sample_size in sample_sizes:
        try:
            results_df = run_benchmark_for_sample_size(df, sample_size, n_runs=40)
            all_results.append(results_df)
        except Exception as e:
            print(f"ERROR processing sample size {sample_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("ERROR: No results generated!")
        return None
    
    # Combine all results
    print("\nCombining results...")
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save results to CSV
    results_file = os.path.join(results_dir, "encoding_benchmark_results.csv")
    final_results.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Group by encoding method and show average performance
    summary = final_results.groupby(['encoding_method']).agg({
        'avg_time': 'mean',
        'avg_memory_mb': 'mean',
        'num_sequences': 'max'
    }).round(4)
    
    print("\nAverage Performance by Encoding Method:")
    print(summary.to_string())
    
    # Find best performers overall
    best_time = final_results.loc[final_results['avg_time'].idxmin()]
    best_memory = final_results.loc[final_results['avg_memory_mb'].idxmin()]
    
    print(f"\nOverall Best Performance:")
    print(f"  Fastest: {best_time['encoding_method']} ({best_time['avg_time']:.4f}s, {best_time['num_sequences']} sequences)")
    print(f"  Most Memory Efficient: {best_memory['encoding_method']} ({best_memory['avg_memory_mb']:.2f}MB, {best_memory['num_sequences']} sequences)")
    
    # Show AS vs PS comparison
    as_results = final_results[final_results['encoding_method'].str.startswith('AS-')]
    ps_results = final_results[final_results['encoding_method'].str.startswith('PS-')]
    
    print(f"\nAS (Aligned) vs PS (Padded) Comparison:")
    print(f"  AS Average Time: {as_results['avg_time'].mean():.4f}s")
    print(f"  PS Average Time: {ps_results['avg_time'].mean():.4f}s")
    print(f"  AS Average Memory: {as_results['avg_memory_mb'].mean():.2f}MB")
    print(f"  PS Average Memory: {ps_results['avg_memory_mb'].mean():.2f}MB")
    
    print(f"\nDetailed results saved in: {results_dir}")
    print("="*80)
    
    return final_results

# Global variable for encoding list (for reference)
encodings = [
    'AS-One Hot', 
    'AS-K-mers', 
    'AS-FFT',
    'AS-Wavelet', 
    'AS-K-mers + FFT', 
    'AS-One Hot + FFT',
    'AS-K-mers + Wavelet', 
    'AS-One Hot + Wavelet', 
    'PS-One Hot',
    'PS-K-mers', 
    'PS-FFT', 
    'PS-Wavelet', 
    'PS-K-mers + FFT',
    'PS-One Hot + FFT', 
    'PS-K-mers + Wavelet', 
    'PS-One Hot + Wavelet'
]

if __name__ == "__main__":
    try:
        # Run complete benchmark
        results = run_complete_benchmark()
        
        if results is not None:
            print("\nBenchmark completed successfully!")
            print("Check the 'benchmark_results' directory for the CSV file.")
            print("You can now generate visualizations using the results.")
        else:
            print("\nBenchmark failed. Please check the error messages above.")
        
    except Exception as e:
        print(f"Unexpected error during execution: {e}")
        import traceback
        traceback.print_exc()