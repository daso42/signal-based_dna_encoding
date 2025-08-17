import time
import numpy as np
import pandas as pd
import duckdb
import os
pd.options.mode.chained_assignment = None

import plotly
import plotly.graph_objs as go

import pywt
from scipy.fft import fft
from itertools import product



# Sequence processing functions
def pad_sequences(sequences, maxlen):
    """Pad sequences to equal length"""
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            seq += 'N' * (maxlen - len(seq))
        else:
            seq = seq[:maxlen]
        padded_sequences.append(seq)
    return padded_sequences


def fourier(sequences, is_str=True):
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
    # Generar todas las posibles combinaciones
    kmers = product(unique_chars, repeat=k)
    # Crear el diccionario
    kmer_dict = {''.join(kmer): i for i,kmer in enumerate(kmers)}
    return kmer_dict

def k_mers(sequencias, k=3, unique_chars=set('ACGNT')):
    kmers_map = generate_kmers_dict(k, unique_chars)
    templist = []
    for seq in sequencias:
        temp = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        templist.append([kmers_map[i] for i in temp])
    return templist

def one_hot(sequences, max_len, unique_chars=set('ACGNT'), reshape=True):
    mapping = {j:i for i,j in enumerate(unique_chars)}
    sequencias_procesadas = []
    if reshape == True:
        for s in sequences:
            temp = np.zeros((max_len,len(unique_chars)))
            for c in zip(s,temp):
                    c[1][mapping[c[0]]] = 1
            sequencias_procesadas.append(temp.reshape(-1))
        return sequencias_procesadas
    elif reshape == False:
        for s in sequences:
            temp = np.zeros((max_len,len(unique_chars)))
            for c in zip(s,temp):
                    c[1][mapping[c[0]]] = 1
            sequencias_procesadas.append(temp)
        return sequencias_procesadas

def wavelet(sequences, numeric=False, wavelet='db1', level=5):
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


def plot_encoding_by_genus(df, genus_name, encoding, output_dir='plots'):
    """
    Create plots for any encoding method by genus
    
    Parameters:
    - df: DataFrame with encoded sequences
    - genus_name: Name of the genus to plot
    - encoding: Encoding method from the encodings list
    - output_dir: Directory to save plots
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for specific genus
    genus_data = df[df['genus'] == genus_name].reset_index(drop=True)
    
    if len(genus_data) == 0:
        print(f"No data found for genus {genus_name}")
        return None
    
    # Get encoding data
    if encoding not in df.columns:
        print(f"Encoding {encoding} not found in dataframe")
        return None
    
    # Calculate transparency based on number of sequences
    n_sequences = len(genus_data)
    transparency = min(1.0, 1.0 / n_sequences)
    
    # Create figure
    fig = go.Figure()
    
    # Collect all encoding data
    encoding_data_list = []
    
    # Add individual sequences with transparency
    for i, row in genus_data.iterrows():
        encoding_data = np.array(row[encoding])
        encoding_data_list.append(encoding_data)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(encoding_data))),
            y=encoding_data,
            mode='lines',
            name=f"{genus_name} - Seq {i+1}",
            showlegend=False,
            line=dict(width=1, color=f'rgba(120,121,117,{transparency})')
        ))
    
    # Add average line
    if encoding_data_list:
        avg_data = np.mean(encoding_data_list, axis=0)
        fig.add_trace(go.Scatter(
            x=list(range(len(avg_data))),
            y=avg_data,
            mode='lines',
            name=f"{genus_name} - Average",
            line=dict(color='rgb(70, 130, 180)', width=3),
            showlegend=False
        ))
    
    # Determine sequence type and y-axis label
    sequence_type = 'AS' if encoding.startswith('AS_') else 'PS'
    
    # Set appropriate y-axis label based on encoding type
    if 'FFT' in encoding:
        y_label = '<b>Amplitude</b>'
    elif 'Wavelet' in encoding:
        y_label = '<b>Wavelet Coefficients</b>'
    elif 'One Hot' in encoding:
        y_label = '<b>One-Hot Values</b>'
    elif 'K-mers' in encoding:
        y_label = '<b>K-mer Values</b>'
    else:
        y_label = '<b>Signal Values</b>'
    
    # Create title
    title_text = f"<b>{encoding} - {genus_name} ({sequence_type})</b><br><span style='font-size:14px; color:rgb(100,100,100)'>N = {n_sequences} sequences</span>"
    
    # Update layout with Quicksand font
    fig.update_layout(
        title={
            'text': title_text,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        width=700,
        height=520,
        showlegend=False,
        font=dict(
            family="Quicksand, Arial, sans-serif",
            size=12,
            color="black"
        ),
        xaxis=dict(title="<b>Frequency</b>" if 'FFT' in encoding else "<b>Position</b>"),
        yaxis=dict(title=y_label),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Create filename
    safe_genus_name = genus_name.replace(' ', '_').replace('/', '_')
    filename_base = f"{output_dir}/{encoding}_{safe_genus_name}"
    
    # Save in multiple formats
    fig.write_image(f"{filename_base}.png", scale=2)
    fig.write_image(f"{filename_base}.pdf")
    fig.write_image(f"{filename_base}.svg")
    
    print(f"Saved: {filename_base} (PNG, PDF, SVG)")
    
    return fig


def generate_all_plots(df, output_dir='plots'):
    """Generate plots for all encodings and all genera"""
    
    # List of all encodings
    encodings = [
        'AS_One Hot', 'AS_K-mers', 'AS_FFT', 'AS_Wavelet', 
        'AS_K-mers + FFT', 'AS_One Hot + FFT',
        'AS_K-mers + Wavelet', 'AS_One Hot + Wavelet', 
        'PS_One Hot', 'PS_K-mers', 'PS_FFT', 'PS_Wavelet', 
        'PS_K-mers + FFT', 'PS_One Hot + FFT', 
        'PS_K-mers + Wavelet', 'PS_One Hot + Wavelet'
    ]
    
    # Get unique genera
    genera = df['genus'].unique()
    
    print(f"Generating plots for {len(genera)} genera and {len(encodings)} encodings...")
    
    # Generate plots for each combination
    for genus in genera:
        print(f"\nProcessing genus: {genus}")
        for encoding in encodings:
            plot_encoding_by_genus(df, genus, encoding, output_dir)
    
    print(f"\nAll plots saved in '{output_dir}' directory")



df = pd.read_csv('datos/datos_filtrados_sin_encoding.csv')
df = duckdb.sql(
    """
    SELECT aligned_sequence, original_sequence, genus FROM df
    WHERE purpose=0
    """
).to_df()


max_aligned_len = len(df['aligned_sequence'][0])
max_original_len = max([len(seq) for seq in df['original_sequence']])

# Pad original sequences
df['ps'] = pad_sequences(df['original_sequence'], max_original_len)

# Aligned Sequences (AS) encodings
df['AS_One Hot'] = one_hot(df['aligned_sequence'].values, len(df['aligned_sequence'][0]))
df['AS_K-mers'] = k_mers(df['aligned_sequence'].values)
df['AS_FFT'] = fourier(df['aligned_sequence'].values)
df['AS_Wavelet'] = wavelet(df['aligned_sequence'].values)
df['AS_K-mers + FFT'] = fourier(df['AS_K-mers'].values, False)
df['AS_One Hot + FFT'] = fourier(df['AS_One Hot'].values, False)
df['AS_K-mers + Wavelet'] = wavelet(df['AS_K-mers'].values, True)
df['AS_One Hot + Wavelet'] = wavelet(df['AS_One Hot'].values, True)

# Padded Sequences (PS) encodings
df['PS_One Hot'] = one_hot(df['original_sequence'].values, len(df['original_sequence'][0]))
df['PS_K-mers'] = k_mers(df['original_sequence'].values)
df['PS_FFT'] = fourier(df['original_sequence'].values)
df['PS_Wavelet'] = wavelet(df['original_sequence'].values)
df['PS_K-mers + FFT'] = fourier(df['PS_K-mers'].values, False)
df['PS_One Hot + FFT'] = fourier(df['PS_One Hot'].values, False)
df['PS_K-mers + Wavelet'] = wavelet(df['PS_K-mers'].values, True)
df['PS_One Hot + Wavelet'] = wavelet(df['PS_One Hot'].values, True)

# List of all encodings
encodings = [
    'AS_One Hot', 
    'AS_K-mers', 
    'AS_FFT',
    'AS_Wavelet', 
    'AS_K-mers + FFT', 
    'AS_One Hot + FFT',
    'AS_K-mers + Wavelet', 
    'AS_One Hot + Wavelet', 
    'PS_One Hot',
    'PS_K-mers', 
    'PS_FFT', 
    'PS_Wavelet', 
    'PS_K-mers + FFT',
    'PS_One Hot + FFT', 
    'PS_K-mers + Wavelet', 
    'PS_One Hot + Wavelet'
]

# Get unique genera
genera = df['genus'].unique()

print(f"Generating plots for {len(genera)} genera and {len(encodings)} encodings...")

# Generate plots for each combination
for genus in genera:
    print(f"\nProcessing genus: {genus}")
    for encoding in encodings:
        plot_encoding_by_genus(df, genus, encoding, 'plots')