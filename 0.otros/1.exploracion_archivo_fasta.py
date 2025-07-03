from Bio import SeqIO
import pandas as pd
import duckdb
import re
import plotly.graph_objects as go
import numpy as np


def parse_taxonomy(description):
    """Extrae información taxonómica de la descripción"""
    tax_match = re.search(r'd([^;]+);p([^;]+);c([^;]+);o([^;]+);f([^;]+);g([^;]+);s__([^\s]+)', description)
    if tax_match:
        return {
            'domain': tax_match.group(1),
            'phylum': tax_match.group(2),
            'class': tax_match.group(3),
            'order': tax_match.group(4),
            'family': tax_match.group(5),
            'genus': tax_match.group(6),
            'species': tax_match.group(7)
        }
    return dict.fromkeys(['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species'])

def fna_to_dataframe(fna_file):
    """Convierte archivo FNA a DataFrame"""
    records = []
    
    for record in SeqIO.parse(fna_file, "fasta"):
        sequence = str(record.seq)
        tax_info = parse_taxonomy(record.description)
        
        records.append({
            'sequence_id': record.id,
            'sequence': sequence,
            'sequence_length': len(sequence),
            'gc_content': (sequence.count("G") + sequence.count("C")) / len(sequence) * 100,
            'domain': tax_info['domain'],
            'phylum': tax_info['phylum'],
            'class': tax_info['class'],
            'order': tax_info['order'],
            'family': tax_info['family'],
            'genus': tax_info['genus'],
            'species': tax_info['species']
        })
    
    return pd.DataFrame(records)

def export_to_fasta(df, output_file):
    """Exporta DataFrame a archivo FASTA con información seleccionada"""
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            header = f">{row['sequence_id']} length={row['sequence_length']} GC={row['gc_content']} genus={row['genus']}"
            f.write(f"{header}\n{row['sequence']}\n")

def read_exported_fasta(fasta_file):
    """Lee el archivo FASTA exportado"""
    records = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        header_info = record.description.split()
        
        # Extraer información del header
        sequence_id = header_info[0]
        length = int(header_info[1].split('=')[1])
        gc = float(header_info[2].split('=')[1])
        genus = header_info[3].split('=')[1]
        
        records.append({
            'sequence_id': sequence_id,
            'sequence': str(record.seq),
            'sequence_length': length,
            'gc_content': gc,
            'genus': genus
        })
    
    return pd.DataFrame(records)


df = fna_to_dataframe('datos/datos_originales_ncbi.fna')


columnas=["domain",'phylum','class','order','family','genus','species']

for c in columnas:
    print(f"{c}: {df[c].nunique()}")


for i,v in enumerate(df['genus']):
    df.loc[i,'genus']=v.replace('__', '')


dfcopia=df.copy()


df=duckdb.sql("""
select * from df
           where sequence_length<1600 and sequence_length>1300
""").to_df()


columnas=["domain",'phylum','class','order','family','genus','species']

for c in columnas:
    print(f">>>{c}: {df[c].nunique()}")


def plot_sequence_length_enhanced(df, col, color_scheme='Viridis', 
                                 show_stats=True, show_cumulative=False,
                                 bin_size=None, text_size_factor=1.3):
    """
    Enhanced function to visualize DNA sequence length distribution
    
    Parameters:
    - df: DataFrame containing the data
    - col: Column name with sequence lengths
    - color_scheme: Color scheme for the chart ('Viridis', 'Plasma', 'Turbo', etc.)
    - show_stats: Show descriptive statistics
    - show_cumulative: Show cumulative distribution
    - bin_size: Bin size (for very large sequences)
    - text_size_factor: Factor to increase all text sizes
    """
    # Extract length data
    sequence_lengths = df[col].copy()
    
    # Calculate statistics
    stats = {
        'Mean': np.mean(sequence_lengths),
        'Median': np.median(sequence_lengths),
        'Mode': sequence_lengths.mode().iloc[0] if not sequence_lengths.mode().empty else None,
        'Min': sequence_lengths.min(),
        'Max': sequence_lengths.max(),
        'Std. Dev': np.std(sequence_lengths)
    }
    
    # Determine if we should use bins for widely dispersed distributions
    unique_lengths = len(sequence_lengths.unique())
    use_histogram = unique_lengths > 50 or (stats['Max'] - stats['Min']) > 100
    
    if bin_size is None:
        # Calculate bin_size automatically to show more columns
        # We want at least 30 bins for a more detailed histogram
        range_length = stats['Max'] - stats['Min']
        bin_size = max(1, int(range_length / 40))  # Ensure at least 40 bins for detailed view
    
    # Create base figure
    fig = go.Figure()
    
    # Count sequence length frequencies
    sequence_counts = df[col].value_counts().reset_index()
    sequence_counts.columns = ['Length', 'Frequency']
    sequence_counts = sequence_counts.sort_values('Length')
    
    # Calculate cumulative distribution
    total = sequence_counts['Frequency'].sum()
    sequence_counts['Percentage'] = sequence_counts['Frequency'] * 100 / total
    sequence_counts['Cumulative'] = sequence_counts['Frequency'].cumsum() * 100 / total
    
    # Add bar chart with larger bars and no colorbar
    fig.add_trace(go.Bar(
        x=sequence_counts['Length'],
        y=sequence_counts['Frequency'],
        name='Frequency',
        marker=dict(
            color=sequence_counts['Frequency'],
            colorscale=color_scheme,
            showscale=False,  # Remove the colorbar
        ),
        hovertemplate='Length: %{x}<br>Frequency: %{y}<br>Percentage: %{text:.2f}%<extra></extra>',
        text=sequence_counts['Percentage']
    ))
    
    # Add cumulative distribution line
    if show_cumulative:
        fig.add_trace(go.Scatter(
            x=sequence_counts['Length'],
            y=sequence_counts['Cumulative'],
            mode='lines+markers',
            name='% Cumulative',
            line=dict(color='rgba(219, 64, 82, 0.8)', width=3),
            marker=dict(size=8),
            yaxis='y2',
            hovertemplate='Length: %{x}<br>Cumulative: %{y:.2f}%<extra></extra>'
        ))
    
    # Add reference lines for important statistics
    if show_stats:
        # Add vertical line for mean
        fig.add_vline(x=stats['Mean'], line_width=2, line_dash="dash", line_color="green",
                     annotation=dict(
                         text=f"Mean: {stats['Mean']:.2f}",
                         font=dict(size=14 * text_size_factor, color="green"),
                         xanchor="right",
                         yanchor="top"
                     ))
        
        # Add vertical line for median
        fig.add_vline(x=stats['Median'], line_width=2, line_dash="dash", line_color="red",
                     annotation=dict(
                         text=f"Median: {stats['Median']}",
                         font=dict(size=14 * text_size_factor, color="red"),
                         xanchor="left",
                         yanchor="top"
                     ))
    
    # Create statistics text block with larger font
    stats_text = "<br>".join([
        f"<b style='font-size:{14 * text_size_factor}px'>Descriptive Statistics:</b>",
        f"<span style='font-size:{13 * text_size_factor}px'>Mean: {stats['Mean']:.2f}</span>",
        f"<span style='font-size:{13 * text_size_factor}px'>Median: {stats['Median']}</span>",
        f"<span style='font-size:{13 * text_size_factor}px'>Mode: {stats['Mode']}</span>",
        f"<span style='font-size:{13 * text_size_factor}px'>Min: {stats['Min']}, Max: {stats['Max']}</span>",
        f"<span style='font-size:{13 * text_size_factor}px'>Std. Dev: {stats['Std. Dev']:.2f}</span>",
        f"<span style='font-size:{13 * text_size_factor}px'>Total sequences: {len(sequence_lengths)}</span>"
    ])
    
    # Customize the layout with larger fonts
    fig.update_layout(
        title={
            'text': 'DNA Sequence Length Distribution',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24 * text_size_factor, color='black')
        },
        xaxis_title={
            'text': 'Sequence Length (bp)',
            'font': dict(size=18 * text_size_factor)
        },
        yaxis_title={
            'text': 'Number of Sequences',
            'font': dict(size=18 * text_size_factor)
        },
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14 * text_size_factor)
        ),
        margin=dict(l=80, r=80, t=120, b=80),
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=0.99,
                y=0.99,
                showarrow=False,
                text=stats_text,
                align='right',
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='grey',
                borderwidth=1
            )
        ],
        width=1600, 
        height=900,
        font=dict(size=14 * text_size_factor)  # Global font size increase
    )
    
    # Configure secondary Y-axis for cumulative distribution
    if show_cumulative:
        fig.update_layout(
            yaxis2=dict(
                title={
                    'text': 'Cumulative Percentage (%)',
                    'font': dict(size=18 * text_size_factor, color='rgba(219, 64, 82, 0.8)')
                },
                tickfont=dict(color='rgba(219, 64, 82, 0.8)', size=14 * text_size_factor),
                overlaying='y',
                side='right',
                range=[0, 100]
            )
        )
    
    # Increase tick font size for better readability
    fig.update_xaxes(tickfont=dict(size=14 * text_size_factor))
    fig.update_yaxes(tickfont=dict(size=14 * text_size_factor))
    
    # Show and save graph
    fig.show()
    
    # Save as PNG and HTML for flexibility
    # fig.write_image('distribucion_sin_filtrar.png', scale=5, width=1600, height=900)
    # fig.write_image('bact_distribution_enhanced.png', scale=5, width=1600, height=900)
    # fig.write_html('bact_distribution_interactive.html', include_plotlyjs='cdn')
    
    # return fig


plot_sequence_length_enhanced(df, 'sequence_length', 
                             color_scheme='Viridis',
                             show_stats=True, 
                             show_cumulative=False,
                             text_size_factor=1.3)


plot_sequence_length_enhanced(dfcopia, 'sequence_length', 
                             color_scheme='Viridis',
                             show_stats=True, 
                             show_cumulative=False,
                             text_size_factor=1.3)



#Limpieza de las secuencias que no tengan los nucleótidos ACGT
base=set('ACGT')
drop_idx=[]
for i,j in enumerate(df['sequence']):
    if set(j)!=base:
        drop_idx.append(i)
    
df=df.drop(drop_idx)
df=df.reset_index()


# Esto se hizo para explorar los datos y conteo de datos por género

conteo={'genus':[],'conteo':[]}
for i in df['genus'].unique():
    conteo['genus'].append(i)
    conteo['conteo'].append(len(df[df['genus']==i]))

df_conteo=pd.DataFrame(conteo)

temp=df_conteo['conteo']>=100
df_conteo=df_conteo.loc[temp]
df_conteo=df_conteo.reset_index(drop=True)

genes=df_conteo['genus'].tolist()


for g in genes:
    print(f">>>{g}")


# Filtro para los géneros que tienen más de 250 represetantes, se limitó a máximo 250 secuencias por género
templist=[]
for i in genes:
    temp=df[df['genus']==i]
    temp=temp.sort_values('sequence_length', ascending=False)
    if len(temp)>250:
        temp=temp[0:250]
    templist.append(temp)
# df_original=df.copy()
df=pd.concat(templist)
df=df.reset_index(drop=True)



df


# Se exportan los datos filtrados del df hacia un archivo fasta para utilizar 
# el programa MAFFT para el alineamiento de las secuencias
export_to_fasta(df, "datos/final_no_estandar.fasta")



# Lectura del archivo generado por el programa MFFT
# mafft --auto --thread 2 --maxiterate 1000 --localpair final_no_estandar.fasta > fne.fasta
df_imported = read_exported_fasta("datos/fne.fasta")


# Se reemplazan los gaps generados por el alineamiento por la letra N
df['se']=[i.replace('-','N').upper() for i in df_imported['sequence']]


# Guardado del archivo ya procesado y listo para los modelos
df.to_csv('datos/dmfne.csv', index=False)


