
import random
import pandas as pd

def generate_dna_sequences(n_sequences, sequence_length, nucleotides="ACGNT", seed=None):
    """
    Genera secuencias aleatorias de ADN.
   
    Parámetros:
    - n_sequences: Número de secuencias a generar
    - sequence_length: Longitud de cada secuencia
    - nucleotides: Nucleótidos disponibles (por defecto "ACGNT")
    - seed: Semilla para reproducibilidad (opcional)
   
    Retorna:
    - Lista de secuencias de ADN
    """
    if seed is not None:
        random.seed(seed)
   
    sequences = []
    nucleotide_list = list(nucleotides)
   
    for i in range(n_sequences):
        sequence = ''.join(random.choices(nucleotide_list, k=sequence_length))
        sequences.append(sequence)
   
    return sequences

def save_sequences_to_txt(sequences, filename="dna_sequences.txt"):
    """
    Guarda las secuencias en un archivo de texto.
   
    Parámetros:
    - sequences: Lista de secuencias de ADN
    - filename: Nombre del archivo de salida
    """
    with open(filename, 'w') as file:
        for i, seq in enumerate(sequences, 1):
            file.write(f">Secuencia_{i}\n{seq}\n")
    print(f"Secuencias guardadas en '{filename}'")

def create_dataframe(sequences):
    """
    Crea un DataFrame con las secuencias generadas.
   
    Parámetros:
    - sequences: Lista de secuencias de ADN
   
    Retorna:
    - DataFrame con las secuencias
    """
    df = pd.DataFrame({
        'ID': [f'Seq_{i+1}' for i in range(len(sequences))],
        'Secuencia': sequences,
        'Longitud': [len(seq) for seq in sequences],
        'GC_content': [calculate_gc_content(seq) for seq in sequences]
    })
    return df

def calculate_gc_content(sequence):
    """
    Calcula el contenido GC de una secuencia.
   
    Parámetros:
    - sequence: Secuencia de ADN
   
    Retorna:
    - Porcentaje de contenido GC
    """
    gc_count = sequence.count('G') + sequence.count('C')
    return round((gc_count / len(sequence)) * 100, 2) if len(sequence) > 0 else 0.0

# Función principal para uso fácil
def generate_and_save_dna(n_sequences, sequence_length, save_txt=True, return_df=True,
                         nucleotides="ACGNT", filename="dna_sequences.txt", seed=None):
    """
    Función principal que genera secuencias y las guarda/retorna según se especifique.
   
    Parámetros:
    - n_sequences: Número de secuencias a generar
    - sequence_length: Longitud de cada secuencia
    - save_txt: Si guardar en archivo txt (True por defecto)
    - return_df: Si retornar DataFrame (True por defecto)
    - nucleotides: Nucleótidos disponibles
    - filename: Nombre del archivo de salida
    - seed: Semilla para reproducibilidad
   
    Retorna:
    - DataFrame si return_df=True, None en caso contrario
    """
    print(f"Generando {n_sequences} secuencias de longitud {sequence_length}...")
   
    # Generar secuencias
    sequences = generate_dna_sequences(n_sequences, sequence_length, nucleotides, seed)
   
    # Guardar en archivo txt si se solicita
    if save_txt:
        save_sequences_to_txt(sequences, filename)
   
    # Crear y retornar DataFrame si se solicita
    if return_df:
        df = create_dataframe(sequences)
        print(f"DataFrame creado con {len(df)} secuencias")
        return df
   
    print("Proceso completado!")
    return None



generate_and_save_dna(
    nucleotides='ACGNT'
    n_sequences=100,
    sequence_length=100,
    save_txt=True,
    return_df=False,
    filename="secuencias_test.txt"
)


