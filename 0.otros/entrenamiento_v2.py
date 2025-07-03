# Se forzó a realizar el entrenamiento con un hilo porque los modelos estaban entrenando con todos los hilos incluso cuando se les especificaba que usaran solo 1 o 2
import os
os.environ['OMP_NUM_THREADS'] = '1'

# Librerías básicas
import time
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from ast import literal_eval

# Librerías de machine learning
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from xgboost import XGBClassifier

# Librerías para encoding
import pywt
from scipy.fft import fft
from itertools import product

# Configurar pandas
pd.options.mode.chained_assignment = None

print("Iniciando entrenamiento")

# Variables compartidas
N_CORES = 1
RANDOM_STATE = 42
TEST_SIZE = 0.2



# Crear directorios necesarios
os.makedirs("models/SVM", exist_ok=True)
os.makedirs("models/RandomForest", exist_ok=True) 
os.makedirs("models/XGBoost", exist_ok=True)
os.makedirs("results/training_results", exist_ok=True)

print(f"Configuración: n_cores={N_CORES}, random_state={RANDOM_STATE}, test_size={TEST_SIZE}")
print("Threading limitado a 1 core configurado")

def log_with_time(message):
    """Imprime mensaje con timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def save_model_safely(model, model_name, encoding_name, algorithm, metrics, params):
    """Guarda modelo con verificación automática"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear directorio si no existe
    model_dir = f"models/{algorithm}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Nombre del archivo
    filename = f"{model_dir}/{encoding_name}_{timestamp}.joblib"
    metadata_file = f"{model_dir}/{encoding_name}_{timestamp}_metadata.json"
    
    try:
        # Guardar modelo
        joblib.dump(model, filename)
        log_with_time(f"Modelo guardado: {filename}")
        
        # Guardar metadata
        metadata = {
            'model_name': model_name,
            'encoding': encoding_name,
            'algorithm': algorithm,
            'timestamp': timestamp,
            'accuracy': float(metrics['accuracy']),
            'f1_score_weighted': float(metrics['f1_score_weighted']),
            'f1_score_macro': float(metrics['f1_score_macro']),
            'parameters': params,
            'model_file': filename
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Verificar que se guardó bien
        test_model = joblib.load(filename)
        log_with_time(f"Verificación exitosa: {filename}")
        
        return filename, metadata_file
        
    except Exception as e:
        log_with_time(f"Error guardando modelo: {e}")
        return None, None

def save_training_results(results, filename):
    """Guarda resultados de entrenamiento para evaluación posterior"""
    filepath = f"results/training_results/{filename}"
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        log_with_time(f"Resultados guardados: {filepath}")
        return filepath
    except Exception as e:
        log_with_time(f"Error guardando resultados: {e}")
        return None

# ============================================================================
# FUNCIONES DE ENCODING
# ============================================================================

def fourier(sequences, is_str=True):
    if is_str:
        templist=[]
        for seq in sequences:
            num_seq=[ord(char) for char in seq]
            fft_seq=fft(num_seq)
            fft_seq=np.abs(fft_seq)
            templist.append(fft_seq[1:len(fft_seq)//2])
        return templist
    else:
        templist=[]
        for seq in sequences:
            fft_seq=fft(seq)
            fft_seq=np.abs(fft_seq)
            templist.append(fft_seq[1:len(fft_seq)//2])
        return templist

def generate_kmers_dict(k, unique_chars=set('ACGNT')):
    kmers = product(unique_chars, repeat=k)
    kmer_dict = {''.join(kmer): i for i,kmer in enumerate(kmers)}
    return kmer_dict

def k_mers(sequencias, k=3, unique_chars=set('ACGNT')):
    kmers_map=generate_kmers_dict(k, unique_chars)
    templist=[]
    for seq in sequencias:
        temp=[seq[i:i+k] for i in range(len(seq) - k + 1)]
        templist.append([kmers_map[i] for i in temp])
    return templist

def one_hot(sequences, max_len, unique_chars=set('ACGNT'), reshape=True):
    mapping={j:i for i,j in enumerate(unique_chars)}
    sequencias_procesadas=[]
    if reshape==True:
        for s in sequences:
            temp=np.zeros((max_len,len(unique_chars)))
            for c in zip(s,temp):
                    c[1][mapping[c[0]]]=1
            sequencias_procesadas.append(temp.reshape(-1))
        return sequencias_procesadas
    elif reshape==False:
        for s in sequences:
            temp=np.zeros((max_len,len(unique_chars)))
            for c in zip(s,temp):
                    c[1][mapping[c[0]]]=1
            sequencias_procesadas.append(temp)
        return sequencias_procesadas

def wavelet(sequences, numeric=False, wavelet='db1', level=5):
    templist=[]
    if numeric==False:
        for seq in sequences:
            num_seq=[ord(char) for char in seq]
            coeffs=pywt.wavedec(num_seq, wavelet, level)
            templist.append(np.concatenate(coeffs))
        return templist
    elif numeric==True:
        for seq in sequences:
            coeffs=pywt.wavedec(seq, wavelet, level)
            templist.append(np.concatenate(coeffs))
        return templist

def pad_sequences(sequences, maxlen):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            seq += 'N' * (maxlen - len(seq))  
        else:
            seq = seq[:maxlen]
        padded_sequences.append(seq)
    return padded_sequences

# ============================================================================
# FUNCIONES DE EVALUACIÓN
# ============================================================================

def calculate_metrics(y_test, y_pred, y_score, classes):
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score_weighted': f1_score(y_test, y_pred, average='weighted'),
        'f1_score_macro': f1_score(y_test, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro')
    }
    
    # Calculate per-class metrics
    n_classes = len(classes)
    class_metrics = {
        'sensitivity_per_class': {},
        'specificity_per_class': {},
        'precision_per_class': {},
        'recall_per_class': {}
    }
    
    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        
        class_metrics['sensitivity_per_class'][classes[i]] = TP / (TP + FN) if (TP + FN) > 0 else 0
        class_metrics['specificity_per_class'][classes[i]] = TN / (TN + FP) if (TN + FP) > 0 else 0
        class_metrics['precision_per_class'][classes[i]] = TP / (TP + FP) if (TP + FP) > 0 else 0
        class_metrics['recall_per_class'][classes[i]] = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    metrics.update(class_metrics)
    return metrics

def evaluate_model_simple(model, X_test, y_test, class_names):
    """Evaluación simple sin generar gráficos"""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    
    # Obtener las clases numéricas únicas
    classes_num = np.unique(y_test)
    # Mapear las clases numéricas a nombres reales
    classes = [class_names[i] for i in classes_num]
    
    metrics = calculate_metrics(y_test, y_pred, y_score, classes)
    
    # Preparar datos para evaluación posterior
    evaluation_data = {
        'y_true': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
        'y_pred': y_pred.tolist(),
        'y_proba': y_score.tolist(),
        'classes': classes,
        'class_names_mapping': class_names
    }
    
    return metrics, evaluation_data

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO CORREGIDAS
# ============================================================================

def train_svm_improved(X_train, X_test, y_train, y_test, encoding_name, class_names, 
                      kernel='rbf', C=1.0, gamma='scale', n_cores=1, random_state=42):
    """Entrena SVM con parámetros específicos"""
    
    log_with_time(f"Entrenando SVM con {encoding_name}")
    log_with_time(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    start_time = time.time()
    
    # Entrenar modelo - SVM no usa n_jobs, pero es inherentemente single-threaded
    model = SVC(probability=True, kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluar modelo
    metrics, evaluation_data = evaluate_model_simple(model, X_test, y_test, class_names)
    
    best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}
    
    log_with_time(f"SVM {encoding_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score_weighted']:.4f}, Tiempo: {training_time:.2f}s")
    
    # Guardar modelo
    model_file, metadata_file = save_model_safely(
        model, 'SVM', encoding_name, 'SVM', metrics, best_params
    )
    
    # Preparar resultados completos
    results = {
        'model_name': 'SVM',
        'encoding': encoding_name,
        'metrics': metrics,
        'parameters': best_params,
        'training_time': training_time,
        'model_file': model_file,
        'metadata_file': metadata_file,
        'evaluation_data': evaluation_data
    }
    
    return results

def train_random_forest_improved(X_train, X_test, y_train, y_test, encoding_name, class_names,
                                n_estimators=100, max_depth=None, min_samples_split=2, 
                                n_cores=1, random_state=42):
    """Entrena Random Forest con parámetros específicos"""
    
    log_with_time(f"Entrenando Random Forest con {encoding_name}")
    log_with_time(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    start_time = time.time()
    
    # Entrenar modelo - FORZAR n_jobs=1
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=1,  # FORZAR a 1, no usar la variable
        verbose=0
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluar modelo
    metrics, evaluation_data = evaluate_model_simple(model, X_test, y_test, class_names)
    
    best_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }
    
    log_with_time(f"Random Forest {encoding_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score_weighted']:.4f}, Tiempo: {training_time:.2f}s")
    
    # Guardar modelo
    model_file, metadata_file = save_model_safely(
        model, 'Random Forest', encoding_name, 'RandomForest', metrics, best_params
    )
    
    # Preparar resultados completos
    results = {
        'model_name': 'Random Forest',
        'encoding': encoding_name,
        'metrics': metrics,
        'parameters': best_params,
        'training_time': training_time,
        'model_file': model_file,
        'metadata_file': metadata_file,
        'evaluation_data': evaluation_data
    }
    
    return results

def train_xgboost_improved(X_train, X_test, y_train, y_test, encoding_name, class_names,
                          n_estimators=100, max_depth=6, learning_rate=0.3, 
                          subsample=0.8, colsample_bytree=0.8, n_cores=1, random_state=42):
    """Entrena XGBoost con parámetros específicos"""
    
    log_with_time(f"Entrenando XGBoost con {encoding_name}")
    log_with_time(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    start_time = time.time()
    
    # Entrenar modelo - CONFIGURACIÓN COMPLETA PARA 1 CORE
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric='mlogloss',
        tree_method='hist',
        grow_policy='lossguide',
        random_state=random_state,
        n_jobs=1,           # FORZAR a 1, no usar la variable
        verbosity=0,        # Sin verbosidad
        nthread=1,          # XGBoost específico
        use_label_encoder=False
    )
    model.fit(X_train, y_train, verbose=False)
    
    training_time = time.time() - start_time
    
    # Evaluar modelo
    metrics, evaluation_data = evaluate_model_simple(model, X_test, y_test, class_names)
    
    best_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    
    log_with_time(f"XGBoost {encoding_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score_weighted']:.4f}, Tiempo: {training_time:.2f}s")
    
    # Guardar modelo
    model_file, metadata_file = save_model_safely(
        model, 'XGBoost', encoding_name, 'XGBoost', metrics, best_params
    )
    
    # Preparar resultados completos
    results = {
        'model_name': 'XGBoost',
        'encoding': encoding_name,
        'metrics': metrics,
        'parameters': best_params,
        'training_time': training_time,
        'model_file': model_file,
        'metadata_file': metadata_file,
        'evaluation_data': evaluation_data
    }
    
    return results

# ============================================================================
# CARGAR Y PREPARAR DATOS
# ============================================================================

log_with_time("Cargando datos...")

# Cargar datos
df = pd.read_csv('datos_filtrados_sin_encoding.csv')
df = df.rename(columns={'sequence': 'aligned_sequence'})

log_with_time(f"Datos cargados: {len(df)} filas")

# Padding de secuencias
maxlen = max([len(i) for i in df['original_sequence']]) 
df['padded_sequences'] = pad_sequences(df['original_sequence'], maxlen)

# Calcular longitudes
df['len_ps'] = [len(i) for i in df['padded_sequences']]
df['len_as'] = [len(i) for i in df['aligned_sequence']]

log_with_time(f"Secuencias procesadas - Max length: {maxlen}")

# ============================================================================
# GENERAR ENCODINGS
# ============================================================================

log_with_time("Generando encodings...")

# Aligned Sequences (AS)
df['AS_One Hot'] = one_hot(df['aligned_sequence'].values, len(df['aligned_sequence'][0]))
df['AS_K-mers'] = k_mers(df['aligned_sequence'].values)
df['AS_FFT'] = fourier(df['aligned_sequence'].values)
df['AS_Wavelet'] = wavelet(df['aligned_sequence'].values)
df['AS_K-mers + FFT'] = fourier(df['AS_K-mers'].values, False)
df['AS_One Hot + FFT'] = fourier(df['AS_One Hot'].values, False)
df['AS_K-mers + Wavelet'] = wavelet(df['AS_K-mers'].values, True)
df['AS_One Hot + Wavelet'] = wavelet(df['AS_One Hot'].values, True)

# Padded Sequences (PS)
df['PS_One Hot'] = one_hot(df['padded_sequences'].values, len(df['padded_sequences'][0]))
df['PS_K-mers'] = k_mers(df['padded_sequences'].values)
df['PS_FFT'] = fourier(df['padded_sequences'].values)
df['PS_Wavelet'] = wavelet(df['padded_sequences'].values)
df['PS_K-mers + FFT'] = fourier(df['PS_K-mers'].values, False)
df['PS_One Hot + FFT'] = fourier(df['PS_One Hot'].values, False)
df['PS_K-mers + Wavelet'] = wavelet(df['PS_K-mers'].values, True)
df['PS_One Hot + Wavelet'] = wavelet(df['PS_One Hot'].values, True)

log_with_time("Todos los encodings generados")

# Cargar mapeo de clases
mapeo_df = pd.read_csv('mapeo_clases.csv')
df = df.merge(mapeo_df.rename(columns={'model_class': 'clases_modelos'}), 
              on='genus', how='left')

# Crear mapeo inverso para nombres de clases
tempdf = df[['genus', 'clases_modelos']].drop_duplicates()
reverse_map_genus = {v2: v1 for v1, v2 in tempdf.values}

log_with_time(f"Mapeo de clases cargado - {len(reverse_map_genus)} clases")

# ============================================================================
# DEFINIR ENCODINGS Y PARÁMETROS ÓPTIMOS
# ============================================================================

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

# Parámetros óptimos del código original
parametros_optimos = {
    'SVM': {
        'AS_One Hot': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'AS_K-mers': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_FFT': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_Wavelet': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_K-mers + FFT': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_One Hot + FFT': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_K-mers + Wavelet': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_One Hot + Wavelet': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'PS_One Hot': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_K-mers': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_FFT': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_Wavelet': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_K-mers + FFT': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_One Hot + FFT': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_K-mers + Wavelet': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'},
        'PS_One Hot + Wavelet': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
    },
    'Random Forest': {
        'AS_One Hot': {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 100},
        'AS_K-mers': {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 100},
        'AS_FFT': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100},
        'AS_Wavelet': {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 50},
        'AS_K-mers + FFT': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100},
        'AS_One Hot + FFT': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200},
        'AS_K-mers + Wavelet': {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 100},
        'AS_One Hot + Wavelet': {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 50},
        'PS_One Hot': {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 100},
        'PS_K-mers': {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 200},
        'PS_FFT': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200},
        'PS_Wavelet': {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200},
        'PS_K-mers + FFT': {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200},
        'PS_One Hot + FFT': {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 200},
        'PS_K-mers + Wavelet': {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 50},
        'PS_One Hot + Wavelet': {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 200}
    },
    'XGBoost': {
        'AS_One Hot': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS_K-mers': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS_FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS_Wavelet': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS_K-mers + FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS_One Hot + FFT': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS_K-mers + Wavelet': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS_One Hot + Wavelet': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS_One Hot': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS_K-mers': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS_FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS_Wavelet': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS_K-mers + FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS_One Hot + FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS_K-mers + Wavelet': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS_One Hot + Wavelet': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8}
    }
}

# ============================================================================
# DIVIDIR DATOS: TRAINING vs EVALUATION
# ============================================================================

log_with_time("Dividiendo datos en training y evaluation...")

# Filtrar datos para entrenamiento (is_training=True)
training_data = df[df['is_training'] == True].copy()
evaluation_data = df[df['is_training'] == False].copy()

log_with_time(f"Training data: {len(training_data)} muestras")
log_with_time(f"Evaluation data: {len(evaluation_data)} muestras")

# Guardar datos de evaluación para uso posterior
evaluation_data.to_parquet('results/evaluation_data.parquet', index=False)
log_with_time("Datos de evaluación guardados en results/evaluation_data.csv")

# ============================================================================
# ENTRENAR MODELOS
# ============================================================================

log_with_time("Iniciando entrenamiento de modelos...")

all_results = []
experiment_start_time = time.time()

for i, enc in enumerate(encodings):
    log_with_time(f"\n{'='*60}")
    log_with_time(f"Procesando encoding {i+1}/{len(encodings)}: {enc}")
    
    # Preparar datos para este encoding
    X = training_data[enc].tolist()
    y = training_data['clases_modelos'].values
    
    # División train/test para evaluación interna
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # ================== SVM ==================
    params_svm = parametros_optimos['SVM'][enc]
    results_svm = train_svm_improved(
        X_train, X_test, y_train, y_test, enc, reverse_map_genus,
        n_cores=N_CORES, random_state=RANDOM_STATE,
        **params_svm
    )
    all_results.append(results_svm)
    
    # Guardar resultados parciales
    save_training_results(results_svm, f"svm_{enc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # ================== Random Forest ==================
    params_rf = parametros_optimos['Random Forest'][enc]
    results_rf = train_random_forest_improved(
        X_train, X_test, y_train, y_test, enc, reverse_map_genus,
        n_cores=N_CORES, random_state=RANDOM_STATE,
        **params_rf
    )
    all_results.append(results_rf)
    
    # Guardar resultados parciales
    save_training_results(results_rf, f"rf_{enc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # ================== XGBoost ==================
    params_xgb = parametros_optimos['XGBoost'][enc]
    results_xgb = train_xgboost_improved(
        X_train, X_test, y_train, y_test, enc, reverse_map_genus,
        n_cores=N_CORES, random_state=RANDOM_STATE,
        **params_xgb
    )
    all_results.append(results_xgb)
    
    # Guardar resultados parciales
    save_training_results(results_xgb, f"xgb_{enc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    log_with_time(f"Completado encoding: {enc}")

# ============================================================================
# GUARDAR RESULTADOS FINALES
# ============================================================================

experiment_end_time = time.time()
total_experiment_time = experiment_end_time - experiment_start_time

log_with_time(f"\n{'='*60}")
log_with_time("Guardando resultados finales...")

# Guardar todos los resultados
final_results = {
    'experiment_info': {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_experiment_time,
        'total_time_minutes': total_experiment_time / 60,
        'n_encodings': len(encodings),
        'n_algorithms': 3,
        'total_models': len(all_results),
        'training_samples': len(training_data),
        'evaluation_samples': len(evaluation_data),
        'config': {
            'n_cores': N_CORES,
            'random_state': RANDOM_STATE,
            'test_size': TEST_SIZE
        }
    },
    'results': all_results,
    'encodings': encodings,
    'class_mapping': reverse_map_genus
}

# Guardar con timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_results_file = f"results/training_results/final_results_{timestamp}.json"

with open(final_results_file, 'w') as f:
    json.dump(final_results, f, indent=2)

log_with_time(f"Resultados finales guardados: {final_results_file}")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

log_with_time(f"\n{'='*60}")
log_with_time("ENTRENAMIENTO COMPLETADO")
log_with_time(f"{'='*60}")
log_with_time(f"Tiempo total: {total_experiment_time/60:.2f} minutos")
log_with_time(f"Encodings procesados: {len(encodings)}")
log_with_time(f"Modelos entrenados: {len(all_results)}")
log_with_time(f"Datos de entrenamiento: {len(training_data)} muestras")
log_with_time(f"Datos de evaluación: {len(evaluation_data)} muestras")
log_with_time(f"Resultados guardados en: {final_results_file}")
log_with_time(f"{'='*60}")

# Mostrar top 5 modelos por accuracy
log_with_time("\nTOP 5 MODELOS POR ACCURACY:")
sorted_results = sorted(all_results, key=lambda x: x['metrics']['accuracy'], reverse=True)
for i, result in enumerate(sorted_results[:5]):
    log_with_time(f"{i+1}. {result['model_name']} + {result['encoding']}: {result['metrics']['accuracy']:.4f}")

log_with_time("\nListo para evaluación y visualización!")