
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from xgboost import XGBClassifier

# Librerías para encoding
import pywt
from scipy.fft import fft
from itertools import product

# Configurar pandas
pd.options.mode.chained_assignment = None

# Variables compartidas
N_CORES = 1
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Crear directorios necesarios
os.makedirs("modelos/SVM", exist_ok=True)
os.makedirs("modelos/RandomForest", exist_ok=True) 
os.makedirs("modelos/XGBoost", exist_ok=True)
os.makedirs("results/training_results", exist_ok=True)
os.makedirs("results/gridsearch_results", exist_ok=True)


def log_with_time(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def save_model_safely(model, model_name, encoding_name, algorithm, metrics, params):
    
    # Crear directorio si no existe
    model_dir = f"modelos/{algorithm}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Nombre del archivo
    filename = f"{model_dir}/{encoding_name}.joblib"
    metadata_file = f"{model_dir}/{encoding_name}_metadata.json"
    
    try:
        # Guardar modelo
        joblib.dump(model, filename)
        print(f"\t\tModelo guardado: {filename}")
        
        # Guardar metadata
        metadata = {
            'model_name': model_name,
            'encoding': encoding_name,
            'algorithm': algorithm,
            'accuracy': float(metrics['accuracy']),
            'f1_score_weighted': float(metrics['f1_score_weighted']),
            'f1_score_macro': float(metrics['f1_score_macro']),
            'precision_macro': float(metrics['precision_macro']),
            'parameters': params,
            'model_file': filename
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
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
        print(f"\t\tResultados guardados: {filepath}")
        return filepath
    except Exception as e:
        log_with_time(f"Error guardando resultados: {e}")
        return None

def save_gridsearch_results(results, algorithm, encoding_name):
    """Guarda resultados detallados de grid search"""
    filepath = f"results/gridsearch_results/{algorithm}_{encoding_name}.json"
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\t\tResultados de grid search guardados: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error guardando resultados de grid search: {e}")
        return None


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



def calculate_unified_metrics(y_true, y_pred, y_proba=None, class_mapping=None):
    # Métricas globales - UNIFICADAS
    global_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_score_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'n_samples': len(y_true),
        'n_classes': len(np.unique(y_true))
    }
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Métricas por clase
    unique_classes = np.unique(y_true)
    class_metrics = {
        'sensitivity_per_class': {},
        'specificity_per_class': {},
        'precision_per_class': {},
        'recall_per_class': {}
    }
    
    # Si no hay mapeo de clases, crear uno genérico
    if class_mapping is None:
        class_mapping = {i: f"Class_{i}" for i in unique_classes}
    
    for i, class_id in enumerate(unique_classes):
        class_name = class_mapping.get(class_id, f"Class_{class_id}")
        
        # Calcular TP, FP, FN, TN
        TP = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
        FP = np.sum(cm[:, i]) - TP if i < cm.shape[1] else 0
        FN = np.sum(cm[i, :]) - TP if i < cm.shape[0] else 0
        TN = np.sum(cm) - (TP + FP + FN)
        
        # Métricas por clase
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics['sensitivity_per_class'][class_name] = recall
        class_metrics['specificity_per_class'][class_name] = specificity
        class_metrics['precision_per_class'][class_name] = precision
        class_metrics['recall_per_class'][class_name] = recall
    
    # ROC AUC por clase si hay probabilidades
    roc_auc_scores = {}
    if y_proba is not None:
        try:
            for i, class_id in enumerate(unique_classes):
                if i < y_proba.shape[1]:
                    class_name = class_mapping.get(class_id, f"Class_{class_id}")
                    y_true_binary = (y_true == class_id).astype(int)
                    
                    if len(np.unique(y_true_binary)) > 1:
                        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                        roc_auc_scores[class_name] = auc(fpr, tpr)
        except Exception as e:
            print(f"Error calculando ROC AUC: {e}")
    
    return {
        'global_metrics': global_metrics,
        'class_metrics': class_metrics,
        'confusion_matrix': cm.tolist(),
        'roc_auc_scores': roc_auc_scores,
        # Agregar métricas individuales para compatibilidad
        'accuracy': global_metrics['accuracy'],
        'f1_score_weighted': global_metrics['f1_score_weighted'],
        'f1_score_macro': global_metrics['f1_score_macro'],
        'precision_weighted': global_metrics['precision_weighted'],
        'precision_macro': global_metrics['precision_macro'],
        'recall_weighted': global_metrics['recall_weighted'],
        'recall_macro': global_metrics['recall_macro'],
        'f1_weighted': global_metrics['f1_weighted'],
        'f1_macro': global_metrics['f1_macro'],
        'f1_micro': global_metrics['f1_micro'],
        'n_samples': global_metrics['n_samples'],
        'n_classes': global_metrics['n_classes']
    }

def calculate_metrics(y_test, y_pred, y_score, classes):
    """
    Función para el código de entrenamiento - ahora incluye TODAS las métricas
    """
    # Crear mapeo de clases
    class_mapping = {i: classes[i] for i in range(len(classes))}
    
    # Usar la función unificada
    unified_results = calculate_unified_metrics(y_test, y_pred, y_score, class_mapping)
    
    # Retornar métricas principales
    metrics = {
        'accuracy': unified_results['accuracy'],
        'f1_score_weighted': unified_results['f1_score_weighted'],
        'f1_score_macro': unified_results['f1_score_macro'],
        'precision_weighted': unified_results['precision_weighted'],
        'precision_macro': unified_results['precision_macro'],
        'recall_weighted': unified_results['recall_weighted'],
        'recall_macro': unified_results['recall_macro']       
    }
    
    # Agregar las métricas por clase
    metrics.update(unified_results['class_metrics'])
    
    return metrics

def evaluate_model_simple(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    
    # Obtener las clases numéricas únicas
    classes_num = np.unique(y_test)
    # Mapear las clases numéricas a nombres reales
    classes = [class_names[i] for i in classes_num]
    
    # Usar la función unificada
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


def train_svm(X_train, X_test, y_train, y_test, encoding_name, class_names, 
              do_gridsearch=False, kernel='rbf', C=1.0, gamma='scale', n_cores=1):
    
    start_time = time.time()
    
    if do_gridsearch:
        print(f"\t\tEjecutando Grid Search para SVM...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        grid_search = GridSearchCV(
            SVC(probability=True),
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=n_cores,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        
        # Guardar resultados detallados de grid search
        gridsearch_results = {
            'algorithm': 'SVM',
            'encoding': encoding_name,
            'best_params': best_params,
            'best_score': cv_score,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'params': grid_search.cv_results_['params']
            }
        }
        save_gridsearch_results(gridsearch_results, 'SVM', encoding_name)
        
    else:
        model = SVC(probability=True, kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train, y_train)
        best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}
        cv_score = None
    
    training_time = time.time() - start_time
    
    # Evaluar modelo
    metrics, evaluation_data = evaluate_model_simple(model, X_test, y_test, class_names)
    
    print(f"\tSVM {encoding_name} - Accuracy: {metrics['accuracy']:.4f}, "
          f"F1: {metrics['f1_score_weighted']:.4f}, "
          f"Precision Macro: {metrics['precision_macro']:.4f}, "
          f"Tiempo: {training_time:.2f}s")
    
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
        'cv_score': cv_score,
        'training_time': training_time,
        'model_file': model_file,
        'metadata_file': metadata_file,
        'evaluation_data': evaluation_data
    }
    
    return results


def train_random_forest(X_train, X_test, y_train, y_test, encoding_name, class_names,
                        do_gridsearch=False, n_estimators=100, max_depth=None, 
                        min_samples_split=2, n_cores=1, random_state=42):
    
    start_time = time.time()
    
    if do_gridsearch:
        print(f"\t\tEjecutando Grid Search para Random Forest...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=n_cores,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        
        # Guardar resultados detallados de grid search
        gridsearch_results = {
            'algorithm': 'RandomForest',
            'encoding': encoding_name,
            'best_params': best_params,
            'best_score': cv_score,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'params': grid_search.cv_results_['params']
            }
        }
        save_gridsearch_results(gridsearch_results, 'RandomForest', encoding_name)
        
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=1,
            verbose=0
        )
        model.fit(X_train, y_train)
        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split
        }
        cv_score = None
    
    training_time = time.time() - start_time
    
    # Evaluar modelo
    metrics, evaluation_data = evaluate_model_simple(model, X_test, y_test, class_names)
    
    print(f"\tRandom Forest {encoding_name} - Accuracy: {metrics['accuracy']:.4f}, "
          f"F1: {metrics['f1_score_weighted']:.4f}, "
          f"Precision Macro: {metrics['precision_macro']:.4f}, "
          f"Tiempo: {training_time:.2f}s")
    
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
        'cv_score': cv_score,
        'training_time': training_time,
        'model_file': model_file,
        'metadata_file': metadata_file,
        'evaluation_data': evaluation_data
    }
    
    return results


def train_xgboost(X_train, X_test, y_train, y_test, encoding_name, class_names,
                  do_gridsearch=False, n_estimators=100, max_depth=6, learning_rate=0.3, 
                  subsample=0.8, colsample_bytree=0.8, n_cores=1, random_state=42):
    
    start_time = time.time()
    
    if do_gridsearch:
        print(f"\t\tEjecutando Grid Search para XGBoost...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.3],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        # Usar GridSearchCV
        grid_search = GridSearchCV(
            XGBClassifier(
                eval_metric='mlogloss',
                tree_method='hist',
                grow_policy='lossguide',
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
                use_label_encoder=False
            ),
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=n_cores,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train, verbose=False)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        
        # Guardar resultados detallados de grid search
        gridsearch_results = {
            'algorithm': 'XGBoost',
            'encoding': encoding_name,
            'best_params': best_params,
            'best_score': cv_score,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'params': grid_search.cv_results_['params']
            }
        }
        save_gridsearch_results(gridsearch_results, 'XGBoost', encoding_name)
        
    else:
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
            n_jobs=1,
            verbosity=0,
            use_label_encoder=False
        )
        model.fit(X_train, y_train, verbose=False)
        
        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        }
        cv_score = None
    
    training_time = time.time() - start_time
    
    # Evaluar modelo
    metrics, evaluation_data = evaluate_model_simple(model, X_test, y_test, class_names)
    
    print(f"\tXGBoost {encoding_name} - Accuracy: {metrics['accuracy']:.4f}, "
          f"F1: {metrics['f1_score_weighted']:.4f}, "
          f"Precision Macro: {metrics['precision_macro']:.4f}, "
          f"Tiempo: {training_time:.2f}s")
    
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
        'cv_score': cv_score,
        'training_time': training_time,
        'model_file': model_file,
        'metadata_file': metadata_file,
        'evaluation_data': evaluation_data
    }
    
    return results


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

print('Cargando datos...')

# Cargar datos
df = pd.read_csv('datos/datos_filtrados_sin_encoding.csv')
df=df[df['purpose']!=2]

print("Datos listos")

# Padding de secuencias
maxlen = max([len(i) for i in df['original_sequence']]) 
df['padded_sequences'] = pad_sequences(df['original_sequence'], maxlen)

# Calcular longitudes
df['len_ps'] = [len(i) for i in df['padded_sequences']]
df['len_as'] = [len(i) for i in df['aligned_sequence']]

print("Codificando secuencias")

# # Aligned Sequences (AS)
print('AS_One Hot...')
df['AS_One Hot'] = one_hot(df['aligned_sequence'].values, len(df['aligned_sequence'][0]))
print('AS_K-mers...')
df['AS_K-mers'] = k_mers(df['aligned_sequence'].values)
print('AS_FFT...')
df['AS_FFT'] = fourier(df['aligned_sequence'].values)
print('AS_Wavelet...')
df['AS_Wavelet'] = wavelet(df['aligned_sequence'].values)
print('AS_K-mers + FFT...')
df['AS_K-mers + FFT'] = fourier(df['AS_K-mers'].values, False)
print('AS_One Hot + FFT...')
df['AS_One Hot + FFT'] = fourier(df['AS_One Hot'].values, False)
print('AS_K-mers + Wavelet...')
df['AS_K-mers + Wavelet'] = wavelet(df['AS_K-mers'].values, True)
print('AS_One Hot + Wavelet...')
df['AS_One Hot + Wavelet'] = wavelet(df['AS_One Hot'].values, True)

# # Padded Sequences (PS)
print('PS_One Hot...')
df['PS_One Hot'] = one_hot(df['padded_sequences'].values, len(df['padded_sequences'][0]))
print('PS_K-mers...')
df['PS_K-mers'] = k_mers(df['padded_sequences'].values)
print('PS_FFT...')
df['PS_FFT'] = fourier(df['padded_sequences'].values)
print('PS_Wavelet...')
df['PS_Wavelet'] = wavelet(df['padded_sequences'].values)
print('PS_K-mers + FFT...')
df['PS_K-mers + FFT'] = fourier(df['PS_K-mers'].values, False)
print('PS_One Hot + FFT...')
df['PS_One Hot + FFT'] = fourier(df['PS_One Hot'].values, False)
print('PS_K-mers + Wavelet...')
df['PS_K-mers + Wavelet'] = wavelet(df['PS_K-mers'].values, True)
print('PS_One Hot + Wavelet...')
df['PS_One Hot + Wavelet'] = wavelet(df['PS_One Hot'].values, True)

print("Secuencias codificadas")

# Cargar mapeo de clases
mapeo_df = pd.read_csv('datos/mapeo_clases.csv')
df = df.merge(mapeo_df.rename(columns={'model_class': 'clases_modelos'}), 
              on='genus', how='left')

# Crear mapeo inverso para nombres de clases
tempdf = df[['genus', 'clases_modelos']].drop_duplicates()
reverse_map_genus = {v2: v1 for v1, v2 in tempdf.values}



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

parametros_optimos = {
    'SVM': {
        'AS_FFT': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_K-mers': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_K-mers + FFT': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_K-mers + Wavelet': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_One Hot': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_One Hot + FFT': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_K-mers + Wavelet': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_One Hot + Wavelet': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS_Wavelet': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'},
        'PS_FFT': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_K-mers': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_K-mers + FFT': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_K-mers + Wavelet': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_One Hot': {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'},
        'PS_One Hot + FFT': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS_One Hot + Wavelet': {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'},
        'PS_Wavelet': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
    },
    'Random Forest': {
        'AS_FFT': {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100},
        'AS_K-mers': {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 50},
        'AS_K-mers + FFT': {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 200},
        'AS_K-mers + Wavelet': {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 100},
        'AS_One Hot': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 50},
        'AS_One Hot + FFT': {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100},
        'AS_One Hot + Wavelet': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200},
        'AS_Wavelet': {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 50},
        'PS_FFT': {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 200},
        'PS_K-mers': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200},
        'PS_K-mers + FFT': {'max_depth': 30, 'min_samples_split': 10, 'n_estimators': 200},
        'PS_K-mers + Wavelet': {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 100},
        'PS_One Hot': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200},
        'PS_One Hot + FFT': {'max_depth': 30, 'min_samples_split': 10, 'n_estimators': 200},
        'PS_One Hot + Wavelet': {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 200},
        'PS_Wavelet': {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 100}
    },
    'XGBoost':{
        'AS_FFT':{'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100, 'subsample': 0.7},
        'AS_K-mers + FFT':{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8},
        'AS_K-mers + Wavelet':{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.7},
        'AS_K-mers':{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.9},
        'AS_One Hot + FFT':{'colsample_bytree': 0.8, 'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8},
        'AS_One Hot + Wavelet':{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8},
        'AS_One Hot':{'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.7},
        'AS_Wavelet':{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.7},
        'PS_FFT':{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.9},
        'PS_K-mers + FFT':{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100, 'subsample': 0.7},
        'PS_K-mers + Wavelet':{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 50, 'subsample': 0.7},
        'PS_K-mers':{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100, 'subsample': 0.8},
        'PS_One Hot + FFT':{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8},
        'PS_One Hot + Wavelet':{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.9},
        'PS_One Hot':{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8},
        'PS_Wavelet':{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100, 'subsample': 0.8}
    }
}


# Filtrar datos para entrenamiento (is_training=True)
training_data = df[df['purpose'] == 0].copy()
evaluation_data = df[df['purpose'] == 1].copy()

print(f"Training data: {len(training_data)} muestras")
print(f"Evaluation data: {len(evaluation_data)} muestras")


# Guardar datos de evaluación para uso posterior
test_data_path='datos/evaluation_data.parquet'
if os.path.exists(test_data_path):
    print(f"datos de testeo guardados en: {test_data_path}")
else:
    evaluation_data.to_parquet(test_data_path, index=False)
    print(f"datos de testeo guardados en: {test_data_path}")


print("Iniciando entrenamiento...")

all_results = []
experiment_start_time = time.time()

for i, enc in enumerate(encodings):
    print(f"\n{'='*60}")
    print(f"\tProcesando encoding {i+1}/{len(encodings)}: {enc}")
    
    # Preparar datos para este encoding
    X = training_data[enc].tolist()
    y = training_data['clases_modelos'].values
    
    # División train/test para evaluación interna
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # ================== SVM ==================
    params_svm = parametros_optimos['SVM'][enc]
    results_svm = train_svm(
        X_train, X_test, y_train, y_test, enc, reverse_map_genus, do_gridsearch=True,
        **params_svm
    )
    all_results.append(results_svm)
    
    # Guardar resultados parciales
    save_training_results(results_svm, f"svm_{enc}.json")
    
    # ================== Random Forest ==================
    params_rf = parametros_optimos['Random Forest'][enc]
    results_rf = train_random_forest(
        X_train, X_test, y_train, y_test, enc, reverse_map_genus,
        n_cores=N_CORES, random_state=RANDOM_STATE,do_gridsearch=True,
        **params_rf
    )
    all_results.append(results_rf)

    # Guardar resultados parciales
    save_training_results(results_rf, f"rf_{enc}.json")

    # ================== XGBoost ==================
    params_xgb = parametros_optimos['XGBoost'][enc]
    results_xgb = train_xgboost(
        X_train, X_test, y_train, y_test, enc, reverse_map_genus,
        n_cores=N_CORES, random_state=RANDOM_STATE,do_gridsearch=True,
        **params_xgb
    )
    all_results.append(results_xgb)
    
    # Guardar resultados parciales
    save_training_results(results_xgb, f"xgb_{enc}.json")


experiment_end_time = time.time()
total_experiment_time = experiment_end_time - experiment_start_time



print("Guardando resultados...")

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

final_results_file = f"results/training_results/final_results.json"

with open(final_results_file, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"Resultados finales guardados: {final_results_file}")

# RESUMEN FINAL
log_with_time(f"\n{'='*60}")
log_with_time("ENTRENAMIENTO COMPLETADO")
log_with_time(f"{'='*60}")
log_with_time(f"Tiempo total: {total_experiment_time/60:.2f} minutos")
log_with_time(f"Encodings procesados: {len(encodings)}")
log_with_time(f"Modelos entrenados: {len(all_results)}")
log_with_time(f"Datos de entrenamiento: {len(training_data)} muestras")
log_with_time(f"Datos de evaluación: {len(evaluation_data)} muestras")
log_with_time(f"Resultados guardados en: {final_results_file}")


