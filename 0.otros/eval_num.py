#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUACIÓN NUMÉRICA COMPLETA DE MODELOS
======================================

Script para evaluación numérica detallada de modelos con cálculo exhaustivo
de métricas, análisis de tiempos y guardado estructurado de resultados.

Uso: python evaluation_numerical.py
"""

import os
import json
import joblib
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime
from glob import glob
import warnings
import sys
import gc
from collections import defaultdict
import logging
import re
warnings.filterwarnings('ignore')

# Librerías de machine learning
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("EVALUACIÓN NUMÉRICA COMPLETA DE MODELOS")
print("=" * 50)

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

class EvaluationConfig:
    """Configuración centralizada para la evaluación"""
    def __init__(self):
        # Configuraciones principales
        self.save_individual_predictions = True
        self.save_detailed_reports = True
        self.calculate_class_metrics = True
        self.calculate_timing_metrics = True
        self.calculate_statistical_analysis = True
        
        # Configuraciones de archivos - MODIFICADO PARA TU ESTRUCTURA
        self.results_dir = "results/numerical_evaluation"
        self.models_dir = "models"  # Cambiado de results/training_results
        self.data_file = "results/evaluation_data.parquet"
        
        # Archivos alternativos para datos de evaluación
        self.alternative_data_files = [
            "eval.csv",
            "data/evaluation_data.csv",
            "datos/evaluation_data.csv"
        ]
        
        # Métricas a calcular
        self.metrics_to_calculate = [
            'accuracy', 'f1_weighted', 'f1_macro', 'f1_micro',
            'precision_weighted', 'precision_macro', 'precision_micro',
            'recall_weighted', 'recall_macro', 'recall_micro'
        ]
        
        # Configuraciones de memoria
        self.memory_efficient = True
        self.max_models_in_memory = 5

config = EvaluationConfig()

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def log_with_time(message, level="INFO"):
    """Log con timestamp"""
    if level == "INFO":
        logger.info(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)

def ensure_dir(file_path):
    """Asegura que el directorio existe"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def get_deep_size(obj, seen=None):
    """Calcula tamaño real de objetos anidados en MB"""
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
    
    return size / (1024 * 1024)  # Convertir a MB

# ============================================================================
# DESCUBRIMIENTO DE MODELOS - NUEVA FUNCIÓN
# ============================================================================

def discover_models(models_dir="models"):
    """Descubre modelos en la estructura de directorios actual"""
    log_with_time(f"Descubriendo modelos en '{models_dir}'")
    
    if not os.path.exists(models_dir):
        log_with_time(f"El directorio {models_dir} no existe", "ERROR")
        return []
    
    discovered_models = []
    
    # Buscar archivos .joblib y .pkl recursivamente
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(('.joblib', '.pkl')):
                full_path = os.path.join(root, file)
                
                # Extraer información del path y nombre del archivo
                rel_path = os.path.relpath(full_path, models_dir)
                path_parts = rel_path.split(os.sep)
                
                # Inferir algoritmo del directorio padre
                algorithm = path_parts[0] if len(path_parts) > 1 else "Unknown"
                
                # Extraer encoding del nombre del archivo
                filename_no_ext = os.path.splitext(file)[0]
                
                # Patrones para extraer encoding
                # Formato esperado: [ALGO_]ENCODING_TIMESTAMP.ext
                encoding = None
                
                # Intentar diferentes patrones
                patterns = [
                    r'^(?:' + re.escape(algorithm) + '_)?(.+?)_\d{8}_\d{6}$',  # ALGO_ENCODING_TIMESTAMP
                    r'^(.+?)_\d{8}_\d{6}$',  # ENCODING_TIMESTAMP
                    r'^(?:' + re.escape(algorithm) + '_)?(.+)$'  # Solo ENCODING o ALGO_ENCODING
                ]
                
                for pattern in patterns:
                    match = re.match(pattern, filename_no_ext, re.IGNORECASE)
                    if match:
                        encoding = match.group(1)
                        break
                
                if not encoding:
                    # Fallback: usar todo el nombre menos timestamp si existe
                    parts = filename_no_ext.split('_')
                    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
                        encoding = '_'.join(parts[:-2])
                    else:
                        encoding = filename_no_ext
                
                # Limpiar encoding
                if encoding.startswith(f"{algorithm}_"):
                    encoding = encoding[len(algorithm)+1:]
                
                model_info = {
                    'model_name': algorithm,
                    'encoding': encoding,
                    'model_file': full_path,
                    'filename': file,
                    'relative_path': rel_path,
                    'metrics': {}  # Vacío ya que no tenemos datos de entrenamiento
                }
                
                discovered_models.append(model_info)
                log_with_time(f"  Encontrado: {algorithm} - {encoding} ({file})")
    
    log_with_time(f"Total de modelos descubiertos: {len(discovered_models)}")
    return discovered_models

# ============================================================================
# DIAGNÓSTICO DE MODELOS - MODIFICADO
# ============================================================================

def diagnose_model_files(models_dir="models"):
    """Diagnostica archivos de modelos para detectar problemas"""
    log_with_time(f"Diagnosticando archivos de modelos en '{models_dir}'")
    
    if not os.path.exists(models_dir):
        log_with_time(f"El directorio {models_dir} no existe", "ERROR")
        return None
    
    # Buscar archivos de modelos recursivamente
    model_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(('.pkl', '.joblib')):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        log_with_time("No se encontraron archivos de modelos", "WARNING")
        return None
    
    log_with_time(f"Encontrados {len(model_files)} archivos de modelos")
    
    diagnosis = {
        'total_files': len(model_files),
        'valid_files': [],
        'corrupted_files': [],
        'empty_files': [],
        'file_details': {}
    }
    
    for filepath in model_files:
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        
        file_info = {
            'path': filepath,
            'size_bytes': file_size,
            'size_mb': file_size / (1024 * 1024),
            'status': 'unknown'
        }
        
        if file_size == 0:
            file_info['status'] = 'empty'
            diagnosis['empty_files'].append(filename)
        else:
            try:
                # Verificar si se puede cargar
                with open(filepath, 'rb') as f:
                    first_bytes = f.read(20)
                    f.seek(0)
                    
                    # Verificar cabecera pickle/joblib
                    if first_bytes.startswith(b'\x80') or b'sklearn' in first_bytes:
                        # Intentar cargar
                        try:
                            if filepath.endswith('.pkl'):
                                model = pickle.load(f)
                            else:
                                model = joblib.load(filepath)
                            
                            file_info['status'] = 'valid'
                            file_info['model_type'] = type(model).__name__
                            diagnosis['valid_files'].append(filename)
                            
                        except Exception as e:
                            file_info['status'] = 'corrupted'
                            file_info['error'] = str(e)
                            diagnosis['corrupted_files'].append(filename)
                    else:
                        file_info['status'] = 'invalid_format'
                        diagnosis['corrupted_files'].append(filename)
                        
            except Exception as e:
                file_info['status'] = 'error'
                file_info['error'] = str(e)
                diagnosis['corrupted_files'].append(filename)
        
        diagnosis['file_details'][filename] = file_info
    
    log_with_time(f"Archivos válidos: {len(diagnosis['valid_files'])}")
    log_with_time(f"Archivos corruptos: {len(diagnosis['corrupted_files'])}")
    log_with_time(f"Archivos vacíos: {len(diagnosis['empty_files'])}")
    
    return diagnosis

# ============================================================================
# CARGA DE DATOS - MODIFICADO
# ============================================================================

def load_evaluation_data():
    """Carga datos de evaluación desde múltiples fuentes posibles"""
    
    # Intentar cargar desde archivo principal
    if os.path.exists(config.data_file):
        try:
            log_with_time(f"Cargando datos desde {config.data_file}")
            data = pd.read_parquet(config.data_file)
            log_with_time(f"Datos cargados exitosamente: {len(data)} filas")
            return data
        except Exception as e:
            log_with_time(f"Error cargando {config.data_file}: {e}", "WARNING")
    
    # Intentar archivos alternativos
    for alt_file in config.alternative_data_files:
        if os.path.exists(alt_file):
            try:
                log_with_time(f"Intentando cargar desde {alt_file}")
                if alt_file.endswith('.csv'):
                    data = pd.read_csv(alt_file)
                elif alt_file.endswith('.parquet'):
                    data = pd.read_parquet(alt_file)
                else:
                    continue
                
                log_with_time(f"Datos cargados exitosamente: {len(data)} filas")
                return data
            except Exception as e:
                log_with_time(f"Error cargando {alt_file}: {e}", "WARNING")
    
    log_with_time("No se pudo cargar ningún archivo de datos", "ERROR")
    return None

def create_class_mapping(evaluation_data):
    """Crea un mapeo de clases basado en los datos disponibles"""
    
    if 'clases_modelos' in evaluation_data.columns:
        # Ya existe el mapeo numérico
        if 'genus' in evaluation_data.columns:
            # Crear mapeo desde genus
            mapping_data = evaluation_data[['genus', 'clases_modelos']].drop_duplicates()
            class_mapping = dict(zip(mapping_data['clases_modelos'], mapping_data['genus']))
        else:
            # Crear mapeo genérico
            unique_classes = evaluation_data['clases_modelos'].unique()
            class_mapping = {i: f"Class_{i}" for i in unique_classes}
    elif 'genus' in evaluation_data.columns:
        # Crear mapeo numérico
        unique_genera = evaluation_data['genus'].unique()
        genus_to_id = {genus: i for i, genus in enumerate(sorted(unique_genera))}
        evaluation_data['clases_modelos'] = evaluation_data['genus'].map(genus_to_id)
        class_mapping = {i: genus for genus, i in genus_to_id.items()}
    else:
        log_with_time("No se encontraron columnas de clase válidas", "ERROR")
        return None, evaluation_data
    
    log_with_time(f"Mapeo de clases creado: {len(class_mapping)} clases")
    return class_mapping, evaluation_data

# ============================================================================
# CÁLCULO DE MÉTRICAS DETALLADAS
# ============================================================================

def calculate_detailed_metrics(y_true, y_pred, y_proba, class_names, timing_info=None):
    """Calcula métricas detalladas incluyendo por clase"""
    
    metrics = {}
    
    # Métricas globales
    metrics['global'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'n_samples': len(y_true),
        'n_classes': len(np.unique(y_true))
    }
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Métricas por clase
    unique_classes = np.unique(y_true)
    metrics['per_class'] = {}
    
    # Calcular métricas por clase usando sklearn
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    for i, class_id in enumerate(unique_classes):
        class_name = class_names.get(class_id, f"Class_{class_id}")
        class_str = str(class_id)
        
        if class_str in class_report:
            class_metrics = class_report[class_str]
            
            # Calcular especificidad manualmente
            TP = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
            FP = np.sum(cm[:, i]) - TP if i < cm.shape[1] else 0
            FN = np.sum(cm[i, :]) - TP if i < cm.shape[0] else 0
            TN = np.sum(cm) - (TP + FP + FN)
            
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            
            metrics['per_class'][class_name] = {
                'class_id': int(class_id),
                'precision': class_metrics['precision'],
                'recall': class_metrics['recall'],
                'f1_score': class_metrics['f1-score'],
                'support': int(class_metrics['support']),
                'sensitivity': class_metrics['recall'],  # sensitivity = recall
                'specificity': specificity,
                'true_positives': int(TP),
                'false_positives': int(FP),
                'false_negatives': int(FN),
                'true_negatives': int(TN)
            }
    
    # ROC AUC por clase (si hay probabilidades)
    if y_proba is not None:
        metrics['roc_auc'] = {}
        try:
            for i, class_id in enumerate(unique_classes):
                if i < y_proba.shape[1]:
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    y_true_binary = (y_true == class_id).astype(int)
                    
                    if len(np.unique(y_true_binary)) > 1:  # Verificar que hay ambas clases
                        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        metrics['roc_auc'][class_name] = float(roc_auc)
        except Exception as e:
            log_with_time(f"Error calculando ROC AUC: {e}", "WARNING")
    
    # Información de timing si está disponible
    if timing_info:
        metrics['timing'] = timing_info
    
    return metrics

# ============================================================================
# EVALUACIÓN PRINCIPAL
# ============================================================================

def load_model_safely(model_path):
    """Carga un modelo de forma segura con manejo de errores"""
    try:
        start_time = time.time()
        
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = joblib.load(model_path)
        
        load_time = time.time() - start_time
        return model, load_time, None
        
    except Exception as e:
        return None, 0, str(e)

def evaluate_single_model(model_info, evaluation_data, class_mapping):
    """Evalúa un solo modelo de forma completa"""
    
    model_name = model_info['model_name']
    encoding = model_info['encoding']
    model_file = model_info['model_file']
    
    log_with_time(f"Evaluando {model_name} - {encoding}")
    
    # Inicializar resultado
    result = {
        'model_name': model_name,
        'encoding': encoding,
        'model_file': model_file,
        'success': False,
        'error': None,
        'test_metrics': model_info.get('metrics', {}),
        'eval_metrics': {},
        'class_metrics': {},
        'timing_info': {},
        'predictions': {},
        'efficiency_score': 0,
        'data_info': {}
    }
    
    try:
        # Verificar que el encoding existe en los datos
        if encoding not in evaluation_data.columns:
            raise ValueError(f"Encoding {encoding} no encontrado en datos de evaluación")
        
        # Cargar modelo
        log_with_time(f"  Cargando modelo...")
        model, load_time, load_error = load_model_safely(model_file)
        
        if model is None:
            raise ValueError(f"Error cargando modelo: {load_error}")
        
        result['timing_info']['model_load_time'] = load_time
        
        # Preparar datos
        log_with_time(f"  Preparando datos...")
        data_prep_start = time.time()
        
        X_eval = evaluation_data[encoding].tolist()
        y_eval = evaluation_data['clases_modelos'].values
        
        # Filtrar valores None/NaN
        valid_indices = [i for i, x in enumerate(X_eval) 
                        if x is not None and not (isinstance(x, (int, float)) and np.isnan(x))]
        
        if len(valid_indices) == 0:
            raise ValueError("No hay datos válidos para evaluación")
        
        X_eval_clean = [X_eval[i] for i in valid_indices]
        y_eval_clean = y_eval[valid_indices]
        
        # Convertir a numpy array
        X_eval_array = np.array(X_eval_clean)
        
        data_prep_time = time.time() - data_prep_start
        result['timing_info']['data_prep_time'] = data_prep_time
        
        # Información de datos
        result['data_info'] = {
            'original_samples': len(X_eval),
            'valid_samples': len(X_eval_clean),
            'feature_dimensions': X_eval_array.shape[1] if len(X_eval_array.shape) > 1 else 1,
            'classes_present': len(np.unique(y_eval_clean)),
            'data_reduction_ratio': len(X_eval_clean) / len(X_eval)
        }
        
        # Hacer predicciones
        log_with_time(f"  Realizando predicciones...")
        prediction_start = time.time()
        
        y_pred = model.predict(X_eval_array)
        
        # Verificar si el modelo tiene predict_proba
        y_proba = None
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_eval_array)
        except Exception as e:
            log_with_time(f"  Warning: No se pudieron obtener probabilidades: {e}", "WARNING")
        
        prediction_time = time.time() - prediction_start
        result['timing_info']['prediction_time'] = prediction_time
        result['timing_info']['total_time'] = load_time + data_prep_time + prediction_time
        
        # Calcular métricas detalladas
        log_with_time(f"  Calculando métricas...")
        metrics_start = time.time()
        
        detailed_metrics = calculate_detailed_metrics(
            y_eval_clean, y_pred, y_proba, class_mapping, result['timing_info']
        )
        
        metrics_time = time.time() - metrics_start
        result['timing_info']['metrics_calc_time'] = metrics_time
        
        # Almacenar resultados
        result['eval_metrics'] = detailed_metrics['global']
        result['class_metrics'] = detailed_metrics['per_class']
        result['confusion_matrix'] = detailed_metrics['confusion_matrix']
        
        if 'roc_auc' in detailed_metrics:
            result['roc_auc'] = detailed_metrics['roc_auc']
        
        # Predicciones detalladas
        if config.save_individual_predictions:
            result['predictions'] = {
                'y_true': y_eval_clean.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist() if y_proba is not None else None,
                'correct_predictions': (y_eval_clean == y_pred).tolist()
            }
        
        # Calcular efficiency score
        if result['timing_info']['total_time'] > 0:
            result['efficiency_score'] = result['eval_metrics']['accuracy'] / result['timing_info']['total_time']
        
        result['success'] = True
        
        log_with_time(f"  Completado - Accuracy: {result['eval_metrics']['accuracy']:.4f}")
        
        # Liberar memoria si está configurado
        if config.memory_efficient:
            del model, X_eval_array, y_pred
            if y_proba is not None:
                del y_proba
            gc.collect()
        
    except Exception as e:
        result['error'] = str(e)
        log_with_time(f"  Error: {e}", "ERROR")
    
    return result

# ============================================================================
# ANÁLISIS ESTADÍSTICO
# ============================================================================

def calculate_statistical_analysis(results_df):
    """Calcula análisis estadístico de los resultados"""
    
    analysis = {}
    
    # Estadísticas descriptivas por métrica
    numeric_columns = ['eval_accuracy', 'eval_f1_weighted', 'eval_f1_macro', 
                      'total_time', 'efficiency_score']
    
    available_columns = [col for col in numeric_columns if col in results_df.columns]
    
    if available_columns:
        analysis['descriptive_stats'] = {}
        for col in available_columns:
            analysis['descriptive_stats'][col] = {
                'mean': float(results_df[col].mean()),
                'std': float(results_df[col].std()),
                'min': float(results_df[col].min()),
                'max': float(results_df[col].max()),
                'median': float(results_df[col].median()),
                'q25': float(results_df[col].quantile(0.25)),
                'q75': float(results_df[col].quantile(0.75))
            }
    
    # Correlaciones entre métricas - CORREGIDO
    if len(available_columns) > 1:
        correlation_matrix = results_df[available_columns].corr()
        # Convertir matriz de correlación a formato JSON-serializable
        analysis['correlations'] = {}
        for col1 in correlation_matrix.columns:
            analysis['correlations'][str(col1)] = {}
            for col2 in correlation_matrix.columns:
                analysis['correlations'][str(col1)][str(col2)] = float(correlation_matrix.loc[col1, col2])
    
    # Rankings
    if 'eval_accuracy' in results_df.columns:
        analysis['rankings'] = {
            'by_accuracy': results_df.nlargest(10, 'eval_accuracy')[
                ['model_name', 'encoding', 'eval_accuracy']
            ].to_dict('records'),
            'by_efficiency': results_df.nlargest(10, 'efficiency_score')[
                ['model_name', 'encoding', 'efficiency_score']
            ].to_dict('records') if 'efficiency_score' in results_df.columns else []
        }
    
    # Análisis por algoritmo - CORREGIDO
    if 'model_name' in results_df.columns:
        algorithm_stats = results_df.groupby('model_name').agg({
            'eval_accuracy': ['mean', 'std', 'count'],
            'total_time': ['mean', 'std'] if 'total_time' in results_df.columns else ['count']
        }).round(4)
        
        # Convertir MultiIndex a formato JSON-serializable
        analysis['by_algorithm'] = {}
        for algorithm in algorithm_stats.index:
            analysis['by_algorithm'][str(algorithm)] = {}
            for col in algorithm_stats.columns:
                # col es una tupla (métrica, estadística)
                metric, stat = col
                key = f"{metric}_{stat}"
                analysis['by_algorithm'][str(algorithm)][key] = float(algorithm_stats.loc[algorithm, col])
    
    # Análisis por encoding - CORREGIDO
    if 'encoding' in results_df.columns:
        encoding_stats = results_df.groupby('encoding').agg({
            'eval_accuracy': ['mean', 'std', 'count'],
            'total_time': ['mean', 'std'] if 'total_time' in results_df.columns else ['count']
        }).round(4)
        
        # Convertir MultiIndex a formato JSON-serializable
        analysis['by_encoding'] = {}
        for encoding in encoding_stats.index:
            analysis['by_encoding'][str(encoding)] = {}
            for col in encoding_stats.columns:
                # col es una tupla (métrica, estadística)
                metric, stat = col
                key = f"{metric}_{stat}"
                analysis['by_encoding'][str(encoding)][key] = float(encoding_stats.loc[encoding, col])
    
    return analysis


def convert_numpy_and_types(obj):
    """Función mejorada para convertir tipos no serializables"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, tuple):
        # Convertir tuplas a listas
        return [convert_numpy_and_types(item) for item in obj]
    elif isinstance(obj, dict):
        # Asegurar que las claves sean strings
        return {str(k): convert_numpy_and_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_and_types(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif pd.isna(obj):
        return None
    return obj

# ============================================================================
# GUARDADO DE RESULTADOS
# ============================================================================

def save_detailed_results(all_results, statistical_analysis, timestamp):
    """Guarda todos los resultados en múltiples formatos"""
    
    # CORRECCIÓN: Crear directorio directamente
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Crear DataFrame principal
    main_data = []
    for result in all_results:
        if result['success']:
            row = {
                'model_name': result['model_name'],
                'encoding': result['encoding'],
                'eval_accuracy': result['eval_metrics'].get('accuracy', 0),
                'eval_f1_weighted': result['eval_metrics'].get('f1_weighted', 0),
                'eval_f1_macro': result['eval_metrics'].get('f1_macro', 0),
                'eval_precision_weighted': result['eval_metrics'].get('precision_weighted', 0),
                'eval_recall_weighted': result['eval_metrics'].get('recall_weighted', 0),
                'n_samples': result['eval_metrics'].get('n_samples', 0),
                'n_classes': result['eval_metrics'].get('n_classes', 0),
                'total_time': result['timing_info'].get('total_time', 0),
                'prediction_time': result['timing_info'].get('prediction_time', 0),
                'model_load_time': result['timing_info'].get('model_load_time', 0),
                'efficiency_score': result['efficiency_score'],
                'data_reduction_ratio': result['data_info'].get('data_reduction_ratio', 1.0),
                'model_file': result['model_file']
            }
            
            main_data.append(row)
    
    df_main = pd.DataFrame(main_data)
    
    if df_main.empty:
        log_with_time("No hay resultados exitosos para guardar", "WARNING")
        return
    
    # Guardar CSV principal
    csv_path = os.path.join(config.results_dir, f"evaluation_results_{timestamp}.csv")
    df_main.to_csv(csv_path, index=False)
    log_with_time(f"Resultados principales guardados: {csv_path}")
    
    # Guardar Excel con múltiples hojas
    excel_path = os.path.join(config.results_dir, f"evaluation_complete_{timestamp}.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Hoja principal
        df_main.to_excel(writer, sheet_name='Resumen', index=False)
        
        # Hoja con métricas por clase
        class_data = []
        for result in all_results:
            if result['success'] and result['class_metrics']:
                for class_name, metrics in result['class_metrics'].items():
                    row = {
                        'model_name': result['model_name'],
                        'encoding': result['encoding'],
                        'class_name': class_name,
                        'class_id': metrics.get('class_id', ''),
                        **metrics
                    }
                    class_data.append(row)
        
        if class_data:
            df_class = pd.DataFrame(class_data)
            df_class.to_excel(writer, sheet_name='Metricas_por_Clase', index=False)
        
        # Hoja con información de timing
        timing_data = []
        for result in all_results:
            if result['success'] and result['timing_info']:
                row = {
                    'model_name': result['model_name'],
                    'encoding': result['encoding'],
                    **result['timing_info']
                }
                timing_data.append(row)
        
        if timing_data:
            df_timing = pd.DataFrame(timing_data)
            df_timing.to_excel(writer, sheet_name='Tiempos_Ejecucion', index=False)
        
        # Hoja con análisis estadístico
        if statistical_analysis:
            # Convertir análisis estadístico a formato tabular
            stats_rows = []
            if 'descriptive_stats' in statistical_analysis:
                for metric, stats in statistical_analysis['descriptive_stats'].items():
                    for stat_name, value in stats.items():
                        stats_rows.append({
                            'metric': metric,
                            'statistic': stat_name,
                            'value': value
                        })
            
            if stats_rows:
                df_stats = pd.DataFrame(stats_rows)
                df_stats.to_excel(writer, sheet_name='Analisis_Estadistico', index=False)
    
    log_with_time(f"Excel completo guardado: {excel_path}")
    
    # Guardar resultados completos en JSON - CORREGIDO
    json_path = os.path.join(config.results_dir, f"evaluation_detailed_{timestamp}.json")
    
    # Preparar datos para JSON
    json_data = {
        'timestamp': timestamp,
        'config': vars(config),
        'results': all_results,
        'statistical_analysis': statistical_analysis,
        'summary': {
            'total_models': len(all_results),
            'successful_evaluations': len([r for r in all_results if r['success']]),
            'failed_evaluations': len([r for r in all_results if not r['success']])
        }
    }
    
    # Convertir todos los tipos no serializables
    json_data = convert_numpy_and_types(json_data)
    
    try:
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        log_with_time(f"JSON detallado guardado: {json_path}")
    except Exception as e:
        log_with_time(f"Error guardando JSON: {e}", "WARNING")
        # Intentar guardar versión simplificada
        simplified_data = {
            'timestamp': timestamp,
            'summary': json_data['summary'],
            'descriptive_stats': statistical_analysis.get('descriptive_stats', {}),
            'rankings': statistical_analysis.get('rankings', {})
        }
        
        try:
            json_simplified_path = os.path.join(config.results_dir, f"evaluation_summary_{timestamp}.json")
            with open(json_simplified_path, 'w') as f:
                json.dump(simplified_data, f, indent=2)
            log_with_time(f"JSON simplificado guardado: {json_simplified_path}")
            json_path = json_simplified_path
        except Exception as e2:
            log_with_time(f"Error guardando JSON simplificado: {e2}", "ERROR")
            json_path = None
    
    return {
        'csv_path': csv_path,
        'excel_path': excel_path,
        'json_path': json_path,
        'main_dataframe': df_main
    }
    
# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de evaluación numérica"""
    
    log_with_time("Iniciando evaluación numérica completa...")
    
    # Crear directorio de resultados
    os.makedirs(config.results_dir, exist_ok=True)
    # ========================================================================
    # DESCUBRIMIENTO Y VERIFICACIÓN DE MODELOS
    # ========================================================================
    log_with_time("Descubriendo modelos...")
    
    discovered_models = discover_models(config.models_dir)
    if not discovered_models:
        log_with_time("No se encontraron modelos", "ERROR")
        return False
    
    # Diagnosticar modelos
    diagnosis = diagnose_model_files(config.models_dir)
    if diagnosis is None or len(diagnosis['valid_files']) == 0:
        log_with_time("No se encontraron modelos válidos", "ERROR")
        return False
    
    log_with_time(f"Modelos válidos encontrados: {len(diagnosis['valid_files'])}")
    
    # ========================================================================
    # CARGAR DATOS
    # ========================================================================
    log_with_time("Cargando datos de evaluación...")
    
    evaluation_data = load_evaluation_data()
    if evaluation_data is None:
        return False
    
    # Crear mapeo de clases
    class_mapping, evaluation_data = create_class_mapping(evaluation_data)
    if class_mapping is None:
        return False
    
    log_with_time(f"Clases disponibles: {len(class_mapping)}")
    
    # ========================================================================
    # EVALUACIÓN DE MODELOS
    # ========================================================================
    log_with_time("Iniciando evaluación de modelos...")
    
    all_results = []
    successful_count = 0
    failed_count = 0
    
    total_models = len(discovered_models)
    
    for i, model_info in enumerate(discovered_models):
        log_with_time(f"Progreso: {i+1}/{total_models}")
        
        # Verificar que el archivo de modelo existe
        if not os.path.exists(model_info['model_file']):
            log_with_time(f"Archivo de modelo no encontrado: {model_info['model_file']}", "WARNING")
            failed_count += 1
            continue
        
        # Evaluar modelo
        result = evaluate_single_model(model_info, evaluation_data, class_mapping)
        all_results.append(result)
        
        if result['success']:
            successful_count += 1
        else:
            failed_count += 1
        
        # Liberar memoria cada ciertos modelos
        if config.memory_efficient and i % config.max_models_in_memory == 0:
            gc.collect()
    
    log_with_time(f"Evaluación completada: {successful_count} exitosos, {failed_count} fallidos")
    
    if successful_count == 0:
        log_with_time("No se evaluaron modelos exitosamente", "ERROR")
        return False
    
    # ========================================================================
    # ANÁLISIS ESTADÍSTICO
    # ========================================================================
    log_with_time("Calculando análisis estadístico...")
    
    # Crear DataFrame para análisis
    successful_results = [r for r in all_results if r['success']]
    
    analysis_data = []
    for result in successful_results:
        row = {
            'model_name': result['model_name'],
            'encoding': result['encoding'],
            'eval_accuracy': result['eval_metrics'].get('accuracy', 0),
            'eval_f1_weighted': result['eval_metrics'].get('f1_weighted', 0),
            'eval_f1_macro': result['eval_metrics'].get('f1_macro', 0),
            'total_time': result['timing_info'].get('total_time', 0),
            'efficiency_score': result['efficiency_score']
        }
        analysis_data.append(row)
    
    df_for_analysis = pd.DataFrame(analysis_data)
    statistical_analysis = calculate_statistical_analysis(df_for_analysis)
    
    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    log_with_time("Guardando resultados...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = save_detailed_results(all_results, statistical_analysis, timestamp)
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    log_with_time("Generando resumen final...")
    
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN NUMÉRICA COMPLETADA")
    print(f"{'='*60}")
    print(f"Modelos evaluados exitosamente: {successful_count}")
    print(f"Modelos fallidos: {failed_count}")
    print(f"Total de modelos procesados: {len(all_results)}")
    
    if saved_files and 'main_dataframe' in saved_files:
        df = saved_files['main_dataframe']
        if not df.empty:
            print(f"\nMEJORES RESULTADOS:")
            print(f"Mejor accuracy: {df['eval_accuracy'].max():.4f}")
            print(f"Accuracy promedio: {df['eval_accuracy'].mean():.4f}")
            print(f"Mejor eficiencia: {df['efficiency_score'].max():.4f}")
            
            best_model = df.loc[df['eval_accuracy'].idxmax()]
            print(f"\nMEJOR MODELO:")
            print(f"  Algoritmo: {best_model['model_name']}")
            print(f"  Encoding: {best_model['encoding']}")
            print(f"  Accuracy: {best_model['eval_accuracy']:.4f}")
            print(f"  Tiempo: {best_model['total_time']:.2f}s")
    
    print(f"\nARCHIVOS GENERADOS:")
    for key, path in saved_files.items():
        if key != 'main_dataframe':
            print(f"  {key}: {path}")
    
    print(f"{'='*60}")
    
    return True

# ============================================================================
# EJECUTAR SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("Iniciando evaluación numérica...")
    
    try:
        success = main()
        
        if success:
            print("\nEVALUACIÓN NUMÉRICA COMPLETADA CON ÉXITO!")
            exit(0)
        else:
            print("\nLA EVALUACIÓN NUMÉRICA FALLÓ")
            exit(1)
            
    except KeyboardInterrupt:
        print("\nEvaluación interrumpida por el usuario")
        exit(1)
        
    except Exception as e:
        print(f"\nERROR FATAL: {e}")
        import traceback
        traceback.print_exc()
        exit(1)