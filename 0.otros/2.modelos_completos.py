
# !pip install numpy pandas plotly scikit-learn xgboost lightgbm catboost PyWavelets nbformat matplotlib scipy


# Librerías básicas
import time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# Librerías de visualización
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Librerías de machine learning
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, auc, roc_curve

from xgboost import XGBClassifier

# # Funciones personalizadas
# from funciones.encodings import one_hot, k_mers, fourier, wavelet


df=pd.read_csv('datos/dmfne.csv')


df=df[['genus', 'se', 'sequence','gc_content']]


df=df.rename(columns={'se':'as'})


#padding de secuencias para igualar el largo de las mismas entre las más largas y las más cortas
def pad_sequences(sequences, maxlen):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            seq += 'N' * (maxlen - len(seq))  # Padding with 'N' (unknown nucleotide)
        else:
            seq = seq[:maxlen]  # Truncate if longer than maxlen
        padded_sequences.append(seq)
    return padded_sequences

maxlen = max([len(i) for i in df['sequence']]) # Adjust based on your specific case
df['ps'] = pad_sequences(df['sequence'], maxlen)


df['len_sequence']=[len(i) for i in df['sequence']]
df['len_ps']=[len(i) for i in df['ps']]
df['len_as']=[len(i) for i in df['as']]


df=df[['genus', 'gc_content', 'sequence', 'len_sequence', 'ps','len_ps','as','len_as']]


map_genus={j:i for i,j in enumerate(df['genus'].unique())}
df['clases_modelos']=df['genus'].map(map_genus)


import pywt
from scipy.fft import fft
from itertools import product

def fourier(sequences, is_str=True):
    if is_str:
        templist=[]
        for seq in sequences:
            num_seq=[ord(char) for char in seq]
            fft_seq=fft(num_seq)
            fft_seq=np.abs(fft_seq)
            # fft_seq=fft[1:len(fft_seq)//2]
            templist.append(fft_seq[1:len(fft_seq)//2])
        return templist
    else:
        templist=[]
        for seq in sequences:
            fft_seq=fft(seq)
            fft_seq=np.abs(fft_seq)
            # fft_seq=fft[1:len(fft_seq)//2]
            templist.append(fft_seq[1:len(fft_seq)//2])
        return templist

def generate_kmers_dict(k, unique_chars=set('ACGNT')):
    
    # Generar todas las posibles combinaciones
    kmers = product(unique_chars, repeat=k)
    
    # Crear el diccionario
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


df['AS-One Hot']=one_hot(df['as'].values, len(df['as'][0]))

df['AS-K-mers']=k_mers(df['as'].values)

df['AS-FFT']=fourier(df['as'].values)

df['AS-Wavelet']=wavelet(df['as'].values)

df['AS-K-mers + FFT']=fourier(df['AS-K-mers'].values, False)
df['AS-One Hot + FFT']=fourier(df['AS-One Hot'].values, False)

df['AS-K-mers + Wavelet']=wavelet(df['AS-K-mers'].values, True)
df['AS-One Hot + Wavelet']=wavelet(df['AS-One Hot'].values, True)


df['PS-One Hot']=one_hot(df['ps'].values, len(df['ps'][0]))

df['PS-K-mers']=k_mers(df['ps'].values)

df['PS-FFT']=fourier(df['ps'].values)

df['PS-Wavelet']=wavelet(df['ps'].values)

df['PS-K-mers + FFT']=fourier(df['PS-K-mers'].values, False)
df['PS-One Hot + FFT']=fourier(df['PS-One Hot'].values, False)

df['PS-K-mers + Wavelet']=wavelet(df['PS-K-mers'].values, True)
df['PS-One Hot + Wavelet']=wavelet(df['PS-One Hot'].values, True)


class ModelResults:
    def __init__(self):
        self.results = pd.DataFrame(columns=[
            'model', 'encoding', 'accuracy', 'f1_score_weighted', 'f1_score_macro',
            'precision_weighted', 'precision_macro', 'sensitivity_per_class',
            'specificity_per_class', 'precision_per_class', 'recall_per_class',
            'best_params', 'cv_score'
        ])
    
    def add_result(self, metrics, model_name, encoding_type):
        new_row = {
            'model': model_name,
            'encoding': encoding_type,
            **metrics  # Ya incluye el nombre del modelo desde evaluate_model
        }
        self.results = pd.concat([self.results, pd.DataFrame([new_row])], ignore_index=True)
    
    def get_best_model(self, metric='f1_score_weighted'):
        """Returns the best model configuration based on the specified metric"""
        return self.results.loc[self.results[metric].idxmax()]


import os
def ensure_dir(file_path):
    """Asegura que el directorio existe, si no, lo crea"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def plot_class_metrics(metrics_dict, model_name, data_name, class_names=None,
                      font="Baskervville, monospace",
                      font_color='black',
                      width=2000,
                      height=1500,
                      show=False,
                      ruta_base="graficos/",
                      scale=5,
                      x_tick_font=18,
                      x_title_font=22,
                      y_tick_font=18,
                      y_title_font=22,
                      legend_size=20):
    
    # Usar nombres de clase reales si se proporcionan
    if class_names is None:
        class_names = list(metrics_dict['sensitivity_per_class'].keys())
    
    # Crear subplots 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('<b>Sensitivity por Clase</b>', 
                       '<b>Specificity por Clase</b>',
                       '<b>Precision por Clase</b>', 
                       '<b>Recall por Clase</b>')
    )
    
    # Colores para cada métrica
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 
              'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
    
    metrics_names = ['sensitivity_per_class', 'specificity_per_class',
                    'precision_per_class', 'recall_per_class']
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for (metric_name, pos, color) in zip(metrics_names, positions, colors):
        values = list(metrics_dict[metric_name].values())
        
        fig.add_trace(
            go.Bar(
                x=class_names,
                y=values,
                name=metric_name.replace('_per_class', '').title(),
                marker_color=color,
                hovertemplate="Clase: %{x}<br>" +
                             f"{metric_name.replace('_per_class', '').title()}: " +
                             "%{y:.3f}<br><extra></extra>"
            ),
            row=pos[0], col=pos[1]
        )

    # Actualizar layout con rotación de etiquetas
    fig.update_layout(
        title={
            'text': f'<b>Métricas por Clase - {model_name}</b>',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        font=dict(
            family=font,
            size=20,
            color=font_color
        ),
        width=width,
        height=height,
        legend=dict(
            font=dict(size=legend_size, color=font_color),
            bgcolor='white',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Actualizar ejes con rotación de etiquetas
    fig.update_xaxes(
        tickangle=45,
        title_text="<b>Clase</b>",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgb(240,240,240)',
        tickfont=dict(size=x_tick_font, family=font, color=font_color),
        title_font=dict(size=x_title_font, family=font, color=font_color)
    )

    fig.update_yaxes(
        title_text="<b>Valor</b>",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgb(240,240,240)',
        range=[0, 1.05],
        tickfont=dict(size=y_tick_font, family=font, color=font_color),
        title_font=dict(size=y_title_font, family=font, color=font_color)
    )

    output_path = os.path.join(ruta_base, f"grafico_metricas_clase_{model_name}_{data_name}.png")
    ensure_dir(output_path)
    fig.write_image(output_path, scale=scale)
    if show:
        fig.show()


def plot_confusion_matrix(y_true, y_pred, classes, model_name, data_name,
                         width=2000,
                         height=1500,
                         show=False,
                         ruta_base="graficos/",
                         scale=5):
    
    # Calcular la matriz de confusión y normalizarla
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Configurar la escala de colores personalizada
    color_scale = [
        [0, 'rgb(255,255,255)'],      # Blanco para 0
        [0.000001, 'rgb(240,248,255)'],  # Casi blanco para valores muy pequeños
        [0.3, 'rgb(65,105,225)'],      # Azul real para valores medios
        [1, 'rgb(0,0,139)']            # Azul oscuro para valores máximos
    ]

    # Crear textos para el hover
    hover_text = [[f'Real: {classes[i]}<br>' +
                   f'Predicted: {classes[j]}<br>' +
                   f'Count: {cm[i][j]}<br>' +
                   f'Percentage: {cm_percent[i][j]:.1f}%'
                   for j in range(len(classes))] for i in range(len(classes))]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        hoverongaps=False,
        colorscale=color_scale,
        hoverinfo='text',
        text=hover_text,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Count",
                font=dict(size=16)
            ),
            tickfont=dict(size=14),
            len=0.75,
            thickness=20,
            x=1.02
        )
    ))

    # Actualizar el layout
    fig.update_layout(
        title=dict(
            text='Confusion Matrix - SVM',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        xaxis_title='Predicted Class',
        yaxis_title='Real Class',
        xaxis=dict(
            tickfont=dict(size=10),
            title_font=dict(size=16),
            tickangle=45,
            side='bottom',
            gridcolor='white',
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            title_font=dict(size=16),
            gridcolor='white',
            showgrid=False,
            # autorange='reversed'  # Invertir el eje Y para que coincida con la imagen
        ),
        width=width,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=150, r=100, t=100, b=150)
    )

    # Añadir anotaciones
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i][j] > 0:  # Solo mostrar valores mayores que 0
                # Valor principal
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"<b>{cm[i][j]}</b>",
                    showarrow=False,
                    font=dict(
                        color="white" if cm[i][j] > cm.max() / 2 else "black",
                        size=12
                    ),
                    yshift=10  # Ajustar posición vertical
                )
                # Porcentaje debajo
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"({cm_percent[i][j]:.1f}%)",
                    showarrow=False,
                    font=dict(
                        color="white" if cm[i][j] > cm.max() / 2 else "black",
                        size=10
                    ),
                    yshift=-10  # Ajustar posición vertical
                )

    # Guardar la figura
    output_path = os.path.join(ruta_base, f"confusion_matrix_{model_name}_{data_name}.png")
    ensure_dir(output_path)
    fig.write_image(output_path, scale=scale)
    fig.write_html('cm.html')
    
    if show:
        fig.show()


def plot_roc_curve(y_true, y_score, n_classes, model_name, data_name, class_names=None,
                   font="Baskervville, monospace",
                   font_color='black',
                   width=2000,
                   height=1500,
                   show=False,
                   ruta_base="graficos/",
                   scale=5,
                   x_tick_font=18,
                   x_title_font=22,
                   y_tick_font=18,
                   y_title_font=22,
                   legend_size=20):
    
    # Usar nombres de clase reales si se proporcionan
    if class_names is None:
        class_names = [f'Clase {i}' for i in range(n_classes)]
    
    # Calcular curvas ROC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Paleta de colores para las diferentes clases
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ] * 3
    
    fig = go.Figure()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        hover_text = [f'Clase: {class_names[i]}<br>' +
                     f'FPR: {fpr[i][j]:.3f}<br>' +
                     f'TPR: {tpr[i][j]:.3f}<br>' +
                     f'AUC: {roc_auc[i]:.3f}'
                     for j in range(len(fpr[i]))]
        
        fig.add_trace(go.Scatter(
            x=fpr[i],
            y=tpr[i],
            mode='lines',
            name=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})',
            line=dict(
                color=colors[i],
                width=2.5
            ),
            hoverinfo='text',
            hovertext=hover_text
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(
            color='gray',
            width=2,
            dash='dash'
        ),
        hoverinfo='skip'
    ))

    fig.update_layout(
        title={
            'text': f'<b>ROC Curves - {model_name}</b>',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=x_title_font + 4)
        },
        xaxis_title='<b>False Positive Rate</b>',
        yaxis_title='<b>True Positive Rate</b>',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgb(240,240,240)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgb(180,180,180)',
            tickfont=dict(size=x_tick_font, family=font, color=font_color),
            title_font=dict(size=x_title_font, family=font, color=font_color)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgb(240,240,240)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgb(180,180,180)',
            scaleanchor="x",
            scaleratio=1,
            tickfont=dict(size=y_tick_font, family=font, color=font_color),
            title_font=dict(size=y_title_font, family=font, color=font_color)
        ),
        font=dict(
            family=font,
            color=font_color
        ),
        width=width,
        height=height,
        legend=dict(
            font=dict(size=legend_size, color=font_color),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgb(180,180,180)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True
    )

    output_path = os.path.join(ruta_base, f"grafico_roc_{model_name}_{data_name}.png")
    ensure_dir(output_path)
    fig.write_image(output_path, scale=scale)
    if show:
        fig.show()


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


def evaluate_model(model, X_train, X_test, y_train, y_test, y_true, 
                  model_name, data_name, best_params, cv_score, class_names, show):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    
    # Obtener las clases numéricas únicas
    classes_num = np.unique(y_true)
    # Mapear las clases numéricas a nombres reales
    classes = [class_names[i] for i in classes_num]
    
    metrics = calculate_metrics(y_test, y_pred, y_score, classes)
    metrics['best_params'] = best_params
    metrics['cv_score'] = cv_score
    metrics['model_name'] = model_name
    
    print(f'\nResultados del modelo {model_name}:')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'F1 Score (Weighted): {metrics["f1_score_weighted"]:.4f}')
    print(f'F1 Score (Macro): {metrics["f1_score_macro"]:.4f}')
    
    if best_params:
        print(f'\nMejores parámetros encontrados:')
        for param, value in best_params.items():
            print(f'{param}: {value}')
    
    if cv_score:
        print(f'CV Score: {cv_score:.4f}')
    
    plot_confusion_matrix(y_test, y_pred, classes, model_name, data_name, show=show)
    plot_roc_curve(y_test, y_score, len(classes), model_name, data_name, classes, show=show)
    plot_class_metrics(metrics, model_name, data_name, show=show)
    
    return metrics


def train_svm(array, y_true, data_name, class_names, do_gridsearch=False, kernel='rbf', C=1.0, gamma='scale', n_cores=2, show=False):
    X_train, X_test, y_train, y_test = train_test_split(array, y_true, test_size=0.2, random_state=42)
    
    if do_gridsearch:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        model = GridSearchCV(
            SVC(probability=True),
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=n_cores
        )
        model.fit(X_train, y_train)
        best_params = model.best_params_
        cv_score = model.best_score_
        model = model.best_estimator_
    else:
        model = SVC(probability=True, kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train, y_train)
        best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}
        cv_score = None

    return evaluate_model(model, X_train, X_test, y_train, y_test, y_true, 'SVM', 
                         data_name, best_params, cv_score, class_names, show)


def train_random_forest(array, y_true, data_name, class_names, do_gridsearch=False, 
                       n_estimators=100, max_depth=None, min_samples_split=2, n_cores=2, show=False):
    X_train, X_test, y_train, y_test = train_test_split(array, y_true, test_size=0.2, random_state=42)
    
    if do_gridsearch:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        model = GridSearchCV(
            RandomForestClassifier(),
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=n_cores
        )
        model.fit(X_train, y_train)
        best_params = model.best_params_
        cv_score = model.best_score_
        model = model.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        model.fit(X_train, y_train)
        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split
        }
        cv_score = None
    
    return evaluate_model(model, X_train, X_test, y_train, y_test, y_true, 
                         'Random Forest', data_name, best_params, cv_score, class_names, show)


def train_xgboost(array, y_true, data_name, class_names, do_gridsearch=False, 
                 n_estimators=100, max_depth=6, learning_rate=0.3, n_cores=2, show=False):
    X_train, X_test, y_train, y_test = train_test_split(array, y_true, test_size=0.2, random_state=42)
    
    if do_gridsearch:
        # Espacio de búsqueda reducido
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.3],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        
        # Usar RandomizedSearchCV en lugar de GridSearchCV para mayor eficiencia
        from sklearn.model_selection import RandomizedSearchCV
        
        model = RandomizedSearchCV(
            XGBClassifier(
                eval_metric='mlogloss',
                tree_method='hist',          # Algoritmo más rápido
                grow_policy='lossguide'      # Política de crecimiento más eficiente
            ),
            param_grid,
            n_iter=10,                        # Reducir número de combinaciones a probar
            cv=3,                             # Reducir CV de 5 a 3
            scoring='f1_weighted',
            n_jobs=n_cores,
            verbose=1
        )
        model.fit(X_train, y_train)
        best_params = model.best_params_
        cv_score = model.best_score_
        model = model.best_estimator_
    else:
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric='mlogloss',
            tree_method='hist',              # Algoritmo más rápido
            grow_policy='lossguide'          # Política de crecimiento más eficiente
        )
        
        # Entrenamiento normal sin early stopping
        model.fit(X_train, y_train)
        
        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }
        cv_score = None
    
    return evaluate_model(model, X_train, X_test, y_train, y_test, y_true, 
                         'XGBoost', data_name, best_params, cv_score, class_names, show)


# Ejemplo de uso:
results = ModelResults()
reverse_map_genus = {i:j for j,i in map_genus.items()}
encodings=[
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



# Agregar un DataFrame para almacenar los tiempos de ejecución con parámetros óptimos
tiempos_optimos_df = pd.DataFrame(columns=[
    'encoding',
    'algoritmo',
    'tiempo_segundos',
    'tiempo_minutos',
    'parametros_optimos',
    'accuracy'
])

# Define los parámetros óptimos encontrados previamente para cada algoritmo y encoding
# Estos datos provienen de los resultados de grid search que compartiste
parametros_optimos = {
    'SVM': {
        'AS-One Hot': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'AS-K-mers': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS-FFT': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS-Wavelet': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS-K-mers + FFT': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS-One Hot + FFT': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS-K-mers + Wavelet': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'AS-One Hot + Wavelet': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'PS-One Hot': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS-K-mers': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS-FFT': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS-Wavelet': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS-K-mers + FFT': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS-One Hot + FFT': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'PS-K-mers + Wavelet': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'},
        'PS-One Hot + Wavelet': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
    },
    'Random Forest': {
        'AS-One Hot': {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 100},
        'AS-K-mers': {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 100},
        'AS-FFT': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100},
        'AS-Wavelet': {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 50},
        'AS-K-mers + FFT': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100},
        'AS-One Hot + FFT': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200},
        'AS-K-mers + Wavelet': {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 100},
        'AS-One Hot + Wavelet': {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 50},
        'PS-One Hot': {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 100},
        'PS-K-mers': {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 200},
        'PS-FFT': {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200},
        'PS-Wavelet': {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200},
        'PS-K-mers + FFT': {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200},
        'PS-One Hot + FFT': {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 200},
        'PS-K-mers + Wavelet': {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 50},
        'PS-One Hot + Wavelet': {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 200}
    },
    'XGBoost': {
        'AS-One Hot': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS-K-mers': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS-FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS-Wavelet': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS-K-mers + FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS-One Hot + FFT': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS-K-mers + Wavelet': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'AS-One Hot + Wavelet': {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS-One Hot': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS-K-mers': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS-FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS-Wavelet': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS-K-mers + FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS-One Hot + FFT': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS-K-mers + Wavelet': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        'PS-One Hot + Wavelet': {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8}
    }
}

print("Iniciando entrenamiento con parámetros óptimos...")
resultados_optimos = ModelResults()

for enc in encodings:
    # SVM con parámetros óptimos
    print(f"\nEntrenando SVM con {enc} usando parámetros óptimos...")
    params = parametros_optimos['SVM'][enc]
    start = time.time()
    
    # En lugar de hacer grid search, usamos directamente los parámetros óptimos
    metrics = train_svm(
        df[enc].tolist(), 
        df['clases_modelos'], 
        enc, 
        reverse_map_genus, 
        do_gridsearch=False,
        kernel=params['kernel'],
        C=params['C'],
        gamma=params['gamma']
    )
    
    resultados_optimos.add_result(metrics, 'SVM', enc)
    end = time.time()
    tiempo_segundos = end - start
    tiempo_minutos = tiempo_segundos / 60
    
    # Añadir al DataFrame de tiempos óptimos
    nueva_fila = {
        'encoding': enc,
        'algoritmo': 'SVM',
        'tiempo_segundos': tiempo_segundos,
        'tiempo_minutos': tiempo_minutos,
        'parametros_optimos': str(params),
        'accuracy': metrics['accuracy']
    }
    tiempos_optimos_df = pd.concat([tiempos_optimos_df, pd.DataFrame([nueva_fila])], ignore_index=True)
    
    print(f"El encoding {enc} en SVM con parámetros óptimos se demoró {tiempo_segundos:.2f} segundos o {tiempo_minutos:.2f} minutos")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Random Forest con parámetros óptimos
    print(f"\nEntrenando Random Forest con {enc} usando parámetros óptimos...")
    params = parametros_optimos['Random Forest'][enc]
    start = time.time()
    
    metrics = train_random_forest(
        df[enc].tolist(), 
        df['clases_modelos'], 
        enc, 
        reverse_map_genus, 
        do_gridsearch=False,
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split']
    )
    
    resultados_optimos.add_result(metrics, 'Random Forest', enc)
    end = time.time()
    tiempo_segundos = end - start
    tiempo_minutos = tiempo_segundos / 60
    
    # Añadir al DataFrame de tiempos óptimos
    nueva_fila = {
        'encoding': enc,
        'algoritmo': 'Random Forest',
        'tiempo_segundos': tiempo_segundos,
        'tiempo_minutos': tiempo_minutos,
        'parametros_optimos': str(params),
        'accuracy': metrics['accuracy']
    }
    tiempos_optimos_df = pd.concat([tiempos_optimos_df, pd.DataFrame([nueva_fila])], ignore_index=True)
    
    print(f"El encoding {enc} en Random Forest con parámetros óptimos se demoró {tiempo_segundos:.2f} segundos o {tiempo_minutos:.2f} minutos")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # XGBoost con parámetros óptimos
    print(f"\nEntrenando XGBoost con {enc} usando parámetros óptimos...")
    params = parametros_optimos['XGBoost'][enc]
    start = time.time()
    
    metrics = train_xgboost(
        df[enc].tolist(), 
        df['clases_modelos'], 
        enc, 
        reverse_map_genus, 
        do_gridsearch=False,
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate']
    )
    
    resultados_optimos.add_result(metrics, 'XGBoost', enc)
    end = time.time()
    tiempo_segundos = end - start
    tiempo_minutos = tiempo_segundos / 60
    
    # Añadir al DataFrame de tiempos óptimos
    nueva_fila = {
        'encoding': enc,
        'algoritmo': 'XGBoost',
        'tiempo_segundos': tiempo_segundos,
        'tiempo_minutos': tiempo_minutos,
        'parametros_optimos': str(params),
        'accuracy': metrics['accuracy']
    }
    tiempos_optimos_df = pd.concat([tiempos_optimos_df, pd.DataFrame([nueva_fila])], ignore_index=True)
    
    print(f"El encoding {enc} en XGBoost con parámetros óptimos se demoró {tiempo_segundos:.2f} segundos o {tiempo_minutos:.2f} minutos")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Guardar resultados parciales después de cada encoding
    # resultados_optimos.results.to_excel(f'resultados_optimos_parciales_{enc}.xlsx')
    # tiempos_optimos_df.to_excel(f'tiempos_optimos_parciales_{enc}.xlsx', index=False)
    
    print("==================================================================")

# Guardar los resultados finales con timestamp
from datetime import datetime 
hoy = datetime.today()

# Guardar los resultados principales
resultados_optimos.results.to_excel(f'resultados_parametros_optimos_{hoy.minute}-{hoy.day}-{hoy.month}-{hoy.year}.xlsx')

# Guardar también el DataFrame de tiempos
tiempos_optimos_df.to_excel(f'tiempos_parametros_optimos_{hoy.minute}-{hoy.day}-{hoy.month}-{hoy.year}.xlsx', index=False)

# Crear un análisis comparativo entre tiempo y rendimiento
analisis_df = tiempos_optimos_df.copy()
analisis_df['eficiencia'] = analisis_df['accuracy'] / analisis_df['tiempo_minutos']

# Encontrar el mejor modelo en términos de eficiencia (accuracy/tiempo)
mejor_eficiencia = analisis_df.loc[analisis_df['eficiencia'].idxmax()]
print("\n=== MEJOR MODELO EN TÉRMINOS DE EFICIENCIA (ACCURACY/TIEMPO) ===")
print(f"Encoding: {mejor_eficiencia['encoding']}")
print(f"Algoritmo: {mejor_eficiencia['algoritmo']}")
print(f"Accuracy: {mejor_eficiencia['accuracy']:.4f}")
print(f"Tiempo (minutos): {mejor_eficiencia['tiempo_minutos']:.2f}")
print(f"Eficiencia: {mejor_eficiencia['eficiencia']:.4f} (accuracy/minuto)")
print(f"Parámetros: {mejor_eficiencia['parametros_optimos']}")

# Guardar el análisis de eficiencia
analisis_df.to_excel(f'analisis_eficiencia_{hoy.minute}-{hoy.day}-{hoy.month}-{hoy.year}.xlsx', index=False)