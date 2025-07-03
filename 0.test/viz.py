import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_benchmark_results(filepath="benchmark_results/encoding_benchmark_results.csv"):
    """Load benchmark results from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded benchmark results: {len(df)} records")
        return df
    except FileNotFoundError:
        print(f"Error: Results file not found at {filepath}")
        print("Please run the benchmark script first!")
        return None
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def create_visualization_directory():
    """Create directory for visualizations."""
    viz_dir = "benchmark_visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    return viz_dir

def save_plot(fig, filename, viz_dir, scale=3):
    """Save plot to visualization directory."""
    filepath = os.path.join(viz_dir, f"{filename}.png")
    fig.write_image(filepath, scale=scale)
    print(f"Visualization saved: {filepath}")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_performance_comparison_by_type(df, viz_dir):
    """Compare AS vs PS performance for each encoding method."""
    
    # Prepare data for comparison
    encoding_methods = [method.replace('AS-', '').replace('PS-', '') for method in df['encoding_method'].unique()]
    encoding_methods = list(set(encoding_methods))
    
    comparison_data = []
    for method in encoding_methods:
        as_method = f"AS-{method}"
        ps_method = f"PS-{method}"
        
        as_data = df[df['encoding_method'] == as_method]
        ps_data = df[df['encoding_method'] == ps_method]
        
        if not as_data.empty and not ps_data.empty:
            # Average across all sample sizes
            comparison_data.append({
                'method': method,
                'as_avg_time': as_data['avg_time'].mean(),
                'ps_avg_time': ps_data['avg_time'].mean(),
                'as_avg_memory': as_data['avg_memory_mb'].mean(),
                'ps_avg_memory': ps_data['avg_memory_mb'].mean(),
                'time_ratio_ps_as': ps_data['avg_time'].mean() / as_data['avg_time'].mean(),
                'memory_ratio_ps_as': ps_data['avg_memory_mb'].mean() / as_data['avg_memory_mb'].mean()
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    if comp_df.empty:
        print("No comparison data available")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Average Execution Time (AS vs PS)",
            "Average Memory Usage (AS vs PS)", 
            "Time Ratio (PS/AS)",
            "Memory Ratio (PS/AS)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Time comparison
    fig.add_trace(
        go.Bar(
            x=comp_df['method'],
            y=comp_df['as_avg_time'],
            name='AS (Original)',
            marker_color='rgb(55, 83, 109)',
            hovertemplate='<b>AS-%{x}</b><br>Time: %{y:.3f}s<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=comp_df['method'],
            y=comp_df['ps_avg_time'],
            name='PS (Aligned)',
            marker_color='rgb(26, 118, 255)',
            hovertemplate='<b>PS-%{x}</b><br>Time: %{y:.3f}s<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Memory comparison
    fig.add_trace(
        go.Bar(
            x=comp_df['method'],
            y=comp_df['as_avg_memory'],
            name='AS (Original)',
            marker_color='rgb(55, 83, 109)',
            showlegend=False,
            hovertemplate='<b>AS-%{x}</b><br>Memory: %{y:.2f}MB<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=comp_df['method'],
            y=comp_df['ps_avg_memory'],
            name='PS (Aligned)',
            marker_color='rgb(26, 118, 255)',
            showlegend=False,
            hovertemplate='<b>PS-%{x}</b><br>Memory: %{y:.2f}MB<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Time ratio
    fig.add_trace(
        go.Bar(
            x=comp_df['method'],
            y=comp_df['time_ratio_ps_as'],
            name='Time Ratio',
            marker_color='rgb(255, 127, 14)',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Ratio: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Memory ratio
    fig.add_trace(
        go.Bar(
            x=comp_df['method'],
            y=comp_df['memory_ratio_ps_as'],
            name='Memory Ratio',
            marker_color='rgb(44, 160, 44)',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Ratio: %{y:.2f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add reference lines for ratios
    fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="AS (Original) vs PS (Aligned) Sequences Performance Comparison",
        height=800,
        width=1400,
        template='presentation'
    )
    
    # Update x-axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(tickangle=45, row=row, col=col)
    
    # Update y-axes
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
    fig.update_yaxes(title_text="Time Ratio (PS/AS)", row=2, col=1)
    fig.update_yaxes(title_text="Memory Ratio (PS/AS)", row=2, col=2)
    
    save_plot(fig, "as_vs_ps_comparison", viz_dir)
    return fig

def plot_scalability_analysis(df, viz_dir):
    """Plot scalability analysis for different sample sizes."""
    
    # Separate AS and PS data
    as_data = df[df['sequence_type'] == 'AS'].copy()
    ps_data = df[df['sequence_type'] == 'PS'].copy()
    
    for seq_type, data in [('AS', as_data), ('PS', ps_data)]:
        if data.empty:
            continue
            
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"Execution Time vs Dataset Size ({seq_type})",
                f"Memory Usage vs Dataset Size ({seq_type})"
            ),
            vertical_spacing=0.15
        )
        
        # Get unique encoding methods for this sequence type
        methods = data['encoding_method'].unique()
        colors = px.colors.qualitative.Set3
        
        for i, method in enumerate(methods):
            method_data = data[data['encoding_method'] == method].sort_values('num_sequences')
            color = colors[i % len(colors)]
            method_name = method.replace(f'{seq_type}-', '')
            
            # Time scalability
            fig.add_trace(
                go.Scatter(
                    x=method_data['num_sequences'],
                    y=method_data['avg_time'],
                    mode='lines+markers',
                    name=method_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{method_name}</b><br>Sequences: %{{x}}<br>Time: %{{y:.3f}}s<extra></extra>',
                    legendgroup=method_name,
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Memory scalability
            fig.add_trace(
                go.Scatter(
                    x=method_data['num_sequences'],
                    y=method_data['avg_memory_mb'],
                    mode='lines+markers',
                    name=method_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{method_name}</b><br>Sequences: %{{x}}<br>Memory: %{{y:.2f}}MB<extra></extra>',
                    legendgroup=method_name,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title_text=f"Scalability Analysis - {seq_type} Sequences",
            height=800,
            width=1200,
            template='presentation'
        )
        
        fig.update_xaxes(title_text="Number of Sequences", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Number of Sequences", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Time (seconds)", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Memory (MB)", type="log", row=2, col=1)
        
        save_plot(fig, f"scalability_analysis_{seq_type.lower()}", viz_dir)

def plot_scatter_with_regression(df, viz_dir):
    """Create scatter plots with regression analysis for each sequence type."""
    
    for seq_type in ['AS', 'PS']:
        data = df[df['sequence_type'] == seq_type].copy()
        
        if data.empty:
            continue
        
        fig = go.Figure()
        
        # Color mapping for encoding methods
        methods = data['encoding_method'].unique()
        colors = px.colors.qualitative.Set3
        
        for i, method in enumerate(methods):
            method_data = data[data['encoding_method'] == method]
            color = colors[i % len(colors)]
            method_name = method.replace(f'{seq_type}-', '')
            
            fig.add_trace(go.Scatter(
                x=method_data['avg_time'],
                y=method_data['avg_memory_mb'],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                text=[f"{method_name}<br>{row['num_sequences']}" for _, row in method_data.iterrows()],
                textposition="top center",
                textfont=dict(size=8),
                name=method_name,
                hovertemplate=f'<b>{method_name}</b><br>Time: %{{x:.3f}}s<br>Memory: %{{y:.2f}}MB<br>Sequences: %{{text}}<extra></extra>'
            ))
        
        # Add polynomial regression line if we have enough data points
        if len(data) > 3:
            X = data['avg_time'].values.reshape(-1, 1)
            y = data['avg_memory_mb'].values
            
            # Fit polynomial regression (degree 2)
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            poly_reg = LinearRegression()
            poly_reg.fit(X_poly, y)
            
            # Generate smooth curve
            X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            X_range_poly = poly_features.transform(X_range)
            y_pred = poly_reg.predict(X_range_poly)
            
            fig.add_trace(go.Scatter(
                x=X_range.flatten(),
                y=y_pred,
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name='Polynomial Trend',
                hovertemplate='Trend Line<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Time vs Memory Usage with Trend Analysis - {seq_type} Sequences',
            xaxis_title='Execution Time (seconds)',
            yaxis_title='Memory Usage (MB)',
            height=700,
            width=1000,
            template='presentation',
            showlegend=True
        )
        
        save_plot(fig, f"scatter_regression_{seq_type.lower()}", viz_dir)

def plot_encoding_efficiency_heatmap(df, viz_dir):
    """Create heatmap showing encoding efficiency across methods and sample sizes."""
    
    for seq_type in ['AS', 'PS']:
        data = df[df['sequence_type'] == seq_type].copy()
        
        if data.empty:
            continue
        
        # Create pivot table for heatmap
        pivot_time = data.pivot_table(
            values='avg_time', 
            index='encoding_method', 
            columns='num_sequences', 
            aggfunc='mean'
        )
        
        pivot_memory = data.pivot_table(
            values='avg_memory_mb', 
            index='encoding_method', 
            columns='num_sequences', 
            aggfunc='mean'
        )
        
        # Remove sequence type prefix for cleaner labels
        pivot_time.index = [idx.replace(f'{seq_type}-', '') for idx in pivot_time.index]
        pivot_memory.index = [idx.replace(f'{seq_type}-', '') for idx in pivot_memory.index]
        
        # Create subplots for time and memory heatmaps
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"Execution Time Heatmap - {seq_type} Sequences (seconds)",
                f"Memory Usage Heatmap - {seq_type} Sequences (MB)"
            ),
            vertical_spacing=0.15
        )
        
        # Time heatmap
        fig.add_trace(
            go.Heatmap(
                z=pivot_time.values,
                x=[f"{int(col/1000)}K" if col >= 1000 else str(col) for col in pivot_time.columns],
                y=pivot_time.index,
                colorscale='Viridis',
                text=np.round(pivot_time.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{y}</b><br>Sequences: %{x}<br>Time: %{z:.3f}s<extra></extra>',
                showscale=True,
                colorbar=dict(title="Time (s)", y=0.8, len=0.35)
            ),
            row=1, col=1
        )
        
        # Memory heatmap
        fig.add_trace(
            go.Heatmap(
                z=pivot_memory.values,
                x=[f"{int(col/1000)}K" if col >= 1000 else str(col) for col in pivot_memory.columns],
                y=pivot_memory.index,
                colorscale='Plasma',
                text=np.round(pivot_memory.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{y}</b><br>Sequences: %{x}<br>Memory: %{z:.2f}MB<extra></extra>',
                showscale=True,
                colorbar=dict(title="Memory (MB)", y=0.2, len=0.35)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title_text=f"Performance Heatmaps - {seq_type} Sequences",
            height=900,
            width=1000,
            template='presentation'
        )
        
        fig.update_xaxes(title_text="Number of Sequences", row=1, col=1)
        fig.update_xaxes(title_text="Number of Sequences", row=2, col=1)
        fig.update_yaxes(title_text="Encoding Method", row=1, col=1)
        fig.update_yaxes(title_text="Encoding Method", row=2, col=1)
        
        save_plot(fig, f"efficiency_heatmap_{seq_type.lower()}", viz_dir)

def plot_best_performers_summary(df, viz_dir):
    """Create summary visualization of best performing methods."""
    
    # Find best performers for each sample size
    best_performers = []
    
    for sample_size in df['num_sequences'].unique():
        sample_data = df[df['num_sequences'] == sample_size]
        
        # Best time performers
        best_time_as = sample_data[sample_data['sequence_type'] == 'AS'].loc[
            sample_data[sample_data['sequence_type'] == 'AS']['avg_time'].idxmin()
        ]
        best_time_ps = sample_data[sample_data['sequence_type'] == 'PS'].loc[
            sample_data[sample_data['sequence_type'] == 'PS']['avg_time'].idxmin()
        ]
        
        # Best memory performers
        best_memory_as = sample_data[sample_data['sequence_type'] == 'AS'].loc[
            sample_data[sample_data['sequence_type'] == 'AS']['avg_memory_mb'].idxmin()
        ]
        best_memory_ps = sample_data[sample_data['sequence_type'] == 'PS'].loc[
            sample_data[sample_data['sequence_type'] == 'PS']['avg_memory_mb'].idxmin()
        ]
        
        best_performers.extend([
            {
                'sample_size': sample_size,
                'metric': 'Time',
                'sequence_type': 'AS',
                'method': best_time_as['encoding_method'].replace('AS-', ''),
                'value': best_time_as['avg_time']
            },
            {
                'sample_size': sample_size,
                'metric': 'Time',
                'sequence_type': 'PS',
                'method': best_time_ps['encoding_method'].replace('PS-', ''),
                'value': best_time_ps['avg_time']
            },
            {
                'sample_size': sample_size,
                'metric': 'Memory',
                'sequence_type': 'AS',
                'method': best_memory_as['encoding_method'].replace('AS-', ''),
                'value': best_memory_as['avg_memory_mb']
            },
            {
                'sample_size': sample_size,
                'metric': 'Memory',
                'sequence_type': 'PS',
                'method': best_memory_ps['encoding_method'].replace('PS-', ''),
                'value': best_memory_ps['avg_memory_mb']
            }
        ])
    
    best_df = pd.DataFrame(best_performers)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Best Time Performance (AS)",
            "Best Time Performance (PS)",
            "Best Memory Performance (AS)",
            "Best Memory Performance (PS)"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Plot configurations
    plot_configs = [
        (1, 1, 'AS', 'Time'),
        (1, 2, 'PS', 'Time'),
        (2, 1, 'AS', 'Memory'),
        (2, 2, 'PS', 'Memory')
    ]
    
    colors = px.colors.qualitative.Set2
    
    for row, col, seq_type, metric in plot_configs:
        plot_data = best_df[(best_df['sequence_type'] == seq_type) & (best_df['metric'] == metric)]
        
        fig.add_trace(
            go.Scatter(
                x=[f"{int(size/1000)}K" if size >= 1000 else str(size) for size in plot_data['sample_size']],
                y=plot_data['value'],
                mode='lines+markers+text',
                text=plot_data['method'],
                textposition="top center",
                textfont=dict(size=9),
                line=dict(color=colors[(row-1)*2 + (col-1)], width=3),
                marker=dict(size=10),
                name=f"{seq_type} {metric}",
                hovertemplate=f'<b>%{{text}}</b><br>Sequences: %{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Best Performing Methods Across Different Sample Sizes",
        height=800,
        width=1400,
        template='presentation'
    )
    
    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="Sample Size", row=row, col=col)
    
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=2, col=2)
    
    save_plot(fig, "best_performers_summary", viz_dir)
    return fig

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def generate_all_visualizations(results_filepath=None):
    """Generate all visualizations from benchmark results."""
    
    print("Starting Visualization Generation")
    print("=" * 50)
    
    # Load results
    if results_filepath is None:
        results_filepath = "benchmark_results/encoding_benchmark_results.csv"
    
    df = load_benchmark_results(results_filepath)
    if df is None:
        return
    
    # Create visualization directory
    viz_dir = create_visualization_directory()
    
    print(f"Generating visualizations for {len(df)} benchmark results...")
    
    # Generate all visualizations
    print("\n1. AS vs PS Performance Comparison...")
    plot_performance_comparison_by_type(df, viz_dir)
    
    print("2. Scalability Analysis...")
    plot_scalability_analysis(df, viz_dir)
    
    print("3. Scatter Plots with Regression...")
    plot_scatter_with_regression(df, viz_dir)
    
    print("4. Efficiency Heatmaps...")
    plot_encoding_efficiency_heatmap(df, viz_dir)
    
    print("5. Best Performers Summary...")
    plot_best_performers_summary(df, viz_dir)
    
    print(f"\nAll visualizations completed!")
    print(f"Check the '{viz_dir}' directory for all generated plots.")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    
    total_methods = len(df['encoding_method'].unique())
    total_sample_sizes = len(df['num_sequences'].unique())
    as_methods = len(df[df['sequence_type'] == 'AS']['encoding_method'].unique())
    ps_methods = len(df[df['sequence_type'] == 'PS']['encoding_method'].unique())
    
    print(f"Total encoding methods analyzed: {total_methods}")
    print(f"AS methods: {as_methods}")
    print(f"PS methods: {ps_methods}")
    print(f"Sample sizes tested: {sorted(df['num_sequences'].unique())}")
    print(f"Total benchmark records: {len(df)}")
    
    # Best overall performers
    fastest_overall = df.loc[df['avg_time'].idxmin()]
    most_efficient_memory = df.loc[df['avg_memory_mb'].idxmin()]
    
    print(f"\nOverall fastest method: {fastest_overall['encoding_method']} ({fastest_overall['avg_time']:.4f}s)")
    print(f"Most memory efficient: {most_efficient_memory['encoding_method']} ({most_efficient_memory['avg_memory_mb']:.2f}MB)")
    
    print("="*60)

if __name__ == "__main__":
    try:
        # Generate all visualizations
        generate_all_visualizations()
        
    except Exception as e:
        print(f"Error during visualization generation: {e}")
        import traceback
        traceback.print_exc()