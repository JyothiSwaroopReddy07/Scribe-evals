"""Enhanced interactive dashboard with real-time monitoring and advanced visualizations."""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from datetime import datetime


st.set_page_config(
    page_title="DeepScribe Enhanced Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background-color: #ffebee;
        color: #c62828;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #c62828;
    }
    .alert-warning {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ef6c00;
    }
    </style>
    """, unsafe_allow_html=True)


def load_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_confidence_distribution(results_data: List[Dict]) -> go.Figure:
    """Create confidence distribution plot."""
    confidences_by_evaluator = {}
    
    for result in results_data:
        for eval_name, eval_result in result['evaluations'].items():
            confidence = eval_result['metrics'].get('confidence', None)
            if confidence is not None:
                if eval_name not in confidences_by_evaluator:
                    confidences_by_evaluator[eval_name] = []
                confidences_by_evaluator[eval_name].append(confidence)
    
    fig = go.Figure()
    
    for eval_name, confidences in confidences_by_evaluator.items():
        fig.add_trace(go.Violin(
            y=confidences,
            name=eval_name,
            box_visible=True,
            meanline_visible=True
        ))
    
    fig.update_layout(
        title="Confidence Score Distribution by Evaluator",
        yaxis_title="Confidence Score",
        xaxis_title="Evaluator",
        height=400,
        showlegend=True
    )
    
    return fig


def create_score_vs_confidence(results_data: List[Dict]) -> go.Figure:
    """Create scatter plot of scores vs confidence."""
    data_points = []
    
    for result in results_data:
        for eval_name, eval_result in result['evaluations'].items():
            score = eval_result.get('score', 0)
            confidence = eval_result['metrics'].get('confidence', None)
            issues = len(eval_result.get('issues', []))
            
            if confidence is not None:
                data_points.append({
                    'evaluator': eval_name,
                    'score': score,
                    'confidence': confidence,
                    'issues': issues,
                    'note_id': result['note_id']
                })
    
    df = pd.DataFrame(data_points)
    
    if df.empty:
        return go.Figure()
    
    fig = px.scatter(
        df,
        x='confidence',
        y='score',
        color='evaluator',
        size='issues',
        hover_data=['note_id', 'issues'],
        title='Evaluation Score vs Confidence',
        labels={
            'confidence': 'Confidence Score',
            'score': 'Evaluation Score',
            'issues': 'Number of Issues'
        }
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_uncertainty_analysis(results_data: List[Dict]) -> go.Figure:
    """Create uncertainty analysis visualization."""
    uncertainty_data = []
    
    for result in results_data:
        for eval_name, eval_result in result['evaluations'].items():
            uncertainty = eval_result['metrics'].get('confidence_uncertainty', None)
            variance = eval_result['metrics'].get('confidence_variance', None)
            
            if uncertainty is not None:
                uncertainty_data.append({
                    'evaluator': eval_name,
                    'uncertainty': uncertainty,
                    'variance': variance if variance is not None else 0,
                    'note_id': result['note_id']
                })
    
    df = pd.DataFrame(uncertainty_data)
    
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Uncertainty Distribution', 'Variance Distribution')
    )
    
    for eval_name in df['evaluator'].unique():
        eval_data = df[df['evaluator'] == eval_name]
        
        fig.add_trace(
            go.Box(y=eval_data['uncertainty'], name=eval_name, showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Box(y=eval_data['variance'], name=eval_name),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Evaluator", row=1, col=1)
    fig.update_xaxes(title_text="Evaluator", row=1, col=2)
    fig.update_yaxes(title_text="Uncertainty", row=1, col=1)
    fig.update_yaxes(title_text="Variance", row=1, col=2)
    
    fig.update_layout(height=400, title_text="Confidence Uncertainty & Variance Analysis")
    
    return fig


def create_issue_severity_trend(results_data: List[Dict]) -> go.Figure:
    """Create issue severity trend across notes."""
    severity_counts = []
    
    for result in results_data:
        note_severities = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for eval_result in result['evaluations'].values():
            for issue in eval_result.get('issues', []):
                severity = issue['severity']
                if severity in note_severities:
                    note_severities[severity] += 1
        
        severity_counts.append({
            'note_id': result['note_id'][:8],  # Truncate for display
            **note_severities
        })
    
    df = pd.DataFrame(severity_counts)
    
    if df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    colors = {
        'critical': '#d62728',
        'high': '#ff7f0e',
        'medium': '#ffbb00',
        'low': '#2ca02c'
    }
    
    for severity in ['critical', 'high', 'medium', 'low']:
        fig.add_trace(go.Bar(
            x=df['note_id'],
            y=df[severity],
            name=severity.capitalize(),
            marker_color=colors[severity]
        ))
    
    fig.update_layout(
        title="Issue Severity Distribution Across Notes",
        xaxis_title="Note ID",
        yaxis_title="Number of Issues",
        barmode='stack',
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return fig


def create_evaluator_performance_heatmap(summary: Dict[str, Any]) -> go.Figure:
    """Create heatmap of evaluator performance metrics."""
    if 'evaluators' not in summary:
        return go.Figure()
    
    evaluators = list(summary['evaluators'].keys())
    metrics = ['average_score', 'average_confidence', 'high_confidence_rate']
    
    data = []
    for metric in metrics:
        row = []
        for eval_name in evaluators:
            value = summary['evaluators'][eval_name].get(metric, 0)
            row.append(value)
        data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=evaluators,
        y=['Avg Score', 'Avg Confidence', 'High Conf Rate'],
        colorscale='RdYlGn',
        text=[[f'{val:.3f}' for val in row] for row in data],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Value")
    ))
    
    fig.update_layout(
        title="Evaluator Performance Heatmap",
        height=300,
        xaxis={'tickangle': -45}
    )
    
    return fig


def create_performance_metrics_chart(summary: Dict[str, Any]) -> go.Figure:
    """Create performance metrics visualization."""
    if 'performance' not in summary:
        return go.Figure()
    
    perf = summary['performance']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Success Rate', 'Average Latency by Evaluator'),
        specs=[[{'type': 'indicator'}, {'type': 'bar'}]]
    )
    
    # Success rate gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=perf['success_rate'] * 100,
            title={'text': "Success Rate (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ),
        row=1, col=1
    )
    
    # Latency bar chart
    if 'evaluator_latencies' in perf:
        eval_names = list(perf['evaluator_latencies'].keys())
        avg_latencies = [perf['evaluator_latencies'][name]['average'] 
                        for name in eval_names]
        
        fig.add_trace(
            go.Bar(
                x=eval_names,
                y=avg_latencies,
                marker_color='indianred',
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Evaluator", row=1, col=2, tickangle=-45)
    fig.update_yaxes(title_text="Latency (s)", row=1, col=2)
    
    fig.update_layout(height=400)
    
    return fig


def display_alerts(summary: Dict[str, Any], results_data: List[Dict]):
    """Display alerts for critical issues and low confidence."""
    st.subheader("Alerts & Warnings")
    
    alerts = []
    
    # Check for notes with critical issues
    if 'issue_analysis' in summary:
        critical_count = summary['issue_analysis'].get('by_severity', {}).get('critical', 0)
        if critical_count > 0:
            alerts.append({
                'type': 'critical',
                'message': f"CRITICAL: {critical_count} critical issues detected across {summary.get('total_notes', 0)} notes"
            })
    
    # Check for low confidence evaluations
    if 'confidence_analysis' in summary:
        low_conf_rate = summary['confidence_analysis'].get('low_confidence_rate', 0)
        if low_conf_rate > 0.2:  # More than 20% low confidence
            alerts.append({
                'type': 'warning',
                'message': f"WARNING: {low_conf_rate:.1%} of evaluations have low confidence (< 0.5)"
            })
    
    # Check for notes with multiple critical issues
    notes_with_multiple_critical = []
    for result in results_data:
        critical_issues = []
        for eval_name, eval_result in result['evaluations'].items():
            critical = [i for i in eval_result.get('issues', []) if i['severity'] == 'critical']
            critical_issues.extend(critical)
        
        if len(critical_issues) >= 2:
            notes_with_multiple_critical.append(result['note_id'])
    
    if notes_with_multiple_critical:
        alerts.append({
            'type': 'critical',
            'message': f"ALERT: {len(notes_with_multiple_critical)} notes have multiple critical issues: {', '.join(notes_with_multiple_critical[:5])}"
        })
    
    # Display alerts
    if alerts:
        for alert in alerts:
            if alert['type'] == 'critical':
                st.markdown(f'<div class="alert-critical">{alert["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-warning">{alert["message"]}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.success("No critical alerts detected")


def display_detailed_note_analysis(result: Dict):
    """Display enhanced detailed view of a single note evaluation."""
    st.subheader(f"Note: {result['note_id']}")
    
    # Overall scores with confidence
    st.write("### Overall Scores & Confidence")
    
    scores_data = []
    for eval_name, eval_result in result['evaluations'].items():
        scores_data.append({
            'Evaluator': eval_name,
            'Score': eval_result['score'],
            'Confidence': eval_result['metrics'].get('confidence', 'N/A'),
            'Uncertainty': eval_result['metrics'].get('confidence_uncertainty', 'N/A'),
            'Issues': len(eval_result['issues'])
        })
    
    scores_df = pd.DataFrame(scores_data)
    st.dataframe(scores_df.style.background_gradient(subset=['Score', 'Confidence'], cmap='RdYlGn'), 
                use_container_width=True)
    
    # Visualize scores
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Score',
        x=scores_df['Evaluator'],
        y=scores_df['Score'],
        marker_color='lightblue'
    ))
    
    if 'Confidence' in scores_df.columns:
        fig.add_trace(go.Scatter(
            name='Confidence',
            x=scores_df['Evaluator'],
            y=scores_df['Confidence'],
            mode='lines+markers',
            marker=dict(size=10, color='orange'),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title='Scores and Confidence',
        yaxis=dict(title='Score', range=[0, 1]),
        yaxis2=dict(title='Confidence', overlaying='y', side='right', range=[0, 1]),
        height=400,
        xaxis={'tickangle': -45}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Issues breakdown
    st.write("### Issues Found")
    
    for eval_name, eval_result in result['evaluations'].items():
        if eval_result['issues']:
            with st.expander(f"{eval_name} ({len(eval_result['issues'])} issues)", expanded=False):
                for issue in eval_result['issues']:
                    severity_icons = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢',
                        'info': '🔵'
                    }
                    icon = severity_icons.get(issue['severity'], '⚪')
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"{icon} **{issue['type']}** ({issue['severity']})")
                        st.write(f"**Description:** {issue['description']}")
                        if issue.get('location'):
                            st.write(f"**Location:** {issue['location']}")
                    
                    with col2:
                        conf = issue.get('confidence', 0)
                        st.metric("Confidence", f"{conf:.2f}")
                    
                    if issue.get('evidence'):
                        with st.expander("Evidence Details"):
                            st.json(issue['evidence'])
                    
                    st.divider()
        else:
            st.success(f"{eval_name}: No issues found")
    
    # Detailed metrics
    st.write("### Detailed Metrics")
    
    for eval_name, eval_result in result['evaluations'].items():
        with st.expander(f"{eval_name} Metrics"):
            metrics_df = pd.DataFrame([eval_result['metrics']]).T
            metrics_df.columns = ['Value']
            st.dataframe(metrics_df)
            
            # Show analysis steps if available
            if 'metadata' in eval_result and 'analysis_steps' in eval_result['metadata']:
                st.write("**Analysis Steps:**")
                st.json(eval_result['metadata']['analysis_steps'])


def main():
    """Main dashboard application."""
    st.title("DeepScribe Enhanced Evaluation Dashboard")
    st.markdown("Advanced evaluation insights with confidence scoring and real-time monitoring")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # File selector
    results_dir = Path("results")
    if not results_dir.exists():
        st.error("No results directory found. Please run the evaluation pipeline first.")
        st.code("python -m src.enhanced_pipeline --num-samples 10")
        return
    
    result_files = list(results_dir.glob("enhanced_evaluation_results_*.json"))
    
    if not result_files:
        st.warning("No enhanced evaluation results found. Looking for standard results...")
        result_files = list(results_dir.glob("evaluation_results_*.json"))
    
    if not result_files:
        st.error("No evaluation results found. Please run the evaluation pipeline first.")
        st.code("python -m src.enhanced_pipeline --num-samples 10")
        return
    
    # Sort by timestamp (newest first)
    result_files = sorted(result_files, reverse=True)
    
    selected_file = st.sidebar.selectbox(
        "Select Evaluation Results",
        result_files,
        format_func=lambda x: x.name
    )
    
    if not selected_file:
        return
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 30s)", value=False)
    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 30 seconds")
        import time
        time.sleep(30)
        st.rerun()
    
    # Load results
    results = load_results(selected_file)
    
    # Sidebar info
    st.sidebar.header("Evaluation Info")
    st.sidebar.write(f"**Timestamp:** {results['metadata']['timestamp']}")
    st.sidebar.write(f"**Notes Evaluated:** {results['metadata']['num_notes']}")
    st.sidebar.write(f"**Evaluators:** {results['metadata']['num_evaluators']}")
    st.sidebar.write("**Evaluator Names:**")
    for eval_name in results['metadata']['evaluators']:
        st.sidebar.write(f"- {eval_name}")
    
    # Display alerts
    display_alerts(results['summary'], results['results'])
    
    # Key metrics
    st.header("Key Metrics")
    
    summary = results['summary']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Notes", summary['total_notes'])
    
    with col2:
        if 'overall_statistics' in summary:
            avg_score = summary['overall_statistics'].get('average_score', 0)
            st.metric("Average Score", f"{avg_score:.3f}")
    
    with col3:
        if 'confidence_analysis' in summary:
            avg_conf = summary['confidence_analysis'].get('average_confidence', 0)
            st.metric("Avg Confidence", f"{avg_conf:.3f}")
    
    with col4:
        if 'issue_analysis' in summary:
            total_issues = summary['issue_analysis'].get('total_issues', 0)
            st.metric("Total Issues", total_issues)
    
    with col5:
        if 'issue_analysis' in summary:
            critical = summary['issue_analysis'].get('by_severity', {}).get('critical', 0)
            st.metric("Critical Issues", critical, delta=None, delta_color="inverse")
    
    # Performance metrics
    if 'performance' in summary:
        st.header("⚡ Performance Metrics")
        
        perf = summary['performance']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Success Rate", f"{perf['success_rate']:.1%}")
        
        with col2:
            st.metric("Avg Latency/Note", f"{perf['average_latency_per_note']:.2f}s")
        
        with col3:
            st.metric("Total Latency", f"{perf['total_latency']:.1f}s")
        
        # Performance visualization
        fig = create_performance_metrics_chart(summary)
        st.plotly_chart(fig, use_container_width=True)
    
    # Main visualizations
    st.header("Detailed Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Confidence Analysis", 
        "Issue Analysis", 
        "Evaluator Performance",
        "Score Distribution",
        "Uncertainty Analysis"
    ])
    
    with tab1:
        st.subheader("Confidence Analysis")
        
        if 'confidence_analysis' in summary:
            conf_analysis = summary['confidence_analysis']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("High Confidence", f"{conf_analysis.get('high_confidence_rate', 0):.1%}")
            with col2:
                st.metric("Medium Confidence", f"{conf_analysis.get('medium_confidence_rate', 0):.1%}")
            with col3:
                st.metric("Low Confidence", f"{conf_analysis.get('low_confidence_rate', 0):.1%}")
        
        fig1 = create_confidence_distribution(results['results'])
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_score_vs_confidence(results['results'])
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Issue Analysis")
        
        fig = create_issue_severity_trend(results['results'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Issue breakdown table
        if 'issue_analysis' in summary:
            st.write("### Issue Breakdown")
            
            issue_data = []
            for severity, count in summary['issue_analysis'].get('by_severity', {}).items():
                issue_data.append({'Severity': severity.capitalize(), 'Count': count})
            
            issue_df = pd.DataFrame(issue_data)
            if not issue_df.empty:
                fig = px.pie(issue_df, values='Count', names='Severity', 
                           title='Issues by Severity',
                           color_discrete_map={
                               'Critical': '#d62728',
                               'High': '#ff7f0e',
                               'Medium': '#ffbb00',
                               'Low': '#2ca02c'
                           })
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Evaluator Performance")
        
        fig = create_evaluator_performance_heatmap(summary)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed evaluator table
        st.write("### Evaluator Details")
        
        eval_data = []
        for eval_name, eval_summary in summary['evaluators'].items():
            eval_data.append({
                'Evaluator': eval_name,
                'Avg Score': f"{eval_summary['average_score']:.3f}",
                'Avg Confidence': f"{eval_summary.get('average_confidence', 0):.3f}",
                'High Conf Rate': f"{eval_summary.get('high_confidence_rate', 0):.1%}",
                'Total Issues': eval_summary['total_issues_found'],
                'Critical Issues': eval_summary.get('issues_by_severity', {}).get('critical', 0)
            })
        
        eval_df = pd.DataFrame(eval_data)
        st.dataframe(eval_df, use_container_width=True)
    
    with tab4:
        st.subheader("Score Distribution")
        
        # Create box plots for all evaluators
        score_data = []
        for result in results['results']:
            for eval_name, eval_result in result['evaluations'].items():
                score_data.append({
                    'Evaluator': eval_name,
                    'Score': eval_result['score'],
                    'Note ID': result['note_id']
                })
        
        score_df = pd.DataFrame(score_data)
        
        if not score_df.empty:
            fig = px.box(score_df, x='Evaluator', y='Score', 
                        title='Score Distribution by Evaluator',
                        points='all')
            fig.update_layout(height=500, xaxis={'tickangle': -45})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Uncertainty Analysis")
        
        fig = create_uncertainty_analysis(results['results'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Uncertainty and variance metrics help assess the reliability of confidence scores. "
                "Lower uncertainty indicates more reliable evaluations.")
    
    # Individual Note Analysis
    st.header("Individual Note Analysis")
    
    note_ids = [r['note_id'] for r in results['results']]
    selected_note = st.selectbox("Select Note to Analyze", note_ids)
    
    if selected_note:
        note_result = next(r for r in results['results'] if r['note_id'] == selected_note)
        display_detailed_note_analysis(note_result)
    
    # Download section
    st.header("Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Download JSON Results",
            data=json.dumps(results, indent=2),
            file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        csv_file = selected_file.with_suffix('.csv')
        if csv_file.exists():
            with open(csv_file, 'r') as f:
                csv_data = f.read()
            st.download_button(
                label="Download CSV Summary",
                data=csv_data,
                file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()

