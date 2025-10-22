"""Enhanced interactive dashboard with real-time monitoring and advanced analytics."""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="DeepScribe Enhanced Evaluation Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-warning {
        background-color: #ffbb33;
        color: black;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-info {
        background-color: #33b5e5;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def check_for_alerts(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check for critical issues and generate alerts."""
    alerts = []
    
    # Check for critical issues
    for result in results.get('results', []):
        note_id = result['note_id']
        
        for eval_name, eval_result in result['evaluations'].items():
            # Alert on low scores
            score = eval_result.get('score', 1.0)
            if score < 0.3:
                alerts.append({
                    'severity': 'critical',
                    'title': f'Critical Score: {eval_name}',
                    'message': f'Note {note_id} scored {score:.2f} in {eval_name}',
                    'note_id': note_id,
                    'evaluator': eval_name
                })
            
            # Alert on critical issues
            critical_issues = [
                i for i in eval_result.get('issues', [])
                if i.get('severity') == 'critical'
            ]
            
            if critical_issues:
                alerts.append({
                    'severity': 'critical',
                    'title': f'Critical Issues Found',
                    'message': f'Note {note_id} has {len(critical_issues)} critical issues in {eval_name}',
                    'note_id': note_id,
                    'evaluator': eval_name,
                    'issues': critical_issues
                })
    
    # Check for system-wide issues
    summary = results.get('summary', {})
    overall_stats = summary.get('overall_statistics', {})
    avg_score = overall_stats.get('average_score', 1.0)
    
    if avg_score < 0.5:
        alerts.append({
            'severity': 'warning',
            'title': 'Low Average Score',
            'message': f'System-wide average score is {avg_score:.2f}',
            'note_id': 'ALL',
            'evaluator': 'System'
        })
    
    return alerts


def display_alerts(alerts: List[Dict[str, Any]]):
    """Display alerts in the dashboard."""
    if not alerts:
        st.success("✅ No critical alerts")
        return
    
    critical_alerts = [a for a in alerts if a['severity'] == 'critical']
    warning_alerts = [a for a in alerts if a['severity'] == 'warning']
    
    if critical_alerts:
        st.markdown(f"### 🚨 {len(critical_alerts)} Critical Alerts")
        for alert in critical_alerts:
            with st.expander(f"🔴 {alert['title']}", expanded=True):
                st.write(alert['message'])
                st.write(f"**Note ID:** {alert['note_id']}")
                st.write(f"**Evaluator:** {alert['evaluator']}")
                if 'issues' in alert:
                    for issue in alert['issues']:
                        st.markdown(f"- {issue.get('description', 'N/A')}")
    
    if warning_alerts:
        st.markdown(f"### ⚠️ {len(warning_alerts)} Warnings")
        for alert in warning_alerts:
            with st.expander(f"🟡 {alert['title']}"):
                st.write(alert['message'])


def create_advanced_score_distribution(results_data: List[Dict]) -> go.Figure:
    """Create advanced score distribution with violin plots."""
    scores_by_evaluator = {}
    
    for result in results_data:
        for eval_name, eval_result in result['evaluations'].items():
            if eval_name not in scores_by_evaluator:
                scores_by_evaluator[eval_name] = []
            scores_by_evaluator[eval_name].append(eval_result['score'])
    
    fig = go.Figure()
    
    for eval_name, scores in scores_by_evaluator.items():
        # Add violin plot
        fig.add_trace(go.Violin(
            y=scores,
            name=eval_name,
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightblue',
            opacity=0.6,
            x0=eval_name
        ))
    
    fig.update_layout(
        title="Score Distribution by Evaluator (Violin Plot)",
        yaxis_title="Score",
        xaxis_title="Evaluator",
        height=500,
        showlegend=False
    )
    
    return fig


def create_confidence_vs_score(results_data: List[Dict]) -> go.Figure:
    """Create scatter plot of confidence vs score."""
    data_points = []
    
    for result in results_data:
        note_id = result['note_id']
        for eval_name, eval_result in result['evaluations'].items():
            score = eval_result.get('score', 0)
            confidence = eval_result.get('metrics', {}).get('confidence', 0)
            
            data_points.append({
                'note_id': note_id,
                'evaluator': eval_name,
                'score': score,
                'confidence': confidence
            })
    
    df = pd.DataFrame(data_points)
    
    fig = px.scatter(
        df,
        x='confidence',
        y='score',
        color='evaluator',
        hover_data=['note_id'],
        title='Confidence vs Score Analysis',
        labels={'confidence': 'Confidence', 'score': 'Score'},
        trendline='ols'
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_uncertainty_heatmap(results_data: List[Dict]) -> go.Figure:
    """Create heatmap showing uncertainty metrics."""
    uncertainty_data = []
    note_ids = []
    
    for result in results_data:
        note_id = result['note_id']
        note_ids.append(note_id)
        
        row_data = {}
        for eval_name, eval_result in result['evaluations'].items():
            metrics = eval_result.get('metrics', {})
            
            # Extract uncertainty-related metrics
            uncertainty_score = metrics.get('uncertainty_score', metrics.get('confidence', 0.7))
            row_data[eval_name] = 1 - uncertainty_score  # Invert for "uncertainty"
        
        uncertainty_data.append(row_data)
    
    df = pd.DataFrame(uncertainty_data, index=note_ids)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='RdYlGn_r',  # Red for high uncertainty
        reversescale=False,
        text=df.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Uncertainty")
    ))
    
    fig.update_layout(
        title="Uncertainty Heatmap Across Notes and Evaluators",
        xaxis_title="Evaluator",
        yaxis_title="Note ID",
        height=max(400, len(note_ids) * 25)
    )
    
    return fig


def create_ensemble_agreement_chart(results_data: List[Dict]) -> go.Figure:
    """Create chart showing model agreement in ensemble evaluations."""
    ensemble_data = []
    
    for result in results_data:
        note_id = result['note_id']
        
        for eval_name, eval_result in result['evaluations'].items():
            metrics = eval_result.get('metrics', {})
            
            if 'agreement_score' in metrics:  # Ensemble evaluation
                ensemble_data.append({
                    'note_id': note_id,
                    'evaluator': eval_name,
                    'agreement': metrics['agreement_score'],
                    'uncertainty': metrics.get('uncertainty', 0),
                    'num_models': len(metrics.get('individual_results', []))
                })
    
    if not ensemble_data:
        return go.Figure().add_annotation(
            text="No ensemble evaluation data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
    
    df = pd.DataFrame(ensemble_data)
    
    fig = px.scatter(
        df,
        x='agreement',
        y='uncertainty',
        size='num_models',
        color='evaluator',
        hover_data=['note_id'],
        title='Ensemble Model Agreement vs Uncertainty',
        labels={'agreement': 'Agreement Score', 'uncertainty': 'Uncertainty'}
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_temporal_trends(result_files: List[Path]) -> go.Figure:
    """Create temporal trends if multiple evaluation runs exist."""
    if len(result_files) < 2:
        return None
    
    trends_data = []
    
    for result_file in sorted(result_files)[-10:]:  # Last 10 runs
        try:
            results = load_results(result_file)
            timestamp = results['metadata']['timestamp']
            
            summary = results.get('summary', {})
            overall_stats = summary.get('overall_statistics', {})
            avg_score = overall_stats.get('average_score', 0)
            
            trends_data.append({
                'timestamp': timestamp,
                'average_score': avg_score,
                'num_notes': results['metadata']['num_notes']
            })
        except:
            continue
    
    if not trends_data:
        return None
    
    df = pd.DataFrame(trends_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['average_score'],
        mode='lines+markers',
        name='Average Score',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Evaluation Score Trends Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Average Score',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def display_interpretability_analysis(result: Dict):
    """Display interpretability analysis for a note."""
    st.subheader("🔍 Interpretability Analysis")
    
    for eval_name, eval_result in result['evaluations'].items():
        with st.expander(f"{eval_name} - Decision Explanation"):
            metrics = eval_result.get('metrics', {})
            score = eval_result['score']
            confidence = metrics.get('confidence', 0.7)
            
            # Display decision summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{score:.3f}")
            with col2:
                st.metric("Confidence", f"{confidence:.3f}")
            with col3:
                uncertainty = metrics.get('uncertainty_score', 1 - confidence)
                st.metric("Uncertainty", f"{uncertainty:.3f}")
            
            # Display reasoning steps if available
            if '_uncertainty' in eval_result.get('metrics', {}):
                unc_data = eval_result['metrics']['_uncertainty']
                st.write("**Uncertainty Metrics:**")
                st.json(unc_data)
            
            # Display key factors
            issues = eval_result.get('issues', [])
            if issues:
                st.write("**Key Factors Affecting Score:**")
                
                critical_issues = [i for i in issues if i['severity'] == 'critical']
                high_issues = [i for i in issues if i['severity'] == 'high']
                
                if critical_issues:
                    st.error(f"🔴 {len(critical_issues)} Critical Issues")
                if high_issues:
                    st.warning(f"🟠 {len(high_issues)} High Severity Issues")
                
                # Show issue breakdown
                for issue in issues[:5]:  # Top 5
                    severity_icon = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢'
                    }
                    icon = severity_icon.get(issue['severity'], '⚪')
                    st.markdown(f"{icon} **{issue['type']}**: {issue['description'][:100]}...")


def main():
    """Main enhanced dashboard application."""
    st.title("🏥 DeepScribe Enhanced Evaluation Dashboard")
    st.markdown("*Advanced Analytics, Real-time Monitoring & Interpretability*")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File selector
    results_dir = Path("results")
    if not results_dir.exists():
        st.error("No results directory found. Please run the evaluation pipeline first.")
        st.code("python -m src.pipeline --num-samples 10")
        return
    
    result_files = list(results_dir.glob("evaluation_results_*.json"))
    
    if not result_files:
        st.error("No evaluation results found.")
        return
    
    # Sort by timestamp (newest first)
    result_files = sorted(result_files, reverse=True)
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 30s)", value=False)
    if auto_refresh:
        st.sidebar.info("Dashboard will auto-refresh...")
        time.sleep(30)
        st.rerun()
    
    selected_file = st.sidebar.selectbox(
        "Select Evaluation Results",
        result_files,
        format_func=lambda x: f"{x.name} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})"
    )
    
    if not selected_file:
        return
    
    # Load results
    with st.spinner("Loading results..."):
        results = load_results(selected_file)
    
    # Display metadata
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Evaluation Info")
    st.sidebar.write(f"**Timestamp:** {results['metadata']['timestamp']}")
    st.sidebar.write(f"**Notes:** {results['metadata']['num_notes']}")
    st.sidebar.write(f"**Evaluators:** {results['metadata']['num_evaluators']}")
    
    # Check for alerts
    st.header("🚨 Alert Dashboard")
    alerts = check_for_alerts(results)
    display_alerts(alerts)
    
    st.markdown("---")
    
    # Key Metrics
    st.header("📈 Key Performance Metrics")
    
    summary = results['summary']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Notes", summary['total_notes'])
    
    with col2:
        if 'overall_statistics' in summary and 'average_score' in summary['overall_statistics']:
            avg_score = summary['overall_statistics']['average_score']
            st.metric("Average Score", f"{avg_score:.3f}")
    
    with col3:
        total_issues = sum(
            eval_sum['total_issues_found']
            for eval_sum in summary['evaluators'].values()
        )
        st.metric("Total Issues", total_issues)
    
    with col4:
        critical_count = sum(
            eval_sum.get('issues_by_severity', {}).get('critical', 0)
            for eval_sum in summary['evaluators'].values()
        )
        st.metric("Critical Issues", critical_count, 
                 delta=None if critical_count == 0 else -critical_count,
                 delta_color="inverse")
    
    # Temporal trends
    st.markdown("---")
    st.header("📊 Temporal Analysis")
    
    trends_fig = create_temporal_trends(result_files)
    if trends_fig:
        st.plotly_chart(trends_fig, use_container_width=True)
    else:
        st.info("Run multiple evaluations to see trends over time")
    
    # Advanced Visualizations
    st.markdown("---")
    st.header("📉 Advanced Analytics")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Score Distribution",
        "Confidence Analysis",
        "Uncertainty Heatmap",
        "Ensemble Agreement",
        "Issues Breakdown"
    ])
    
    with tab1:
        fig = create_advanced_score_distribution(results['results'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = create_confidence_vs_score(results['results'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretation:** Points should cluster near the diagonal. 
        - High confidence + Low score: Confidently wrong (needs investigation)
        - Low confidence + High score: Uncertain but correct (model uncertainty)
        """)
    
    with tab3:
        fig = create_uncertainty_heatmap(results['results'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = create_ensemble_agreement_chart(results['results'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        from dashboard import create_issues_breakdown
        fig = create_issues_breakdown(results['results'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Evaluator Performance Table
    st.markdown("---")
    st.header("🎯 Evaluator Performance")
    
    eval_summary_data = []
    for eval_name, eval_summary in summary['evaluators'].items():
        eval_summary_data.append({
            'Evaluator': eval_name,
            'Avg Score': f"{eval_summary['average_score']:.3f}",
            'Min Score': f"{eval_summary['min_score']:.3f}",
            'Max Score': f"{eval_summary['max_score']:.3f}",
            'Std Dev': f"{eval_summary.get('std_dev', 0):.3f}",
            'Total Issues': eval_summary['total_issues_found'],
            'Critical': eval_summary.get('issues_by_severity', {}).get('critical', 0)
        })
    
    st.dataframe(pd.DataFrame(eval_summary_data), use_container_width=True)
    
    # Individual Note Analysis with Interpretability
    st.markdown("---")
    st.header("🔬 Individual Note Analysis")
    
    note_ids = [r['note_id'] for r in results['results']]
    selected_note = st.selectbox("Select Note to Analyze", note_ids)
    
    if selected_note:
        note_result = next(r for r in results['results'] if r['note_id'] == selected_note)
        
        # Display interpretability analysis
        display_interpretability_analysis(note_result)
        
        # Display traditional note details
        from dashboard import display_note_details
        display_note_details(note_result)
    
    # Download section
    st.markdown("---")
    st.header("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📥 Download JSON",
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
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Generate alerts report
        alerts_report = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'alerts': alerts,
            'summary': {
                'total_alerts': len(alerts),
                'critical': len([a for a in alerts if a['severity'] == 'critical']),
                'warnings': len([a for a in alerts if a['severity'] == 'warning'])
            }
        }, indent=2)
        
        st.download_button(
            label="🚨 Download Alerts",
            data=alerts_report,
            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
