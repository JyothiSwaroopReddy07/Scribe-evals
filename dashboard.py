"""Interactive dashboard for visualizing SOAP note evaluation results."""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, List


st.set_page_config(
    page_title="DeepScribe Evaluation Dashboard",
    page_icon="🏥",
    layout="wide"
)


def load_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_score_distribution(results_data: List[Dict]) -> go.Figure:
    """Create score distribution plot."""
    scores_by_evaluator = {}
    
    for result in results_data:
        for eval_name, eval_result in result['evaluations'].items():
            if eval_name not in scores_by_evaluator:
                scores_by_evaluator[eval_name] = []
            scores_by_evaluator[eval_name].append(eval_result['score'])
    
    fig = go.Figure()
    
    for eval_name, scores in scores_by_evaluator.items():
        fig.add_trace(go.Box(
            y=scores,
            name=eval_name,
            boxmean='sd'
        ))
    
    fig.update_layout(
        title="Score Distribution by Evaluator",
        yaxis_title="Score",
        xaxis_title="Evaluator",
        height=400
    )
    
    return fig


def create_issues_breakdown(results_data: List[Dict]) -> go.Figure:
    """Create issues breakdown chart."""
    issue_counts = {}
    
    for result in results_data:
        for eval_name, eval_result in result['evaluations'].items():
            if eval_name not in issue_counts:
                issue_counts[eval_name] = {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0,
                    'info': 0
                }
            
            for issue in eval_result['issues']:
                severity = issue['severity']
                issue_counts[eval_name][severity] = issue_counts[eval_name].get(severity, 0) + 1
    
    # Prepare data for stacked bar chart
    evaluators = list(issue_counts.keys())
    severities = ['critical', 'high', 'medium', 'low', 'info']
    
    fig = go.Figure()
    
    colors = {
        'critical': '#d62728',
        'high': '#ff7f0e',
        'medium': '#ffbb00',
        'low': '#2ca02c',
        'info': '#1f77b4'
    }
    
    for severity in severities:
        counts = [issue_counts[eval][severity] for eval in evaluators]
        fig.add_trace(go.Bar(
            name=severity.capitalize(),
            x=evaluators,
            y=counts,
            marker_color=colors[severity]
        ))
    
    fig.update_layout(
        title="Issues by Severity and Evaluator",
        barmode='stack',
        xaxis_title="Evaluator",
        yaxis_title="Number of Issues",
        height=400
    )
    
    return fig


def create_metrics_heatmap(results_data: List[Dict]) -> go.Figure:
    """Create heatmap of metrics across notes."""
    # Collect all metrics
    all_metrics = {}
    note_ids = []
    
    for result in results_data:
        note_id = result['note_id']
        note_ids.append(note_id)
        
        for eval_name, eval_result in result['evaluations'].items():
            for metric_name, metric_value in eval_result['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    full_name = f"{eval_name}:{metric_name}"
                    if full_name not in all_metrics:
                        all_metrics[full_name] = []
                    all_metrics[full_name].append(metric_value)
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics, index=note_ids)
    
    if df.empty:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='RdYlGn',
        zmid=0.5
    ))
    
    fig.update_layout(
        title="Metrics Heatmap Across Notes",
        xaxis_title="Metric",
        yaxis_title="Note ID",
        height=max(400, len(note_ids) * 20)
    )
    
    return fig


def display_note_details(result: Dict):
    """Display detailed view of a single note evaluation."""
    st.subheader(f"Note: {result['note_id']}")
    
    # Overall scores
    st.write("### Overall Scores")
    scores = {}
    for eval_name, eval_result in result['evaluations'].items():
        scores[eval_name] = eval_result['score']
    
    cols = st.columns(len(scores))
    for i, (eval_name, score) in enumerate(scores.items()):
        with cols[i]:
            st.metric(eval_name, f"{score:.3f}")
    
    # Issues
    st.write("### Issues Found")
    for eval_name, eval_result in result['evaluations'].items():
        if eval_result['issues']:
            with st.expander(f"{eval_name} ({len(eval_result['issues'])} issues)"):
                for issue in eval_result['issues']:
                    severity_color = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢',
                        'info': '🔵'
                    }
                    icon = severity_color.get(issue['severity'], '⚪')
                    
                    st.markdown(f"{icon} **{issue['type']}** ({issue['severity']})")
                    st.write(f"Description: {issue['description']}")
                    if issue.get('location'):
                        st.write(f"Location: {issue['location']}")
                    if issue.get('evidence'):
                        st.json(issue['evidence'])
                    st.divider()
    
    # Metrics
    st.write("### Detailed Metrics")
    for eval_name, eval_result in result['evaluations'].items():
        with st.expander(f"{eval_name} Metrics"):
            metrics_df = pd.DataFrame([eval_result['metrics']]).T
            metrics_df.columns = ['Value']
            st.dataframe(metrics_df)


def main():
    """Main dashboard application."""
    st.title("🏥 DeepScribe SOAP Note Evaluation Dashboard")
    st.markdown("---")
    
    # File selector
    results_dir = Path("results")
    if not results_dir.exists():
        st.error("No results directory found. Please run the evaluation pipeline first.")
        return
    
    result_files = list(results_dir.glob("evaluation_results_*.json"))
    
    if not result_files:
        st.error("No evaluation results found. Please run the evaluation pipeline first.")
        st.code("python -m src.pipeline --num-samples 10")
        return
    
    # Sort by timestamp (newest first)
    result_files = sorted(result_files, reverse=True)
    
    selected_file = st.selectbox(
        "Select Evaluation Results",
        result_files,
        format_func=lambda x: x.name
    )
    
    if not selected_file:
        return
    
    # Load results
    results = load_results(selected_file)
    
    # Display metadata
    st.sidebar.header("Evaluation Info")
    st.sidebar.write(f"**Timestamp:** {results['metadata']['timestamp']}")
    st.sidebar.write(f"**Notes Evaluated:** {results['metadata']['num_notes']}")
    st.sidebar.write(f"**Evaluators:** {results['metadata']['num_evaluators']}")
    st.sidebar.write("**Evaluator Names:**")
    for eval_name in results['metadata']['evaluators']:
        st.sidebar.write(f"- {eval_name}")
    
    # Summary Statistics
    st.header("📊 Summary Statistics")
    
    summary = results['summary']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Notes", summary['total_notes'])
    
    with col2:
        if 'overall_statistics' in summary and 'average_score' in summary['overall_statistics']:
            st.metric("Average Score", f"{summary['overall_statistics']['average_score']:.3f}")
    
    with col3:
        total_issues = sum(
            eval_sum['total_issues_found']
            for eval_sum in summary['evaluators'].values()
        )
        st.metric("Total Issues", total_issues)
    
    # Evaluator-specific summaries
    st.subheader("Evaluator Performance")
    
    eval_summary_data = []
    for eval_name, eval_summary in summary['evaluators'].items():
        eval_summary_data.append({
            'Evaluator': eval_name,
            'Avg Score': f"{eval_summary['average_score']:.3f}",
            'Min Score': f"{eval_summary['min_score']:.3f}",
            'Max Score': f"{eval_summary['max_score']:.3f}",
            'Total Issues': eval_summary['total_issues_found']
        })
    
    st.dataframe(pd.DataFrame(eval_summary_data), use_container_width=True)
    
    # Visualizations
    st.header("📈 Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Issues Breakdown", "Metrics Heatmap"])
    
    with tab1:
        fig = create_score_distribution(results['results'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = create_issues_breakdown(results['results'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = create_metrics_heatmap(results['results'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual Note Analysis
    st.header("🔍 Individual Note Analysis")
    
    note_ids = [r['note_id'] for r in results['results']]
    selected_note = st.selectbox("Select Note to Analyze", note_ids)
    
    if selected_note:
        note_result = next(r for r in results['results'] if r['note_id'] == selected_note)
        display_note_details(note_result)
    
    # Download section
    st.header("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Download JSON Results",
            data=json.dumps(results, indent=2),
            file_name=f"evaluation_results.json",
            mime="application/json"
        )
    
    with col2:
        # Check if CSV exists
        csv_file = selected_file.with_suffix('.csv')
        if csv_file.exists():
            with open(csv_file, 'r') as f:
                csv_data = f.read()
            st.download_button(
                label="Download CSV Summary",
                data=csv_data,
                file_name=f"evaluation_summary.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()

