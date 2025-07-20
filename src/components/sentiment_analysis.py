#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentiment analysis components for the VC Software Reddit Dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from plotly.subplots import make_subplots
import numpy as np

def extract_ai_summary_tags(df, software_list):
    """
    Extract and count AI summary tags for the selected software.
    
    Parameters:
    df: DataFrame containing the data
    software_list: List of software to analyze
    
    Returns:
    DataFrame with tag counts for each software
    """
    # Define the tags we want to extract
    tags = [
        "complains about [software] functionality",
        "complains about [software] pricing",
        "describes [software] positively",
        "asks for alternative to [software]",
        "recommends/suggests [software]",
        "used [software] to report findings",
        "considering [software] for use"
    ]
    
    # Create a dictionary to store the results
    results = {tag: [] for tag in tags}
    results['Software'] = []
    
    # Process each software
    for software in software_list:
        results['Software'].append(software)
        
        # Filter for rows mentioning this software
        software_df = df[df['matched_terms'].str.lower().str.contains(software.lower(), na=False)]
        
        # For each tag, count occurrences for this software
        for tag in tags:
            # Replace [software] with the actual software name for pattern matching
            pattern = tag.replace('[software]', software.lower())
            
            # Count how many AI summaries contain this pattern
            count = 0
            for summary in software_df['AI_Summary'].dropna():
                if isinstance(summary, str) and pattern.lower() in summary.lower():
                    count += 1
            
            results[tag].append(count)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df

def display_ai_summary_breakdown(df, software_list):
    """
    Display a breakdown of AI Summary tags as grouped bar charts with reasons on x-axis.
    
    Parameters:
    df: DataFrame containing the data
    software_list: List of software to analyze
    """
    st.subheader("AI Summary Analysis")
    st.markdown('<div class="chart-description">Breakdown of AI-generated sentiment tags for mentions of selected software.</div>', unsafe_allow_html=True)
    
    if not software_list:
        st.warning("No software selected to display.")
        return
    
    # Extract AI summary tags for all software combined
    tag_counts_df = extract_ai_summary_tags(df, software_list)
    
    # Check if tag_counts is empty or None
    if tag_counts_df is None or (hasattr(tag_counts_df, 'empty') and tag_counts_df.empty) or len(tag_counts_df) == 0:
        st.warning("No AI summary data found for the selected software.")
        return
    
    # Convert the DataFrame to a format suitable for plotting
    tag_columns = [col for col in tag_counts_df.columns if col != 'Software']
    
    # Create a list to store all tag counts (including zero counts for consistent display)
    all_tags = []
    for _, row in tag_counts_df.iterrows():
        software = row['Software']
        for tag in tag_columns:
            count = row[tag]
            # Include all tags, even with zero counts
            all_tags.append({'Tag': tag, 'Count': count, 'Software': software})
    
    if not all_tags:
        st.warning("No AI summary data found for the selected software.")
        return
    
    # Create DataFrame for plotting
    tags_df = pd.DataFrame(all_tags)
    
    # Map tags to clearer labels and sentiment
    tag_mapping = {
        'describes [software] positively': ('Describes this software positively', 'positive'),
        'recommends/suggests [software]': ('Recommends/suggests this software', 'positive'),
        'used [software] to report findings': ('Used this software to report findings', 'positive'),
        'considering [software] for use': ('Considering/trialling this software', 'positive'),
        'complains about [software] functionality': ('Complains about this software functionality', 'negative'),
        'complains about [software] pricing': ('Complains about this software pricing', 'negative'),
        'asks for alternative to [software]': ('Seeking alternative for this software', 'negative')
    }
    
    # Update tags with clearer labels and add sentiment classification
    tags_df['Clear_Label'] = tags_df['Tag'].map(lambda x: tag_mapping.get(x, (x, 'neutral'))[0])
    tags_df['Sentiment'] = tags_df['Tag'].map(lambda x: tag_mapping.get(x, (x, 'neutral'))[1])
    
    # Split into positive and negative sentiment
    positive_tags = tags_df[tags_df['Sentiment'] == 'positive']
    negative_tags = tags_df[tags_df['Sentiment'] == 'negative']
    
    # Create consistent color mapping for software across both charts
    unique_software = tags_df['Software'].unique()
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_mapping = {software: color_palette[i % len(color_palette)] for i, software in enumerate(unique_software)}
    
    # Define all possible sentiment labels for consistent x-axis
    all_positive_labels = [
        'Describes this software positively',
        'Recommends/suggests this software', 
        'Used this software to report findings',
        'Considering/trialling this software'
    ]
    
    all_negative_labels = [
        'Complains about this software functionality',
        'Complains about this software pricing',
        'Seeking alternative for this software'
    ]
    
    # Positive sentiment chart (full row)
    fig_positive = px.bar(
        positive_tags,
        x='Clear_Label',
        y='Count',
        color='Software',
        barmode='group',
        title="Positive Sentiment",
        labels={'Count': 'Number of Mentions', 'Clear_Label': 'Sentiment Tag', 'Software': 'Software'},
        color_discrete_map=color_mapping
    )
    
    # Ensure all positive sentiment labels are shown on x-axis
    fig_positive.update_layout(
        height=500,
        xaxis_title="Sentiment Tag",
        yaxis_title="Number of Mentions",
        legend_title="Software",
        xaxis={'categoryorder': 'array', 'categoryarray': all_positive_labels},
        showlegend=True
    )
    
    fig_positive.update_xaxes(tickangle=45)
    st.plotly_chart(fig_positive, use_container_width=True)

    # Negative sentiment chart (full row)
    fig_negative = px.bar(
        negative_tags,
        x='Clear_Label',
        y='Count',
        color='Software',
        barmode='group',
        title="Negative Sentiment",
        labels={'Count': 'Number of Mentions', 'Clear_Label': 'Sentiment Tag', 'Software': 'Software'},
        color_discrete_map=color_mapping
    )
    
    # Ensure all negative sentiment labels are shown on x-axis
    fig_negative.update_layout(
        height=500,
        xaxis_title="Sentiment Tag",
        yaxis_title="Number of Mentions",
        legend_title="Software",
        xaxis={'categoryorder': 'array', 'categoryarray': all_negative_labels},
        showlegend=True
    )
    
    fig_negative.update_xaxes(tickangle=45)
    st.plotly_chart(fig_negative, use_container_width=True)

def display_sentiment_breakdown(df, software_list):
    """
    Display a breakdown of sentiment tags split into positive and negative charts.
    
    Parameters:
    df: DataFrame containing the data
    software_list: List of software to analyze
    """
    st.subheader("Sentiment Analysis")
    st.markdown('<div class="chart-description">Breakdown of sentiment tags for mentions of selected software.</div>', unsafe_allow_html=True)
    
    if not software_list:
        st.warning("No software selected to display.")
        return
    
    # Extract AI summary tags for all software combined
    tag_counts_df = extract_ai_summary_tags(df, software_list)
    
    # Check if tag_counts is empty or None
    if tag_counts_df is None or (hasattr(tag_counts_df, 'empty') and tag_counts_df.empty) or len(tag_counts_df) == 0:
        st.warning("No sentiment data found for the selected software.")
        return
    
    # Convert the DataFrame to a format suitable for plotting
    tag_columns = [col for col in tag_counts_df.columns if col != 'Software']
    
    # Create a list to store all tag counts (including zero counts for consistent display)
    all_tags = []
    for _, row in tag_counts_df.iterrows():
        software = row['Software']
        for tag in tag_columns:
            count = row[tag]
            # Include all tags, even with zero counts
            all_tags.append({'Tag': tag, 'Count': count, 'Software': software})
    
    if not all_tags:
        st.warning("No sentiment data found for the selected software.")
        return
    
    # Create DataFrame for plotting
    tags_df = pd.DataFrame(all_tags)
    
    # Map tags to clearer labels and sentiment
    tag_mapping = {
        'describes [software] positively': ('Describes this software positively', 'positive'),
        'recommends/suggests [software]': ('Recommends/suggests this software', 'positive'),
        'used [software] to report findings': ('Used this software to report findings', 'positive'),
        'considering [software] for use': ('Considering/trialling this software', 'positive'),
        'complains about [software] functionality': ('Complains about this software functionality', 'negative'),
        'complains about [software] pricing': ('Complains about this software pricing', 'negative'),
        'asks for alternative to [software]': ('Seeking alternative for this software', 'negative')
    }
    
    # Update tags with clearer labels and add sentiment classification
    tags_df['Clear_Label'] = tags_df['Tag'].map(lambda x: tag_mapping.get(x, (x, 'neutral'))[0])
    tags_df['Sentiment'] = tags_df['Tag'].map(lambda x: tag_mapping.get(x, (x, 'neutral'))[1])
    
    # Split into positive and negative sentiment
    positive_tags = tags_df[tags_df['Sentiment'] == 'positive']
    negative_tags = tags_df[tags_df['Sentiment'] == 'negative']
    
    # Create consistent color mapping for software across both charts
    unique_software = tags_df['Software'].unique()
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_mapping = {software: color_palette[i % len(color_palette)] for i, software in enumerate(unique_software)}
    
    # Define all possible sentiment labels for consistent x-axis
    all_positive_labels = [
        'Describes this software positively',
        'Recommends/suggests this software', 
        'Used this software to report findings',
        'Considering/trialling this software'
    ]
    
    all_negative_labels = [
        'Complains about this software functionality',
        'Complains about this software pricing',
        'Seeking alternative for this software'
    ]
    
    # Positive sentiment chart (full row)
    fig_positive = px.bar(
        positive_tags,
        x='Clear_Label',
        y='Count',
        color='Software',
        barmode='group',
        title="Positive Sentiment",
        labels={'Count': 'Number of Mentions', 'Clear_Label': 'Sentiment Tag', 'Software': 'Software'},
        color_discrete_map=color_mapping
    )
    
    # Ensure all positive sentiment labels are shown on x-axis
    fig_positive.update_layout(
        height=500,
        xaxis_title="Sentiment Tag",
        yaxis_title="Number of Mentions",
        legend_title="Software",
        xaxis={'categoryorder': 'array', 'categoryarray': all_positive_labels},
        showlegend=True
    )
    
    fig_positive.update_xaxes(tickangle=45)
    st.plotly_chart(fig_positive, use_container_width=True)

    # Negative sentiment chart (full row)
    fig_negative = px.bar(
        negative_tags,
        x='Clear_Label',
        y='Count',
        color='Software',
        barmode='group',
        title="Negative Sentiment",
        labels={'Count': 'Number of Mentions', 'Clear_Label': 'Sentiment Tag', 'Software': 'Software'},
        color_discrete_map=color_mapping
    )
    
    # Ensure all negative sentiment labels are shown on x-axis
    fig_negative.update_layout(
        height=500,
        xaxis_title="Sentiment Tag",
        yaxis_title="Number of Mentions",
        legend_title="Software",
        xaxis={'categoryorder': 'array', 'categoryarray': all_negative_labels},
        showlegend=True
    )
    
    fig_negative.update_xaxes(tickangle=45)
    st.plotly_chart(fig_negative, use_container_width=True)

def display_sentiment_stacked_bar(df, software_list):
    """
    Display a 100% stacked bar chart showing the breakdown of positive vs negative sentiment for each software.
    
    Parameters:
    df: DataFrame containing the data
    software_list: List of software to analyze
    """
    st.subheader("Positive vs Negative Sentiment Breakdown")
    st.markdown('<div class="chart-description">Shows the percentage breakdown between positive and negative mentions for each software. Negative mentions include complaints about functionality/pricing and requests for alternatives. Positive mentions include positive descriptions, recommendations, usage reports, and consdering use.</div>', unsafe_allow_html=True)
    
    if not software_list:
        st.warning("No software selected to display.")
        return
    
    # Define negative tags
    negative_tags = [
        "complains about [software] functionality",
        "complains about [software] pricing",
        "asks for alternative to [software]"
    ]
    
    # Define positive tags
    positive_tags = [
        "describes [software] positively",
        "recommends/suggests [software]",
        "used [software] to report findings",
        "evaluates [software] for use"
    ]
    
    # Create a dictionary to store results
    results = {
        'Software': [],
        'Positive': [],
        'Negative': [],
        'Total': []
    }
    
    # Process each software
    for software in software_list:
        results['Software'].append(software)
        
        # Filter for posts mentioning this software
        software_df = df[df['matched_terms'].str.lower().str.contains(software.lower(), na=False)]
        
        positive_count = 0
        negative_count = 0
        
        for _, row in software_df.iterrows():
            if pd.notna(row['AI_Summary']):
                summary = row['AI_Summary'].lower()
                
                # Check for negative tags
                is_negative = False
                for tag in negative_tags:
                    pattern = tag.replace('[software]', software.lower())
                    if pattern.lower() in summary:
                        negative_count += 1
                        is_negative = True
                        break
                
                # Check for positive tags if not already classified as negative
                if not is_negative:
                    for tag in positive_tags:
                        pattern = tag.replace('[software]', software.lower())
                        if pattern.lower() in summary:
                            positive_count += 1
                            break
        
        total = positive_count + negative_count
        results['Positive'].append(positive_count)
        results['Negative'].append(negative_count)
        results['Total'].append(total)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Calculate percentages
    result_df['Positive_Pct'] = result_df['Positive'] / result_df['Total'] * 100
    result_df['Negative_Pct'] = result_df['Negative'] / result_df['Total'] * 100
    
    # Replace NaN with 0
    result_df = result_df.fillna(0)
    
    # Sort by total mentions
    result_df = result_df.sort_values('Total', ascending=False)
    
    # Create the stacked bar chart
    fig = go.Figure()
    
    # Add positive percentage bars
    fig.add_trace(go.Bar(
        y=result_df['Software'],
        x=result_df['Positive_Pct'],
        orientation='h',
        name='Positive',
        marker=dict(color='#4CAF50'),
        hovertemplate='%{y}<br>Positive: %{customdata[0]} (%{x:.1f}%)<extra></extra>',
        customdata=result_df[['Positive']].values
    ))
    
    # Add negative percentage bars
    fig.add_trace(go.Bar(
        y=result_df['Software'],
        x=result_df['Negative_Pct'],
        orientation='h',
        name='Negative',
        marker=dict(color='#F44336'),
        hovertemplate='%{y}<br>Negative: %{customdata[0]} (%{x:.1f}%)<extra></extra>',
        customdata=result_df[['Negative']].values
    ))
    
    # Update layout for 100% stacked bars
    fig.update_layout(
        title="Sentiment Breakdown by Software",
        barmode='stack',
        xaxis=dict(
            title="Percentage",
            ticksuffix="%"
        ),
        yaxis=dict(
            title="Software",
            autorange="reversed"  # To match the order of the heatmap
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add legend explaining positive vs negative
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 10px;">
        <b>Sentiment Classification:</b><br>
        <span style="color: #F44336;">Negative Tags:</span> Complaints about functionality, complaints about pricing, requests for alternatives<br>
        <span style="color: #4CAF50;">Positive Tags:</span> Positive descriptions, recommendations, usage reports, evaluations
    </div>
    """, unsafe_allow_html=True)

def display_sentiment_over_time(df, software_list, time_period="Half Year"):
    """
    Display positive and negative sentiment over time as a line chart with positive values
    above the x-axis and negative values reflected below.
    
    Parameters:
    df: DataFrame containing the data
    software_list: List of software to analyze
    time_period: Time period for grouping data
    """
    st.subheader("Positive vs Negative Sentiment Over Time")
    st.markdown('<div class="chart-description">Shows the count of positive and negative mentions for each software over time. Positive values are shown above the x-axis, while negative values are reflected below the x-axis.</div>', unsafe_allow_html=True)
    
    if not software_list:
        st.warning("No software selected to display.")
        return
    
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Group by time period
    if time_period == "Monthly":
        df['time_period'] = df['created_at'].dt.strftime('%Y-%m')
        x_title = "Month"
    elif time_period == "Quarterly":
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-Q' + ((df['created_at'].dt.month - 1) // 3 + 1).astype(str)
        x_title = "Quarter"
    elif time_period == "Third Year":
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-T' + ((df['created_at'].dt.month - 1) // 4 + 1).astype(str)
        x_title = "Third of Year"
    elif time_period == "Half Year":
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-H' + ((df['created_at'].dt.month - 1) // 6 + 1).astype(str)
        x_title = "Half Year"
    elif time_period == "Yearly":
        df['time_period'] = df['created_at'].dt.year.astype(str)
        x_title = "Year"
    
    # Get time periods and sort chronologically
    time_periods_list = df['time_period'].unique().tolist()
    
    # Sort time periods chronologically
    if time_period == "Monthly":
        time_periods = sorted(time_periods_list, key=lambda x: pd.to_datetime(x + "-01"))
    elif time_period == "Quarterly":
        time_periods = sorted(time_periods_list, key=lambda x: (int(x.split('-')[0]), int(x.split('-Q')[1])))
    elif time_period == "Third Year":
        time_periods = sorted(time_periods_list, key=lambda x: (int(x.split('-')[0]), int(x.split('-T')[1])))
    elif time_period == "Half Year":
        time_periods = sorted(time_periods_list, key=lambda x: (int(x.split('-')[0]), int(x.split('-H')[1])))
    elif time_period == "Yearly":
        time_periods = sorted(time_periods_list)
    
    # Define negative tags
    negative_tags = [
        "complains about [software] functionality",
        "complains about [software] pricing",
        "asks for alternative to [software]"
    ]
    
    # Define positive tags
    positive_tags = [
        "describes [software] positively",
        "recommends/suggests [software]",
        "used [software] to report findings",
        "considering [software] for use"
    ]
    
    # Create a figure
    fig = go.Figure()
    
    # Track max count for y-axis scaling
    max_count = 0
    
    # Process each software
    for software in software_list:
        # Create dictionaries to store positive and negative counts
        positive_counts = {period: 0 for period in time_periods}
        negative_counts = {period: 0 for period in time_periods}
        
        # Filter for posts mentioning this software
        software_df = df[df['matched_terms'].str.lower().str.contains(software.lower(), na=False)]
        
        for period in time_periods:
            period_df = software_df[software_df['time_period'] == period]
            
            for _, row in period_df.iterrows():
                if pd.notna(row['AI_Summary']):
                    summary = row['AI_Summary'].lower()
                    
                    # Check for negative tags
                    is_negative = False
                    for tag in negative_tags:
                        pattern = tag.replace('[software]', software.lower())
                        if pattern.lower() in summary:
                            negative_counts[period] += 1
                            is_negative = True
                            break
                    
                    # Check for positive tags if not already classified as negative
                    if not is_negative:
                        for tag in positive_tags:
                            pattern = tag.replace('[software]', software.lower())
                            if pattern.lower() in summary:
                                positive_counts[period] += 1
                                break
        
        # Update max count
        max_count = max(max_count, max(list(positive_counts.values()) + [0]))
        max_count = max(max_count, max(list(negative_counts.values()) + [0]))
        
        # Add positive trace
        fig.add_trace(go.Scatter(
            x=time_periods,
            y=list(positive_counts.values()),
            mode='lines+markers',
            name=f"{software} (Positive)",
            line=dict(color=px.colors.qualitative.Plotly[software_list.index(software) % len(px.colors.qualitative.Plotly)]),
            marker=dict(symbol='circle', size=8),
            hovertemplate=f"{software}<br>%{{x}}<br>Positive: %{{y}}<extra></extra>"
        ))
        
        # Add negative trace (as a reflection)
        fig.add_trace(go.Scatter(
            x=time_periods,
            y=[-count for count in negative_counts.values()],
            mode='lines+markers',
            name=f"{software} (Negative)",
            line=dict(
                color=px.colors.qualitative.Plotly[software_list.index(software) % len(px.colors.qualitative.Plotly)],
                dash='dash'
            ),
            marker=dict(symbol='square', size=8),
            hovertemplate=f"{software}<br>%{{x}}<br>Negative: %{{customdata}}<extra></extra>",
            customdata=list(negative_counts.values())
        ))
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=time_periods[0],
        x1=time_periods[-1],
        y0=0,
        y1=0,
        line=dict(color="black", width=2),
        layer="below"
    )
    
    # Update layout
    fig.update_layout(
        height=800,  # Increase height for better visualization
        title=f"Positive vs Negative Sentiment Over Time ({time_period})",
        xaxis_title=x_title,
        yaxis_title="Number of Mentions",
        yaxis=dict(
            range=[-max_count * 1.1, max_count * 1.1],  # Add 10% padding
            tickvals=list(range(-max_count, max_count + 1, max(1, max_count // 5))),
            ticktext=[str(abs(val)) for val in range(-max_count, max_count + 1, max(1, max_count // 5))],
            # Add more space between positive and negative areas
            domain=[0.05, 0.95]  # Compress the y-axis domain to create more separation
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                x=0.5,
                y=max_count * 0.9,
                xref="paper",
                yref="y",
                text="Positive Mentions",
                showarrow=False,
                font=dict(color="#4CAF50", size=14)
            ),
            dict(
                x=0.5,
                y=-max_count * 0.9,
                xref="paper",
                yref="y",
                text="Negative Mentions",
                showarrow=False,
                font=dict(color="#F44336", size=14)
            )
        ]
    )
    
    # Add a second x-axis at the bottom for negative values
    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="bottom",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[time_periods[0], time_periods[-1]]
        ),
        shapes=[
            # Line separating positive and negative sections
            dict(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=0,
                y1=0,
                line=dict(color="black", width=2, dash="solid"),
            ),
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_exposure_over_time(df, software_list, time_period="Half Year"):
    """
    Display average exposure (score) over time with positive and negative sentiment split.
    
    Parameters:
    df: DataFrame containing the data
    software_list: List of software to analyze
    time_period: Time period for grouping data
    """
    st.subheader("Average Exposure Over Time")
    st.markdown('<div class="chart-description">Shows the average exposure (post score) for positive and negative mentions over time. Positive sentiment is shown above the x-axis, negative sentiment below.</div>', unsafe_allow_html=True)
    
    if not software_list:
        st.warning("No software selected to display.")
        return
    
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Group by time period
    if time_period == "Monthly":
        df['time_period'] = df['created_at'].dt.strftime('%Y-%m')
        x_title = "Month"
    elif time_period == "Quarterly":
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-Q' + ((df['created_at'].dt.month - 1) // 3 + 1).astype(str)
        x_title = "Quarter"
    elif time_period == "Third Year":
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-T' + ((df['created_at'].dt.month - 1) // 4 + 1).astype(str)
        x_title = "Third of Year"
    elif time_period == "Half Year":
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-H' + ((df['created_at'].dt.month - 1) // 6 + 1).astype(str)
        x_title = "Half Year"
    elif time_period == "Yearly":
        df['time_period'] = df['created_at'].dt.year.astype(str)
        x_title = "Year"
    
    # Get time periods and sort chronologically
    time_periods_list = df['time_period'].unique().tolist()
    
    # Sort time periods chronologically
    if time_period == "Monthly":
        time_periods = sorted(time_periods_list, key=lambda x: pd.to_datetime(x + "-01"))
    elif time_period == "Quarterly":
        time_periods = sorted(time_periods_list, key=lambda x: (int(x.split('-')[0]), int(x.split('-Q')[1])))
    elif time_period == "Third Year":
        time_periods = sorted(time_periods_list, key=lambda x: (int(x.split('-')[0]), int(x.split('-T')[1])))
    elif time_period == "Half Year":
        time_periods = sorted(time_periods_list, key=lambda x: (int(x.split('-')[0]), int(x.split('-H')[1])))
    elif time_period == "Yearly":
        time_periods = sorted(time_periods_list)
    
    # Define negative and positive tags with clearer labels
    negative_tags = [
        "complains about [software] functionality",
        "complains about [software] pricing", 
        "seeking alternative for [software]"
    ]
    
    positive_tags = [
        "describes [software] positively",
        "recommends/suggests [software]",
        "used [software] to report findings",
        "considering [software] for use"
    ]
    
    # Create a figure
    fig = go.Figure()
    
    # Track max exposure for y-axis scaling
    max_exposure = 0
    
    # Process each software
    for software in software_list:
        # Create dictionaries to store positive and negative average exposures
        positive_exposures = {period: [] for period in time_periods}
        negative_exposures = {period: [] for period in time_periods}
        
        # Filter for posts mentioning this software
        software_df = df[df['matched_terms'].str.lower().str.contains(software.lower(), na=False)]
        
        for period in time_periods:
            period_df = software_df[software_df['time_period'] == period]
            
            for _, row in period_df.iterrows():
                if pd.notna(row['AI_Summary']):
                    summary = row['AI_Summary'].lower()
                    score = row.get('score', 0)
                    
                    # Check for negative tags
                    is_negative = False
                    for tag in negative_tags:
                        pattern = tag.replace('[software]', software.lower())
                        if pattern.lower() in summary:
                            negative_exposures[period].append(score)
                            is_negative = True
                            break
                    
                    # Check for positive tags if not already classified as negative
                    if not is_negative:
                        for tag in positive_tags:
                            pattern = tag.replace('[software]', software.lower())
                            if pattern.lower() in summary:
                                positive_exposures[period].append(score)
                                break
        
        # Calculate average exposures
        positive_avg = []
        negative_avg = []
        
        for period in time_periods:
            pos_avg = np.mean(positive_exposures[period]) if positive_exposures[period] else 0
            neg_avg = np.mean(negative_exposures[period]) if negative_exposures[period] else 0
            positive_avg.append(pos_avg)
            negative_avg.append(neg_avg)
        
        # Update max exposure
        max_exposure = max(max_exposure, max(positive_avg + [0]))
        max_exposure = max(max_exposure, max(negative_avg + [0]))
        
        # Add positive trace
        fig.add_trace(go.Scatter(
            x=time_periods,
            y=positive_avg,
            mode='lines+markers',
            name=f"{software} (Positive)",
            line=dict(color=px.colors.qualitative.Plotly[software_list.index(software) % len(px.colors.qualitative.Plotly)]),
            marker=dict(symbol='circle', size=8),
            hovertemplate=f"{software}<br>%{{x}}<br>Avg Positive Exposure: %{{y:.1f}}<extra></extra>"
        ))
        
        # Add negative trace (as a reflection below x-axis)
        fig.add_trace(go.Scatter(
            x=time_periods,
            y=[-avg for avg in negative_avg],
            mode='lines+markers',
            name=f"{software} (Negative)",
            line=dict(
                color=px.colors.qualitative.Plotly[software_list.index(software) % len(px.colors.qualitative.Plotly)],
                dash='dash'
            ),
            marker=dict(symbol='square', size=8),
            hovertemplate=f"{software}<br>%{{x}}<br>Avg Negative Exposure: %{{customdata:.1f}}<extra></extra>",
            customdata=negative_avg
        ))
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=time_periods[0],
        x1=time_periods[-1],
        y0=0,
        y1=0,
        line=dict(color="black", width=2),
        layer="below"
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title=f"Average Exposure Over Time ({time_period})",
        xaxis_title=x_title,
        yaxis_title="Average Exposure (Score)",
        yaxis=dict(
            range=[-max_exposure * 1.1, max_exposure * 1.1],
            domain=[0.05, 0.95]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                x=0.5,
                y=max_exposure * 0.9,
                xref="paper",
                yref="y",
                text="Positive Exposure",
                showarrow=False,
                font=dict(color="#4CAF50", size=14)
            ),
            dict(
                x=0.5,
                y=-max_exposure * 0.9,
                xref="paper",
                yref="y",
                text="Negative Exposure", 
                showarrow=False,
                font=dict(color="#F44336", size=14)
            )
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_exposure_diverging_chart(df, software_list):
    """
    Display a diverging bar chart showing positive vs negative exposure (weighted by post score).
    
    Parameters:
    df: DataFrame containing the data
    software_list: List of software to analyze
    """
    st.subheader("Positive vs Negative Exposure")
    st.markdown('<div class="chart-description">Shows the exposure (sum of post scores) for each software, split between positive and negative mentions. Longer bars indicate higher total exposure.</div>', unsafe_allow_html=True)
    
    if not software_list:
        st.warning("No software selected to display.")
        return
    
    # Define negative tags
    negative_tags = [
        "complains about [software] functionality",
        "complains about [software] pricing",
        "asks for alternative to [software]"
    ]
    
    # Define positive tags
    positive_tags = [
        "describes [software] positively",
        "recommends/suggests [software]",
        "used [software] to report findings",
        "evaluates [software] for use"
    ]
    
    # Create a dictionary to store results
    results = {
        'Software': [],
        'Positive_Exposure': [],
        'Negative_Exposure': []
    }
    
    # Process each software
    for software in software_list:
        results['Software'].append(software)
        
        # Filter for posts mentioning this software
        software_df = df[df['matched_terms'].str.lower().str.contains(software.lower(), na=False)]
        
        positive_exposure = 0
        negative_exposure = 0
        
        for _, row in software_df.iterrows():
            if pd.notna(row['AI_Summary']) and pd.notna(row['score']):
                summary = row['AI_Summary'].lower()
                score = row['score']
                
                # Check for negative tags
                is_negative = False
                for tag in negative_tags:
                    pattern = tag.replace('[software]', software.lower())
                    if pattern.lower() in summary:
                        negative_exposure += score
                        is_negative = True
                        break
                
                # Check for positive tags if not already classified as negative
                if not is_negative:
                    for tag in positive_tags:
                        pattern = tag.replace('[software]', software.lower())
                        if pattern.lower() in summary:
                            positive_exposure += score
                            break
        
        results['Positive_Exposure'].append(positive_exposure)
        results['Negative_Exposure'].append(-negative_exposure)  # Make negative for diverging chart
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Sort by total exposure (absolute sum of positive and negative)
    result_df['Total_Exposure'] = result_df['Positive_Exposure'] + result_df['Negative_Exposure'].abs()
    result_df = result_df.sort_values('Total_Exposure', ascending=False)
    
    # Create the diverging bar chart
    fig = go.Figure()
    
    # Add negative exposure bars
    fig.add_trace(go.Bar(
        y=result_df['Software'],
        x=result_df['Negative_Exposure'],
        orientation='h',
        name='Negative Exposure',
        marker=dict(color='#F44336'),
        hovertemplate='%{y}<br>Negative Exposure: %{x:.0f}<extra></extra>'
    ))
    
    # Add positive exposure bars
    fig.add_trace(go.Bar(
        y=result_df['Software'],
        x=result_df['Positive_Exposure'],
        orientation='h',
        name='Positive Exposure',
        marker=dict(color='#4CAF50'),
        hovertemplate='%{y}<br>Positive Exposure: %{x:.0f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title="Software Exposure: Positive vs Negative (Weighted by Post Score)",
        barmode='relative',
        xaxis=dict(
            title="Exposure (Post Score Sum)",
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        yaxis=dict(
            title="Software",
            autorange="reversed"  # To match the order of the heatmap
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 10px;">
        <b>Exposure Calculation:</b><br>
        Each bar represents the sum of post scores (upvotes) for a software, split between positive and negative mentions.<br>
        <span style="color: #F44336;">Left (Red):</span> Score sum for posts with negative sentiment<br>
        <span style="color: #4CAF50;">Right (Green):</span> Score sum for posts with positive sentiment
    </div>
    """, unsafe_allow_html=True) 