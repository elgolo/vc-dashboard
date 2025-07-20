#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Software analysis components for the VC Software Reddit Dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.visualization import create_time_series_chart

def display_software_mentions_over_time(df, software_list, time_period="Monthly", key=None):
    """Display software mentions over time as a line chart."""
    st.subheader("Software Mentions Over Time")
    
    if not software_list:
        st.warning("No software selected to display.")
        return
    
    # Create a dictionary to store counts for each software
    software_counts = {}
    
    # Group by time period
    if time_period == "Monthly":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.strftime('%Y-%m')
        x_title = "Month"
    elif time_period == "Quarterly":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-Q' + ((df['created_at'].dt.month - 1) // 3 + 1).astype(str)
        x_title = "Quarter"
    elif time_period == "Third Year":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-T' + ((df['created_at'].dt.month - 1) // 4 + 1).astype(str)
        x_title = "Third of Year"
    elif time_period == "Half Year":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-H' + ((df['created_at'].dt.month - 1) // 6 + 1).astype(str)
        x_title = "Half Year"
    elif time_period == "Yearly":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.year.astype(str)
        x_title = "Year"
    
    # Get all time periods
    all_time_periods = sorted(df['time_period'].unique())
    
    # Count mentions for each software over time
    for software in software_list:
        # Filter for posts mentioning this software
        software_df = df[df['matched_terms'].str.lower().str.contains(software.lower(), na=False)]
        
        # Group by time period and count
        counts = software_df.groupby('time_period').size()
        software_counts[software] = counts
    
    # Get all time periods across all software
    all_time_periods = set()
    for counts in software_counts.values():
        all_time_periods.update(counts.index)
    
    # Create a DataFrame with all time periods and software counts
    # Sort time periods chronologically by parsing the period format
    if time_period == "Monthly":
        time_periods_sorted = sorted(all_time_periods, key=lambda x: pd.to_datetime(x + "-01"))
    elif time_period == "Quarterly":
        time_periods_sorted = sorted(all_time_periods, key=lambda x: (int(x.split('-')[0]), int(x.split('-Q')[1])))
    elif time_period == "Third Year":
        time_periods_sorted = sorted(all_time_periods, key=lambda x: (int(x.split('-')[0]), int(x.split('-T')[1])))
    elif time_period == "Half Year":
        time_periods_sorted = sorted(all_time_periods, key=lambda x: (int(x.split('-')[0]), int(x.split('-H')[1])))
    elif time_period == "Yearly":
        time_periods_sorted = sorted(all_time_periods)
    
    data = {'Time': time_periods_sorted}
    
    for software, counts in software_counts.items():
        data[software] = [counts.get(period, 0) for period in time_periods_sorted]
    
    # Create DataFrame
    df_plot = pd.DataFrame(data)
    
    # Melt the DataFrame for plotting
    df_melted = df_plot.melt(id_vars=['Time'], var_name='Software', value_name='Mentions')
    
    # Create line chart
    fig = px.line(
        df_melted,
        x='Time',
        y='Mentions',
        color='Software',
        markers=True,
        title=f"Software Mentions Over Time ({time_period})"
    )
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Number of Mentions",
        legend_title="Software",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Use the key parameter if provided
    if key:
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key="software_mentions_over_time")

def display_software_sentiment_over_time(df, software_list, time_period="Monthly", key=None):
    """Display software sentiment over time as a line chart."""
    st.subheader("Software Sentiment Over Time")
    
    if not software_list:
        st.warning("No software selected to display.")
        return
    
    # Create a figure with go.Figure() for more control over line connections
    fig = go.Figure()
    
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
    
    # Get all time periods from the dataset and sort chronologically
    time_periods = df['time_period'].unique()
    
    # Sort time periods chronologically
    if time_period == "Monthly":
        all_time_periods = sorted(time_periods, key=lambda x: pd.to_datetime(x + "-01"))
    elif time_period == "Quarterly":
        all_time_periods = sorted(time_periods, key=lambda x: (int(x.split('-')[0]), int(x.split('-Q')[1])))
    elif time_period == "Third Year":
        all_time_periods = sorted(time_periods, key=lambda x: (int(x.split('-')[0]), int(x.split('-T')[1])))
    elif time_period == "Half Year":
        all_time_periods = sorted(time_periods, key=lambda x: (int(x.split('-')[0]), int(x.split('-H')[1])))
    elif time_period == "Yearly":
        all_time_periods = sorted(time_periods)
    
    # Calculate sentiment for each software over time
    for software in software_list:
        # Filter for posts mentioning this software
        software_df = df[df['matched_terms'].str.lower().str.contains(software.lower(), na=False)]
        
        # Group by time period and calculate average sentiment
        sentiment_series = software_df.groupby('time_period')['vader_sentiment_score'].mean().round(2)
        
        # Convert to dictionary for easier access
        sentiment_dict = sentiment_series.to_dict()
        
        # Create x and y values for the line chart, preserving only periods with data
        x_values = []
        y_values = []
        
        for period in all_time_periods:
            if period in sentiment_dict:
                x_values.append(period)
                y_values.append(sentiment_dict[period])
        
        # Add trace if we have data points
        if x_values:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=software,
                connectgaps=True,  # Connect gaps between data points
                line=dict(shape='linear'),  # Linear interpolation between points
                marker=dict(size=8)
            ))
    
    # Add a horizontal line at y=0 to highlight positive vs negative sentiment
    if all_time_periods:
        fig.add_shape(
            type="line",
            x0=all_time_periods[0],
            x1=all_time_periods[-1],
            y0=0,
            y1=0,
            line=dict(color="black", width=2),
            layer="below"
        )
    
    # Update layout
    fig.update_layout(
        title=f"Software Sentiment Over Time ({time_period})",
        xaxis_title=x_title,
        yaxis_title="Average Sentiment Score",
        legend_title="Software",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # Format y-axis to show 2 decimal places
        yaxis=dict(
            tickformat=".2f"
        )
    )
    
    # Use the key parameter if provided
    if key:
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key="software_sentiment_over_time")

def display_software_heatmap(df, software_list, metric="Mentions", time_period="Half Year", key=None):
    """Display software comparison heatmap."""
    st.subheader(f"Software {metric} Heatmap")
    
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
    
    # Create empty DataFrame for heatmap
    heatmap_data = pd.DataFrame(index=software_list, columns=time_periods)
    
    # Fill the DataFrame based on the selected metric
    for software in software_list:
        software_df = df[df['matched_terms'].str.lower().str.contains(software.lower(), na=False)]
        
        for period in time_periods:
            period_df = software_df[software_df['time_period'] == period]
            
            # Initialize value as None
            value = None
            
            if metric == "Mentions":
                value = len(period_df)
            elif metric == "VADER Sentiment":
                if not period_df.empty:
                    value = round(period_df['vader_sentiment_score'].mean(), 2)
            elif metric == "TextBlob Sentiment":
                if not period_df.empty:
                    value = round(period_df['textblob_sentiment_score'].mean(), 2)
            elif metric == "Score":
                if not period_df.empty:
                    value = round(period_df['score'].mean(), 2)
            
            heatmap_data.at[software, period] = value
    
    # Convert to format needed for heatmap
    heatmap_df = heatmap_data.reset_index()
    heatmap_df = heatmap_df.melt(id_vars='index', var_name='Period', value_name='Value')
    heatmap_df.columns = ['Software', 'Period', 'Value']
    
    # Set color scale based on metric
    if "Sentiment" in metric:
        color_scale = [
            [0, '#F44336'],      # Red for negative
            [0.5, '#FFFFFF'],    # White for neutral
            [1, '#4CAF50']       # Green for positive
        ]
        mid_point = 0
    else:
        color_scale = 'Blues'
        mid_point = None
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data.values,
        labels=dict(x=x_title, y="Software", color=metric),
        x=time_periods,
        y=software_list,
        color_continuous_scale=color_scale,
        text_auto=True,  # Just use True for all cases to avoid type errors
        title=f"Software {metric} by {time_period}"
    )
    
    # Set custom hover template to show "N/A" for None values
    hover_template = "%{y}<br>%{x}<br>%{z}"
    if "Sentiment" in metric:
        hover_template = "%{y}<br>%{x}<br>%{z:.2f}"
    
    fig.update_traces(
        hovertemplate=hover_template,
        # Show "N/A" for None values in the cells
        text=[[f"{val:.2f}" if val is not None else "N/A" for val in row] 
              if "Sentiment" in metric else 
              [str(int(val)) if val is not None else "N/A" for val in row] 
              for row in heatmap_data.values]
    )
    
    # For sentiment, set the color scale midpoint at 0
    if "Sentiment" in metric:
        fig.update_layout(coloraxis_colorbar=dict(title=metric))
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Software",
        coloraxis_showscale=True,
    )
    
    # For sentiment, make cells with no data white instead of using the color scale
    if "Sentiment" in metric:
        # Create a mask for None values
        mask = heatmap_data.isna().values
        
        # Apply white color to cells with None values
        for i in range(len(software_list)):
            for j in range(len(time_periods)):
                if mask[i, j]:
                    fig.add_shape(
                        type="rect",
                        x0=j - 0.5,
                        x1=j + 0.5,
                        y0=i - 0.5,
                        y1=i + 0.5,
                        fillcolor="white",
                        line=dict(width=1, color="lightgrey"),
                        layer="below"
                    )
    
    # Use the key parameter if provided
    if key:
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key="software_heatmap") 

def display_keyword_mentions_over_time(df, keyword_list, time_period="Monthly", key=None):
    """Display keyword mentions over time as a line chart."""
    st.subheader("Keyword Mentions Over Time")
    
    if not keyword_list:
        st.warning("No keywords selected to display.")
        return
    
    # Create a dictionary to store counts for each keyword
    keyword_counts = {}
    
    # Group by time period
    if time_period == "Monthly":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.strftime('%Y-%m')
        x_title = "Month"
    elif time_period == "Quarterly":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-Q' + ((df['created_at'].dt.month - 1) // 3 + 1).astype(str)
        x_title = "Quarter"
    elif time_period == "Third Year":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-T' + ((df['created_at'].dt.month - 1) // 4 + 1).astype(str)
        x_title = "Third of Year"
    elif time_period == "Half Year":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.year.astype(str) + '-H' + ((df['created_at'].dt.month - 1) // 6 + 1).astype(str)
        x_title = "Half Year"
    elif time_period == "Yearly":
        df = df.copy()  # Create a copy to avoid modifying the original
        df['time_period'] = df['created_at'].dt.year.astype(str)
        x_title = "Year"
    
    # Get all time periods
    all_time_periods = sorted(df['time_period'].unique())
    
    # Count mentions for each keyword over time
    for keyword in keyword_list:
        # Filter for posts mentioning this keyword
        keyword_df = df[df['keywords'].str.lower().str.contains(keyword.lower(), na=False)]
        
        # Group by time period and count
        counts = keyword_df.groupby('time_period').size()
        keyword_counts[keyword] = counts
    
    # Get all time periods across all keywords
    all_time_periods = set()
    for counts in keyword_counts.values():
        all_time_periods.update(counts.index)
    
    # Create a DataFrame with all time periods and keyword counts
    time_periods_sorted = sorted(all_time_periods)
    data = {'Time': time_periods_sorted}
    
    for keyword, counts in keyword_counts.items():
        data[keyword] = [counts.get(period, 0) for period in time_periods_sorted]
    
    # Create DataFrame
    df_plot = pd.DataFrame(data)
    
    # Melt the DataFrame for plotting
    df_melted = df_plot.melt(id_vars=['Time'], var_name='Keyword', value_name='Mentions')
    
    # Create line chart
    fig = px.line(
        df_melted,
        x='Time',
        y='Mentions',
        color='Keyword',
        markers=True,
        title=f"Keyword Mentions Over Time ({time_period})"
    )
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Number of Mentions",
        legend_title="Keyword",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Use the key parameter if provided
    if key:
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key="keyword_mentions_over_time")

def display_keyword_sentiment_over_time(df, keyword_list, time_period="Monthly", key=None):
    """Display keyword sentiment over time as a line chart."""
    st.subheader("Keyword Sentiment Over Time")
    
    if not keyword_list:
        st.warning("No keywords selected to display.")
        return
    
    # Create a figure with go.Figure() for more control over line connections
    fig = go.Figure()
    
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
    
    # Get all time periods from the dataset
    all_time_periods = sorted(df['time_period'].unique())
    
    # Calculate sentiment for each keyword over time
    for keyword in keyword_list:
        # Filter for posts mentioning this keyword
        keyword_df = df[df['keywords'].str.lower().str.contains(keyword.lower(), na=False)]
        
        # Group by time period and calculate average sentiment
        sentiment_series = keyword_df.groupby('time_period')['vader_sentiment_score'].mean().round(2)
        
        # Convert to dictionary for easier access
        sentiment_dict = sentiment_series.to_dict()
        
        # Create x and y values for the line chart, preserving only periods with data
        x_values = []
        y_values = []
        
        for period in all_time_periods:
            if period in sentiment_dict:
                x_values.append(period)
                y_values.append(sentiment_dict[period])
        
        # Add trace if we have data points
        if x_values:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=keyword,
                connectgaps=True,  # Connect gaps between data points
                line=dict(shape='linear'),  # Linear interpolation between points
                marker=dict(size=8)
            ))
    
    # Add a horizontal line at y=0 to highlight positive vs negative sentiment
    if all_time_periods:
        fig.add_shape(
            type="line",
            x0=all_time_periods[0],
            x1=all_time_periods[-1],
            y0=0,
            y1=0,
            line=dict(color="black", width=2),
            layer="below"
        )
    
    # Update layout
    fig.update_layout(
        title=f"Keyword Sentiment Over Time ({time_period})",
        xaxis_title=x_title,
        yaxis_title="Average Sentiment Score",
        legend_title="Keyword",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # Format y-axis to show 2 decimal places
        yaxis=dict(
            tickformat=".2f"
        )
    )
    
    # Use the key parameter if provided
    if key:
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key="keyword_sentiment_over_time")

def display_keyword_heatmap(df, keyword_list, metric="Mentions", time_period="Half Year", key=None):
    """Display keyword comparison heatmap."""
    st.subheader(f"Keyword {metric} Heatmap")
    
    if not keyword_list:
        st.warning("No keywords selected to display.")
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
    
    # Create a matrix for the heatmap
    time_periods = sorted(df['time_period'].unique())
    
    # Create empty DataFrame for heatmap
    heatmap_data = pd.DataFrame(index=keyword_list, columns=time_periods)
    
    # Fill the DataFrame based on the selected metric
    for keyword in keyword_list:
        keyword_df = df[df['keywords'].str.lower().str.contains(keyword.lower(), na=False)]
        
        for period in time_periods:
            period_df = keyword_df[keyword_df['time_period'] == period]
            
            # Initialize value as None
            value = None
            
            if metric == "Mentions":
                value = len(period_df)
            elif metric == "VADER Sentiment":
                if not period_df.empty:
                    value = round(period_df['vader_sentiment_score'].mean(), 2)
            elif metric == "TextBlob Sentiment":
                if not period_df.empty:
                    value = round(period_df['textblob_sentiment_score'].mean(), 2)
            elif metric == "Score":
                if not period_df.empty:
                    value = round(period_df['score'].mean(), 2)
            
            heatmap_data.at[keyword, period] = value
    
    # Convert to format needed for heatmap
    heatmap_df = heatmap_data.reset_index()
    heatmap_df = heatmap_df.melt(id_vars='index', var_name='Period', value_name='Value')
    heatmap_df.columns = ['Keyword', 'Period', 'Value']
    
    # Set color scale based on metric
    if "Sentiment" in metric:
        color_scale = [
            [0, '#F44336'],      # Red for negative
            [0.5, '#FFFFFF'],    # White for neutral
            [1, '#4CAF50']       # Green for positive
        ]
        mid_point = 0
    else:
        color_scale = 'Blues'
        mid_point = None
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data.values,
        labels=dict(x=x_title, y="Keyword", color=metric),
        x=time_periods,
        y=keyword_list,
        color_continuous_scale=color_scale,
        text_auto=True,  # Just use True for all cases to avoid type errors
        title=f"Keyword {metric} by {time_period}"
    )
    
    # Set custom hover template to show "N/A" for None values
    hover_template = "%{y}<br>%{x}<br>%{z}"
    if "Sentiment" in metric:
        hover_template = "%{y}<br>%{x}<br>%{z:.2f}"
    
    fig.update_traces(
        hovertemplate=hover_template,
        # Show "N/A" for None values in the cells
        text=[[f"{val:.2f}" if val is not None else "N/A" for val in row] 
              if "Sentiment" in metric else 
              [str(int(val)) if val is not None else "N/A" for val in row] 
              for row in heatmap_data.values]
    )
    
    # For sentiment, set the color scale midpoint at 0
    if "Sentiment" in metric:
        fig.update_layout(coloraxis_colorbar=dict(title=metric))
    
    # Use the key parameter if provided
    if key:
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key="keyword_heatmap") 