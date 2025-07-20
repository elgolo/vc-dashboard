#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics display components for the VC Software Reddit Dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import create_sentiment_pie_chart, create_time_series_chart

def display_key_metrics(df):
    """Display key metrics about the dataset."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Posts/Comments", f"{len(df):,}")
    
    with col2:
        unique_subreddits = df['subreddit'].nunique()
        st.metric("Subreddits", f"{unique_subreddits:,}")

def display_sentiment_distribution(df, data_filter="all"):
    """
    Display sentiment distribution as a pie chart.
    
    Parameters:
    df: DataFrame containing the data
    data_filter: 'robust' to filter for textblob subjectivity >= 0.2, 'all' for all data
    """
    st.subheader("Sentiment Distribution")
    
    # Filter data if robust option is selected
    display_df = df.copy()
    if data_filter == "robust":
        display_df = display_df[display_df['textblob_subjectivity'] >= 0.2]
    
    # Get sentiment counts using textblob sentiment
    sentiment_counts = display_df['textblob_sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Create pie chart
    fig = create_sentiment_pie_chart(sentiment_counts)
    st.plotly_chart(fig, use_container_width=True)

def display_sentiment_quadrant(df):
    """Display sentiment quadrant chart with TextBlob subjectivity and sentiment."""
    st.subheader("Sentiment Analysis Quadrant")
    
    # Create a scatter plot with TextBlob subjectivity on x-axis and sentiment on y-axis
    fig = go.Figure()
    
    # Add TextBlob sentiment points
    fig.add_trace(go.Scatter(
        x=df['textblob_subjectivity'],
        y=df['textblob_sentiment_score'],
        mode='markers',
        name='TextBlob Sentiment',
        marker=dict(
            color='blue',
            size=8
        )
    ))
    
    # Add VADER sentiment points (one point per row with the same x-value)
    fig.add_trace(go.Scatter(
        x=df['textblob_subjectivity'],
        y=df['vader_sentiment_score'],
        mode='markers',
        name='VADER Sentiment',
        marker=dict(
            color='red',
            size=8
        )
    ))
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Add vertical line at x=0.2 (subjectivity threshold)
    fig.add_shape(
        type="line",
        x0=0.2,
        x1=0.2,
        y0=-1,
        y1=1,
        line=dict(color="green", width=1, dash="dash")
    )
    
    # Add rectangle to highlight the area with subjectivity >= 0.2
    fig.add_shape(
        type="rect",
        x0=0.2,
        x1=1,
        y0=-1,
        y1=1,
        fillcolor="rgba(0,255,0,0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    # Update layout
    fig.update_layout(
        title="Sentiment Analysis: TextBlob vs VADER",
        xaxis_title="TextBlob Subjectivity",
        yaxis_title="Sentiment Score",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[-1, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add annotations
    fig.add_annotation(
        x=0.6,
        y=0.9,
        text="More Subjective (≥0.2)",
        showarrow=False,
        font=dict(color="green")
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_activity_over_time(df, time_period="Monthly"):
    """Display activity over time as a line chart."""
    st.subheader("Activity Over Time")
    
    # Group by time period
    if time_period == "Monthly":
        df_grouped = df.groupby(df['created_at'].dt.strftime('%Y-%m')).size().reset_index(name='count')
        x_title = "Month"
    elif time_period == "Quarterly":
        df['quarter'] = df['created_at'].dt.year.astype(str) + '-Q' + ((df['created_at'].dt.month - 1) // 3 + 1).astype(str)
        df_grouped = df.groupby('quarter').size().reset_index(name='count')
        x_title = "Quarter"
    elif time_period == "Third Year":
        df['third'] = df['created_at'].dt.year.astype(str) + '-T' + ((df['created_at'].dt.month - 1) // 4 + 1).astype(str)
        df_grouped = df.groupby('third').size().reset_index(name='count')
        x_title = "Third of Year"
    elif time_period == "Half Year":
        df['half_year'] = df['created_at'].dt.year.astype(str) + '-H' + ((df['created_at'].dt.month - 1) // 6 + 1).astype(str)
        df_grouped = df.groupby('half_year').size().reset_index(name='count')
        x_title = "Half Year"
    elif time_period == "Yearly":
        df_grouped = df.groupby(df['created_at'].dt.year).size().reset_index(name='count')
        x_title = "Year"
    
    # Create line chart
    fig = create_time_series_chart(
        df_grouped, 
        x_col=df_grouped.columns[0], 
        y_col='count', 
        title=f"Activity Over Time ({time_period})",
        x_title=x_title,
        y_title="Number of Posts/Comments"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_focus_mode_metrics(df, focus_df, focused_term):
    """Display metrics for focus mode."""
    st.subheader(f"Focus Mode: {focused_term.capitalize()} vs Other Software")
    
    # Calculate metrics
    total_posts = len(df)
    focus_posts = len(focus_df)
    focus_percentage = (focus_posts / total_posts * 100) if total_posts > 0 else 0
    
    # Average sentiment scores
    focus_sentiment = focus_df['vader_sentiment_score'].mean() if not focus_df.empty else 0
    overall_sentiment = df['vader_sentiment_score'].mean() if not df.empty else 0
    
    # Average post scores
    focus_score = focus_df['score'].mean() if not focus_df.empty else 0
    overall_score = df['score'].mean() if not df.empty else 0
    
    # Average mentions per post (if available)
    focus_mentions = focus_df['matched_terms'].str.count(',').mean() + 1 if not focus_df.empty and 'matched_terms' in focus_df.columns else 0
    overall_mentions = df['matched_terms'].str.count(',').mean() + 1 if not df.empty and 'matched_terms' in df.columns else 0
    
    # Round sentiment scores to 2 decimal places
    focus_sentiment = round(focus_sentiment, 2)
    overall_sentiment = round(overall_sentiment, 2)
    focus_score = round(focus_score, 2)
    overall_score = round(overall_score, 2)
    
    # Display basic metrics
    st.markdown(f"<div class='focus-metrics'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            f"{focused_term.capitalize()} Mentions", 
            f"{focus_posts:,}", 
            f"{focus_percentage:.1f}% of total"
        )
    
    with col2:
        st.metric(
            "Other Software Mentions", 
            f"{total_posts - focus_posts:,}",
            f"{100 - focus_percentage:.1f}% of total"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create bar charts for the three metrics
    st.markdown("### Comparison Metrics")
    
    # Create three columns for the three bar charts
    col1, col2, col3 = st.columns(3)
    
    # 1. Average Sentiment Chart
    with col1:
        st.subheader("Average Sentiment")
        sentiment_data = {
            'Software': [focused_term.capitalize(), 'Others'],
            'Sentiment': [focus_sentiment, overall_sentiment]
        }
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Create bar chart for sentiment
        fig_sentiment = px.bar(
            sentiment_df, 
            x='Software', 
            y='Sentiment',
            color='Software',
            text=sentiment_df['Sentiment'].apply(lambda x: f"{x:.2f}"),
            title="Average Sentiment Score"
        )
        fig_sentiment.update_layout(height=300)
        # Add a horizontal line at y=0 to highlight positive vs negative
        fig_sentiment.add_shape(
            type="line",
            x0=-0.5,
            x1=1.5,
            y0=0,
            y1=0,
            line=dict(color="black", width=2)
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # 2. Average Score Chart
    with col2:
        st.subheader("Average Score")
        score_data = {
            'Software': [focused_term.capitalize(), 'Others'],
            'Score': [focus_score, overall_score]
        }
        score_df = pd.DataFrame(score_data)
        
        # Create bar chart for score
        fig_score = px.bar(
            score_df, 
            x='Software', 
            y='Score',
            color='Software',
            text=score_df['Score'].apply(lambda x: f"{x:.2f}"),
            title="Average Post Score"
        )
        fig_score.update_layout(height=300)
        st.plotly_chart(fig_score, use_container_width=True)
    
    # 3. Average Mentions Chart
    with col3:
        st.subheader("Average Mentions")
        mentions_data = {
            'Software': [focused_term.capitalize(), 'Others'],
            'Mentions': [focus_mentions, overall_mentions]
        }
        mentions_df = pd.DataFrame(mentions_data)
        
        # Create bar chart for mentions
        fig_mentions = px.bar(
            mentions_df, 
            x='Software', 
            y='Mentions',
            color='Software',
            text=mentions_df['Mentions'].apply(lambda x: f"{x:.2f}"),
            title="Average Mentions per Post"
        )
        fig_mentions.update_layout(height=300)
        st.plotly_chart(fig_mentions, use_container_width=True) 