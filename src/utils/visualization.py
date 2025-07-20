#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for the VC Software Reddit Dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

def create_sentiment_pie_chart(sentiment_counts):
    """Create a pie chart for sentiment distribution."""
    fig = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment', 
        color='Sentiment',
        color_discrete_map={
            'positive': '#4CAF50',
            'neutral': '#FFC107',
            'negative': '#F44336'
        },
        hole=0.4
    )
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_time_series_chart(df, x_col, y_col, title, x_title, y_title):
    """Create a time series chart."""
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        markers=True
    )
    
    fig.update_layout(
        xaxis_title=x_title, 
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_wordcloud(text_data, max_words=100, width=800, height=400):
    """Generate a word cloud from text data."""
    wordcloud = WordCloud(
        width=width, 
        height=height, 
        max_words=max_words,
        background_color='white',
        colormap='viridis',
        collocations=False
    ).generate(text_data)
    
    # Create a matplotlib figure
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    # Encode the bytes as base64
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def create_bar_chart(df, x_col, y_col, title, x_title, y_title, color=None, text_auto=True):
    """Create a bar chart."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color,
        title=title,
        text_auto=text_auto
    )
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title=color if color else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if color else None
    )
    
    return fig 