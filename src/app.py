#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VC Software Reddit Dashboard

This Streamlit app visualizes data from Reddit posts and comments related to VC software.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data, extract_matched_terms, extract_textblob_keywords
from components.metrics import display_key_metrics
from components.software_analysis import display_software_mentions_over_time
from components.content_display import display_popular_software_mentions, display_combined_posts_data
from components.sentiment_analysis import display_sentiment_breakdown, display_sentiment_stacked_bar, display_exposure_diverging_chart
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import plotly.express as px
import time # Added for message duration

def main():
    # Set page config
    st.set_page_config(
        page_title="VC Software Reddit Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E3A8A;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2563EB;
            margin-top: 1rem;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #1E3A8A;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #E5E7EB;
        }
        .card {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #F3F4F6;
            margin-bottom: 1rem;
        }
        .highlight {
            background-color: #FFFF00;
            padding: 0.1rem 0.2rem;
            border-radius: 0.2rem;
        }
        .stRadio > div {
            display: flex;
            flex-direction: row;
        }
        .stRadio label {
            margin-right: 15px;
            padding: 5px 10px;
            border-radius: 4px;
            background-color: #f0f2f6;
        }
        .focus-metrics {
            background-color: #f0f8ff;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .chart-description {
            font-size: 0.9rem;
            color: #4B5563;
            margin-bottom: 1rem;
            font-style: italic;
        }
        .chart-container {
            background-color: #F9FAFB;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #E5E7EB;
        }
        /* Fix for legend display */
        .js-plotly-plot .plotly .legend {
            max-height: none !important;
            overflow-y: visible !important;
        }
        /* Zero line styling for sentiment charts */
        .zero-line {
            stroke: #000 !important;
            stroke-width: 2px !important;
            stroke-dasharray: none !important;
            opacity: 1 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load data
    try:
        df = load_data()
        with st.empty():
            st.success("Data loaded successfully!")
            time.sleep(2)  # Show message for 2 seconds
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Header
    st.markdown('<div class="main-header">VC Software Reddit Analysis Dashboard</div>', unsafe_allow_html=True)
    st.write("This dashboard summarizes Reddit mentions of venture capital (VC) software tools across two subreddits. Each post is analyzed for sentiment, exposure (via post score), and how the tool is discussed — whether it's recommended, criticized, considered, or simply mentioned." +
    "\n\n Note: Data is based on public posts from two VC-related subreddits. Due to API limitations, results reflect a sample of discussions, not the full universe of Reddit activity." +
    "\n\n To see other projects that may interest you please visit my website at www.hasanelgohary.com")

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")

    # Get matched terms counts for initial software selection
    matched_term_counts = extract_matched_terms(df)
    # Convert Counter to list of tuples before creating DataFrame
    term_counts_list = list(matched_term_counts.most_common())
    matched_term_df = pd.DataFrame(term_counts_list, columns=['Term', 'Count'])

    # Get software terms (excluding general terms)
    general_terms = ['software', 'fund', 'experience', 'phone']
    software_terms = matched_term_df[~matched_term_df['Term'].str.lower().isin(general_terms)]

    # Get top software terms
    top_software = software_terms.head(5)['Term'].tolist()

    # Get all software terms for custom selection (sorted alphabetically)
    all_software = sorted(software_terms['Term'].tolist())

    # Software selection at the top of the sidebar
    st.sidebar.markdown("### Software Selection")

    # Chart preset options
    preset_options = ["Top 5 Most Mentioned Software", "All Software", "Custom Selection"]
    chart_preset = st.sidebar.radio("Chart Display Mode:", preset_options, horizontal=True, index=0)

    # Custom selection if chosen
    selected_software = top_software.copy()
    if chart_preset == "All Software":
        selected_software = all_software
    elif chart_preset == "Custom Selection":
        selected_software = st.sidebar.multiselect(
            "Select Software to Display:", 
            options=all_software,
            default=top_software
        )
        if not selected_software:  # If nothing selected, default to top 5
            selected_software = top_software

    # Display selected software count in sidebar
    st.sidebar.markdown(f"**{len(selected_software)} software** selected")

    st.sidebar.markdown("---")

    # Date range filter - separate start and end date inputs with default values
    min_date = df['created_at'].min().date()
    max_date = df['created_at'].max().date()

    # Set default dates to 01/01/2020 - 01/07/2025
    default_start = date(2020, 1, 1)
    default_end = date(2025, 7, 1)

    # Ensure defaults are within the available date range
    default_start = max(min_date, default_start)
    default_end = min(max_date, default_end)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY"
        )

    # Content type filter
    content_types = ['All'] + sorted(df['type'].unique().tolist())
    selected_type = st.sidebar.selectbox("Content Type", content_types)

    # Subreddit filter - simplified to dropdown with 3 options
    subreddit_options = ['All', 'startups', 'venturecapital']
    selected_subreddit = st.sidebar.selectbox("Subreddit", subreddit_options, index=0)

    # Time period filter for charts - default to Half Year
    time_periods = ["Monthly", "Quarterly", "Third Year", "Half Year", "Yearly"]
    selected_time_period = st.sidebar.selectbox("Time Period for Charts", time_periods, index=3)  # Default to Half Year

    # Content filter for mention charts
    content_filters = ["All Content", "Comments Only", "Posts Only"]
    selected_content_filter = st.sidebar.selectbox("Content Filter for Mention Charts", content_filters)

    # Apply filters
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['created_at'].dt.date >= start_date) & 
                              (filtered_df['created_at'].dt.date <= end_date)]

    # Apply subreddit filter
    if selected_subreddit != 'All':
        filtered_df = filtered_df[filtered_df['subreddit'] == selected_subreddit]
        
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['type'] == selected_type]

    # Apply content filter for mentions
    content_filtered_df = filtered_df.copy()
    if selected_content_filter == "Comments Only":
        content_filtered_df = filtered_df[filtered_df['type'] == 'comment']
    elif selected_content_filter == "Posts Only":
        content_filtered_df = filtered_df[filtered_df['type'] == 'submission']

    # Extract all matched terms for the filtered data
    all_matched_terms = []
    for terms in filtered_df['matched_terms'].dropna():
        if isinstance(terms, str):
            terms_list = [term.strip().lower() for term in terms.split(',')]
            all_matched_terms.extend(terms_list)

    # Get matched terms and keywords for the current filtered data
    filtered_matched_terms = extract_matched_terms(filtered_df)
    term_counts_list = list(filtered_matched_terms.most_common())
    filtered_term_df = pd.DataFrame(term_counts_list, columns=['Term', 'Count'])

    # Always filter the term dataframe to only include selected software
    filtered_term_df = filtered_term_df[filtered_term_df['Term'].isin(selected_software)]

    # Recreate the matched terms list based on filtered dataframe
    filtered_matched_terms_list = []
    for term, count in zip(filtered_term_df['Term'], filtered_term_df['Count']):
        filtered_matched_terms_list.extend([term] * count)

    # Display key metrics at the top
    display_key_metrics(filtered_df)

    # Software Analysis Section - Collapsible
    with st.expander("Software Popularity Analysis", expanded=True):
        # Display word frequency chart and word cloud
        display_popular_software_mentions(filtered_term_df, filtered_matched_terms_list, key_suffix="top_software")
        
        # Software mentions over time
        st.markdown('<div class="chart-description">Shows how frequently different software is mentioned over time.</div>', unsafe_allow_html=True)
        display_software_mentions_over_time(content_filtered_df, selected_software, selected_time_period, key="software_mentions")

    # Sentiment Analysis Section - Collapsible
    with st.expander("AI Sentiment Analysis", expanded=True):
        # Display AI Summary Analysis
        display_sentiment_breakdown(filtered_df, selected_software)
        
        # Display 100% stacked bar chart for positive/negative sentiment
        display_sentiment_stacked_bar(filtered_df, selected_software)
        
        # Display Exposure Diverging Chart
        display_exposure_diverging_chart(filtered_df, selected_software)

    # Posts & Raw Data Section - Collapsible
    with st.expander("Posts & Raw Data", expanded=True):
        # Combined posts and raw data functionality
        display_combined_posts_data(filtered_df)

    # Footer
    st.markdown("---")
    st.markdown("VC Software Reddit Analysis Dashboard | Created with Streamlit")


if __name__ == "__main__":
    main() 