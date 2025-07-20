#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loading and processing utilities for the VC Software Reddit Dashboard.
"""

import pandas as pd
import streamlit as st
from collections import Counter
from datetime import datetime

@st.cache_data
def load_data():
    """Load and preprocess the Reddit data from Master_Raw_Data_UPDATED."""
    try:
        # Try to load from the src/data directory (for deployment)
        df = pd.read_excel('src/data/20250718_Master_Raw_Data_UPDATED.xlsx', sheet_name='RAWDATA')
    except FileNotFoundError:
        try:
            # Try to load from the relative path (when running from src directory)
            df = pd.read_excel('../20250718_Master_Raw_Data_UPDATED.xlsx', sheet_name='RAWDATA')
        except FileNotFoundError:
            try:
                # Try to load from the absolute path
                df = pd.read_excel('20250718_Master_Raw_Data_UPDATED.xlsx', sheet_name='RAWDATA')
            except FileNotFoundError:
                # Try to load directly from the root directory
                df = pd.read_excel('20250718_Master_Raw_Data_UPDATED.xlsx', sheet_name='RAWDATA')
    
    # Convert created_at to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        df['created_at'] = pd.to_datetime(df['created_at'])
    # Add month and year columns for time-based analysis
    df['month_year'] = df['created_at'].dt.strftime('%Y-%m')
    df['year'] = df['created_at'].dt.year
    df['half_year'] = df['created_at'].dt.year.astype(str) + '-H' + ((df['created_at'].dt.month - 1) // 6 + 1).astype(str)
    
    # Add an ID column if it doesn't exist
    if 'id' not in df.columns:
        df['id'] = range(len(df))
        
    return df

def extract_matched_terms(df):
    """Extract and process matched terms from the dataframe."""
    all_terms = []
    for terms in df['matched_terms'].dropna():
        if isinstance(terms, str):
            terms_list = [term.strip().lower() for term in terms.split(',')]
            all_terms.extend(terms_list)
    
    return Counter(all_terms)

def extract_textblob_keywords(df):
    """Extract and process textblob keywords from the dataframe."""
    all_keywords = []
    for keywords in df['textblob_keywords'].dropna():
        if isinstance(keywords, str):
            keywords_list = [keyword.strip().lower() for keyword in keywords.split(',')]
            all_keywords.extend(keywords_list)
    
    return Counter(all_keywords)

def create_time_axis_ticks(dates):
    """Create axis ticks with major ticks for years and minor ticks for months."""
    dates_sorted = sorted(dates)
    if not dates_sorted:
        return [], [], []
    
    # Convert to datetime objects if they're strings
    if isinstance(dates_sorted[0], str):
        dates_sorted = [pd.to_datetime(d + '-01') for d in dates_sorted]
    
    # Get unique years and create major ticks
    years = sorted(set(d.year for d in dates_sorted))
    major_ticks = [datetime(year=y, month=1, day=1) for y in years]
    major_labels = [str(y) for y in years]
    
    # Create minor ticks for each month
    minor_ticks = []
    for d in dates_sorted:
        dt = datetime(year=d.year, month=d.month, day=1)
        if dt not in minor_ticks:
            minor_ticks.append(dt)
    
    return major_ticks, major_labels, minor_ticks 