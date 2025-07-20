#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Content display components for the VC Software Reddit Dashboard.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from utils.text_processing import highlight_matched_terms
from collections import Counter
from utils.data_loader import extract_textblob_keywords

def extract_keywords(df):
    """Extract and process keywords from the dataframe."""
    all_keywords = []
    for keywords in df['keywords'].dropna():
        if isinstance(keywords, str):
            keywords_list = [keyword.strip().lower() for keyword in keywords.split(',')]
            all_keywords.extend(keywords_list)
    
    return Counter(all_keywords)

def display_popular_software_mentions(matched_term_df, all_matched_terms, key_suffix=""):
    """Display popular software mentions bar chart and word cloud."""
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Bar chart of top matched terms
        fig = px.bar(
            matched_term_df.head(20),
            x='Count',
            y='Term',
            orientation='h',
            labels={'Count': 'Frequency', 'Term': ''},
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{key_suffix}")
    
    with col2:
        # Word cloud of matched terms
        if all_matched_terms:
            # Create a unique key for the container
            with st.container(key=f"wordcloud_container_{key_suffix}"):
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis',
                    max_words=100
                ).generate(' '.join(all_matched_terms))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        else:
            st.info("No matched terms available for the current filter selection.")

def display_popular_keywords(keyword_df, all_keywords, key_suffix=""):
    """Display popular keywords bar chart and word cloud."""
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Bar chart of top keywords
        fig = px.bar(
            keyword_df.head(20),
            x='Count',
            y='Keyword',
            orientation='h',
            labels={'Count': 'Frequency', 'Keyword': ''},
            color='Count',
            color_continuous_scale='Greens'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"keyword_bar_chart_{key_suffix}")
    
    with col2:
        # Word cloud of keywords
        if all_keywords:
            # Create a unique key for the container
            with st.container(key=f"keyword_wordcloud_container_{key_suffix}"):
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis',
                    max_words=100
                ).generate(' '.join(all_keywords))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        else:
            st.info("No keywords available for the current filter selection.")

def display_textblob_keyword_cloud(df, key_suffix=""):
    """Display TextBlob keywords word cloud."""
    # Extract TextBlob keywords
    all_textblob_keywords = []
    for keywords in df['textblob_keywords'].dropna():
        if isinstance(keywords, str):
            keywords_list = [keyword.strip().lower() for keyword in keywords.split(',')]
            all_textblob_keywords.extend(keywords_list)
    
    # Create word cloud
    if all_textblob_keywords:
        # Create a unique key for the container
        with st.container(key=f"textblob_wordcloud_container_{key_suffix}"):
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(' '.join(all_textblob_keywords))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    else:
        st.info("No TextBlob keywords available for the current filter selection.")

def display_keyword_trends(df, keyword_list, time_period="Monthly", key_suffix=""):
    """Display keyword trends over time."""
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
    
    # Create a dictionary to store counts for each keyword
    keyword_counts = {}
    
    # Count mentions for each keyword over time
    for keyword in keyword_list:
        # Filter for posts mentioning this keyword
        keyword_df = df[df['textblob_keywords'].str.lower().str.contains(keyword.lower(), na=False)]
        
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
        title=f"Keyword Trends Over Time ({time_period})"
    )
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Number of Mentions",
        legend_title="Keyword",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"keyword_trends_{key_suffix}")

def display_top_posts(df):
    """Display top posts by score."""
    # Sort by score and get top posts
    top_posts = df.nlargest(10, 'score')
    
    # Create a container for better layout
    for idx, (_, post) in enumerate(top_posts.iterrows()):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Post title and content - check if title column exists
                if 'title' in post and pd.notna(post.get('title', '')):
                    st.markdown(f"**{post['title']}**")
                
                # Show text with expandable functionality and highlight matched terms
                full_text = str(post.get('text', ''))
                
                # Highlight matched terms in the text
                highlighted_text = full_text
                if 'matched_terms' in post and pd.notna(post['matched_terms']):
                    matched_terms_list = [term.strip() for term in post['matched_terms'].split(',')]
                    for term in matched_terms_list:
                        if term.lower() in highlighted_text.lower():
                            # Replace the term with highlighted version (case-insensitive)
                            import re
                            pattern = re.compile(re.escape(term), re.IGNORECASE)
                            highlighted_text = pattern.sub(f"**{term}**", highlighted_text)
                
                if len(full_text) > 200:
                    # Show preview with expand option
                    text_preview = highlighted_text[:200] + "..."
                    st.markdown(text_preview)
                    
                    # Use session state to track expanded state
                    expand_key = f"expand_{idx}"
                    if expand_key not in st.session_state:
                        st.session_state[expand_key] = False
                    
                    if st.button("See more", key=f"see_more_{idx}"):
                        st.session_state[expand_key] = not st.session_state[expand_key]
                    
                    if st.session_state[expand_key]:
                        st.markdown(highlighted_text)
                else:
                    st.markdown(highlighted_text)
                
                # Post metadata
                st.caption(f"👤 {post.get('author', 'Unknown')} | "
                          f"📍 r/{post.get('subreddit', 'Unknown')} | "
                          f"📅 {post.get('created_at', 'Unknown')}")
            
            with col2:
                # Upvotes (changed from Score)
                st.metric("Upvotes", f"{post.get('score', 0)}")
                
                # Software mentioned (changed from matched_terms)
                if 'matched_terms' in post and pd.notna(post['matched_terms']):
                    st.write(f"**Software:** {post['matched_terms']}")
                
                # Sentiment summary if available (removed AI reference)
                if 'AI_Summary' in post and pd.notna(post['AI_Summary']):
                    st.write(f"**Summary:** {post['AI_Summary']}")
                
                # URL if available
                if 'url' in post and pd.notna(post['url']):
                    st.markdown(f"[View Post]({post['url']})")
        
        st.divider()

def display_combined_posts_data(df):
    """
    Display posts with filtering capabilities and expandable text.
    
    Parameters:
    df: DataFrame containing the data
    """
    st.markdown('<div class="chart-description">Browse and filter posts with expandable text content.</div>', unsafe_allow_html=True)
    
    # Create filters at the top
    st.markdown("### Filters")
    
    # First row of filters
    col1, col2, col3 = st.columns(3)
    with col1:
        # Date range filter - separate start and end dates
        min_date = df['created_at'].min().date()
        max_date = df['created_at'].max().date()
        
        col1a, col1b = st.columns(2)
        with col1a:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                format="DD/MM/YYYY",
                key="posts_start_date"
            )
        with col1b:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                format="DD/MM/YYYY",
                key="posts_end_date"
            )
    
    with col2:
        # Subreddit filter
        subreddit_options = ['All'] + sorted(df['subreddit'].unique().tolist())
        selected_subreddit_filter = st.selectbox(
            "Subreddit",
            subreddit_options,
            key="posts_subreddit_filter"
        )
    
    with col3:
        # Content type filter
        content_types = ['All'] + sorted(df['type'].unique().tolist())
        selected_type_filter = st.selectbox(
            "Content Type",
            content_types,
            key="posts_type_filter"
        )
    
    # Second row of filters
    col4, col5, col6 = st.columns(3)
    with col4:
        # Software filter
        software_options = ['All']
        if 'matched_terms' in df.columns:
            # Extract unique software terms
            all_software = set()
            for terms in df['matched_terms'].dropna():
                if isinstance(terms, str):
                    software_list = [term.strip() for term in terms.split(',')]
                    all_software.update(software_list)
            # Sort alphabetically
            software_options.extend(sorted(list(all_software)))
        
        selected_software_filter = st.selectbox(
            "Software",
            software_options,
            key="posts_software_filter"
        )
    
    with col5:
        # Sentiment filter - only show the 7 specific sentiment tags we use (without [software])
        sentiment_options = ['All']
        specific_sentiments = [
            'describes positively',
            'recommends/suggests',
            'used to report findings',
            'considering for use',
            'complains about functionality',
            'complains about pricing',
            'asks for alternative'
        ]
        sentiment_options.extend(specific_sentiments)
        
        selected_sentiment_filter = st.selectbox(
            "Sentiment",
            sentiment_options,
            key="posts_sentiment_filter"
        )
    
    with col6:
        # Sort options
        sort_options = ['Upvotes (High to Low)', 'Upvotes (Low to High)', 'Date (Newest)', 'Date (Oldest)']
        selected_sort = st.selectbox(
            "Sort By",
            sort_options,
            key="posts_sort_filter"
        )
    
    # Third row of filters
    col7, col8 = st.columns(2)
    with col7:
        # Score range filter
        min_score = int(df['score'].min())
        max_score = int(df['score'].max())
        filter_score = st.slider(
            "Upvotes Range",
            min_score,
            max_score,
            (min_score, max_score),
            key="posts_score_filter"
        )
    
    with col8:
        # Text search filter
        search_term = st.text_input("Search in text content:", key="posts_search_filter")
    
    # Fourth row - pagination controls
    col9, col10, col11 = st.columns(3)
    with col9:
        # Results per page filter
        results_per_page = st.selectbox(
            "Results per page",
            [10, 20, 50, 100],
            index=0,  # Default to 10
            key="posts_per_page"
        )
    
    with col10:
        # Placeholder for page navigation (moved to bottom)
        st.write("Page navigation at bottom")
    
    with col11:
        # Placeholder for page info (moved to bottom)
        st.write("Page info at bottom")

    # Apply filters
    filtered_posts = df.copy()
    
    # Date filter
    filtered_posts = filtered_posts[(filtered_posts['created_at'].dt.date >= start_date) & 
                                    (filtered_posts['created_at'].dt.date <= end_date)]
    
    # Subreddit filter
    if selected_subreddit_filter != 'All':
        filtered_posts = filtered_posts[filtered_posts['subreddit'] == selected_subreddit_filter]
    
    # Content type filter
    if selected_type_filter != 'All':
        filtered_posts = filtered_posts[filtered_posts['type'] == selected_type_filter]
    
    # Software filter
    if selected_software_filter != 'All':
        filtered_posts = filtered_posts[filtered_posts['matched_terms'].str.contains(selected_software_filter, na=False, case=False)]
    
    # Sentiment filter - handle simplified sentiment labels
    if selected_sentiment_filter != 'All':
        # Map simplified labels back to full patterns
        sentiment_mapping = {
            'describes positively': 'describes [software] positively',
            'recommends/suggests': 'recommends/suggests [software]',
            'used to report findings': 'used [software] to report findings',
            'considering for use': 'considering [software] for use',
            'complains about functionality': 'complains about [software] functionality',
            'complains about pricing': 'complains about [software] pricing',
            'asks for alternative': 'asks for alternative to [software]'
        }
        
        full_sentiment_pattern = sentiment_mapping.get(selected_sentiment_filter, selected_sentiment_filter)
        
        # Get the selected software for substitution
        if selected_software_filter != 'All':
            # Replace [software] with the selected software name
            sentiment_pattern = full_sentiment_pattern.replace('[software]', selected_software_filter)
            filtered_posts = filtered_posts[filtered_posts['AI_Summary'] == sentiment_pattern]
        else:
            # If no software is selected, show posts with any software in that sentiment category
            # Create a pattern that matches any software
            base_sentiment = full_sentiment_pattern.replace('[software]', '.*')
            import re
            pattern = re.compile(base_sentiment, re.IGNORECASE)
            mask = filtered_posts['AI_Summary'].apply(lambda x: bool(pattern.match(str(x))) if pd.notna(x) else False)
            filtered_posts = filtered_posts[mask]
    
    # Score filter
    min_val, max_val = filter_score
    filtered_posts = filtered_posts[(filtered_posts['score'] >= min_val) & 
                                    (filtered_posts['score'] <= max_val)]
    
    # Text search filter
    if search_term:
        try:
            mask = filtered_posts['text'].astype(str).str.lower().str.contains(search_term.lower(), na=False)
            filtered_posts = filtered_posts[mask]
        except Exception as e:
            st.error(f"Search error: {e}")
    
    # Apply sorting
    if selected_sort == 'Upvotes (High to Low)':
        filtered_posts = filtered_posts.sort_values('score', ascending=False)
    elif selected_sort == 'Upvotes (Low to High)':
        filtered_posts = filtered_posts.sort_values('score', ascending=True)
    elif selected_sort == 'Date (Newest)':
        filtered_posts = filtered_posts.sort_values('created_at', ascending=False)
    elif selected_sort == 'Date (Oldest)':
        filtered_posts = filtered_posts.sort_values('created_at', ascending=True)
    
    # Display filtered results count
    st.write(f"Showing {len(filtered_posts)} of {len(df)} posts")
    
    # Calculate pagination variables
    total_posts = len(filtered_posts)
    total_pages = (total_posts + results_per_page - 1) // results_per_page
    
    # Initialize current page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Apply pagination
    start_idx = (st.session_state.current_page - 1) * results_per_page
    end_idx = min(start_idx + results_per_page, total_posts)
    display_posts = filtered_posts.iloc[start_idx:end_idx]
    
    # Display posts (limit to 20 for performance)
    display_posts = filtered_posts.head(20)
    
    # Display posts with improved text display
    for idx, (_, post) in enumerate(display_posts.iterrows()):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Post title and content - check if title column exists
                if 'title' in post and pd.notna(post.get('title', '')):
                    st.markdown(f"**{post['title']}**")
                
                # Show text with highlight any software mentioned (compact design)
                full_text = str(post.get('text', ''))
                
                # Truncate text if longer than 500 characters
                if len(full_text) > 500:
                    full_text = full_text[:500] + "......"
                
                # Highlight any software mentioned in the text
                highlighted_text = full_text
                
                # Get all software terms from the dataset to highlight
                if 'matched_terms' in df.columns:
                    all_software_terms = set()
                    for terms in df['matched_terms'].dropna():
                        if isinstance(terms, str):
                            software_list = [term.strip() for term in terms.split(',')]
                            all_software_terms.update(software_list)
                    
                    # Highlight any software term found in the text
                    for software_term in all_software_terms:
                        if software_term.lower() in highlighted_text.lower():
                            # Replace the term with highlighted version (case-insensitive)
                            import re
                            pattern = re.compile(re.escape(software_term), re.IGNORECASE)
                            highlighted_text = pattern.sub(f'<span style="background-color: yellow;">{software_term}</span>', highlighted_text)
                
                # Display full text (compact) with HTML highlighting
                st.markdown(highlighted_text, unsafe_allow_html=True)
                
                # Post metadata
                st.caption(f"👤 {post.get('author', 'Unknown')} | "
                          f"📍 r/{post.get('subreddit', 'Unknown')} | "
                          f"📅 {post.get('created_at', 'Unknown')}")
            
            with col2:
                # Upvotes (changed from Score)
                st.metric("Upvotes", f"{post.get('score', 0)}")
                
                # Software mentioned (changed from matched_terms)
                if 'matched_terms' in post and pd.notna(post['matched_terms']):
                    st.write(f"**Software:** {post['matched_terms']}")
                
                # Sentiment summary if available (removed AI reference)
                if 'AI_Summary' in post and pd.notna(post['AI_Summary']):
                    st.write(f"**Summary:** {post['AI_Summary']}")
                
                # URL if available
                if 'url' in post and pd.notna(post['url']):
                    st.markdown(f"[View Post]({post['url']})")
        
        st.divider()
    
    # Pagination controls at the bottom
    if total_posts > 0:
        st.markdown("---")
        col_pag1, col_pag2, col_pag3 = st.columns(3)
        
        with col_pag1:
            # Results per page info
            st.write(f"**Results per page:** {results_per_page}")
        
        with col_pag2:
            # Page navigation
            if total_pages > 1:
                current_page = st.selectbox(
                    f"**Page** (1-{total_pages})",
                    range(1, total_pages + 1),
                    index=st.session_state.current_page - 1,
                    key="posts_page_select_bottom"
                )
                st.session_state.current_page = current_page
            else:
                current_page = 1
                st.session_state.current_page = 1
        
        with col_pag3:
            # Page info
            if total_pages > 1:
                start_idx = (current_page - 1) * results_per_page
                end_idx = min(start_idx + results_per_page, total_posts)
                st.write(f"**Showing:** {start_idx + 1}-{end_idx} of {total_posts}")
            else:
                st.write(f"**Showing:** All {total_posts} posts") 