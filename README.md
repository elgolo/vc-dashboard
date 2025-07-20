# VC Software Reddit Analysis Dashboard

A comprehensive Streamlit dashboard that analyzes Reddit discussions about venture capital (VC) software tools. The dashboard provides insights into software popularity, sentiment analysis, and user discussions across VC-related subreddits.

## 🚀 Live Demo

[Deploy on Streamlit Cloud](https://share.streamlit.io/)

## 📊 Features

- **Software Popularity Analysis**: Track mentions and popularity of VC software tools over time
- **AI Sentiment Analysis**: Analyze sentiment breakdown using AI-generated summary tags
- **Interactive Visualizations**: Line charts, bar charts, heatmaps, and word clouds
- **Advanced Filtering**: Filter by date range, subreddit, content type, and software selection
- **Posts & Raw Data Viewer**: Browse and search through actual Reddit posts with highlighting
- **Pagination**: Navigate through large datasets efficiently
- **Responsive Design**: Professional UI with collapsible sections

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vc-dashboard.git
   cd vc-dashboard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your data file**
   - Place your Excel file (e.g., `20250718_Master_Raw_Data_UPDATED.xlsx`) in the `src/data/` directory
   - Update the file path in `src/utils/data_loader.py` if needed

5. **Run the dashboard**
   ```bash
   streamlit run src/app.py
   ```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Add your data file** to the `src/data/` directory in your forked repository

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with GitHub
   - Click "New app"
   - Select your forked repository
   - Set the main file path to: `streamlit_app.py`
   - Click "Deploy"

## 📁 Project Structure

```
vcdashboard/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── components/            # Modular dashboard components
│   │   ├── metrics.py         # Key metrics display
│   │   ├── software_analysis.py # Software popularity charts
│   │   ├── content_display.py # Posts and data viewer
│   │   └── sentiment_analysis.py # Sentiment analysis charts
│   ├── utils/                 # Utility functions
│   │   ├── data_loader.py     # Data loading and processing
│   │   └── visualization.py   # Chart creation utilities
│   └── data/                  # Data files (not in repo)
├── streamlit_app.py           # Streamlit Cloud entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore rules
```

## 📊 Data Requirements

The dashboard expects an Excel file with the following columns:
- `created_at`: Date/time of the post/comment
- `subreddit`: Subreddit name
- `type`: Content type (submission/comment)
- `text`: Post/comment text
- `score`: Post score
- `matched_terms`: Comma-separated software mentions
- `AI_Summary`: AI-generated sentiment summary tags
- Additional sentiment and analysis columns

## 🎯 Key Features

### Software Analysis
- **Popularity Tracking**: See which software tools are most discussed
- **Time Series Analysis**: Track mentions over different time periods
- **Word Clouds**: Visualize most common terms and keywords

### Sentiment Analysis
- **AI Summary Tags**: 7 specific sentiment categories
- **Positive/Negative Breakdown**: Separate charts for sentiment types
- **Exposure Analysis**: Weighted sentiment by post exposure

### Content Viewer
- **Post Browser**: Search and filter through actual Reddit posts
- **Software Highlighting**: Yellow highlighting for mentioned software
- **Pagination**: Navigate large datasets efficiently
- **Raw Data Export**: View and filter the underlying data

## 🔧 Configuration

### Software Selection
- **Top 5 Most Mentioned**: Default view of most popular software
- **All Software**: View all available software tools
- **Custom Selection**: Choose specific software to compare

### Time Periods
- Monthly, Quarterly, Third Year, Half Year, Yearly views
- Custom date range selection
- Real-time filtering

### Content Filters
- Subreddit selection (startups/venturecapital)
- Content type (posts/comments)
- Sentiment tag filtering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Hasan Elgohary**
- Website: [www.hasanelgohary.com](https://www.hasanelgohary.com)
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Reddit API for providing the data
- Streamlit for the amazing dashboard framework
- Plotly for interactive visualizations
- The VC community for valuable insights

---

**Note**: This dashboard analyzes public Reddit posts from VC-related subreddits. Due to API limitations, results reflect a sample of discussions, not the full universe of Reddit activity. 