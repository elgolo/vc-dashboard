# 🚀 Streamlit Deployment Guide

This folder contains only the necessary files for deploying the VC Software Reddit Dashboard to Streamlit Cloud.

## 📁 Files Included

- `streamlit_app.py` - Main entry point for Streamlit Cloud
- `requirements.txt` - Python dependencies
- `src/` - Application source code
- `README.md` - Project documentation
- `.gitignore` - Git ignore rules

## 🚀 Deploy to Streamlit Cloud

### 1. Create GitHub Repository
1. Go to [GitHub](https://github.com) and sign in
2. Click "New repository"
3. Name it: `vc-dashboard-streamlit`
4. **Make it PUBLIC** (required for free Streamlit Cloud)
5. Don't initialize with README (we already have one)
6. Click "Create repository"

### 2. Push to GitHub
```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/vc-dashboard-streamlit.git

# Push to GitHub
git push -u origin main
```

### 3. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click "New app"
4. Fill in the form:
   - **Repository**: `YOUR_USERNAME/vc-dashboard-streamlit`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Leave as default
5. Click "Deploy"

## 📊 Dashboard Features

- **Software Analysis**: Popularity tracking, word clouds, time series
- **Sentiment Analysis**: AI summary tags, positive/negative breakdown
- **Content Viewer**: Post browser with highlighting, pagination
- **Advanced Filtering**: Date range, subreddit, content type, software selection

## 🔗 Share Your Dashboard

After deployment:
- **GitHub Repository**: `https://github.com/YOUR_USERNAME/vc-dashboard-streamlit`
- **Live Dashboard**: `https://your-app-name.streamlit.app/`

---

**🎉 Ready to deploy!** Follow the steps above and your dashboard will be live. 