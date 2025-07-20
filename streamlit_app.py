#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VC Software Reddit Dashboard - Streamlit Cloud Entry Point

This file serves as the entry point for Streamlit Cloud deployment.
It imports and runs the main dashboard application.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app
from app import main

if __name__ == "__main__":
    main() 