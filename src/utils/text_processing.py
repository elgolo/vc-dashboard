#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text processing utilities for the VC Software Reddit Dashboard.
"""

import re

def highlight_matched_terms(text, terms):
    """Highlight matched terms in the text."""
    if not isinstance(text, str) or not terms:
        return text
    
    highlighted_text = text
    for term in terms:
        if not term or not isinstance(term, str):
            continue
        
        # Case-insensitive replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted_text = pattern.sub(f'<span class="highlight">{term}</span>', highlighted_text)
    
    return highlighted_text 