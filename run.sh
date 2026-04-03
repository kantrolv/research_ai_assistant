#!/bin/bash
# Run the Research AI Assistant with Python 3.12
# Usage: ./run.sh
cd "$(dirname "$0")"
.venv/bin/streamlit run app.py
