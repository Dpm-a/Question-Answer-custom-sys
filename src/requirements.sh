#!/bin/bash

# Install dependencies
pip install spacy
pip install openai==0.28.1
pip install tiktoken
pip install argparse
pip install pandas
pip install scipy

# Download spacy model
python -m spacy download it_core_news_sm
python -m nltk.downloader punkt
python -m nltk.downloader wordnet
