import os
from os.path import join, dirname
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv(join(dirname(__file__), '.env'))
        self.av_api_key = os.getenv('ALPHA_VANTAGE_KEY', None)
        self.av_url = os.getenv('ALPHA_VANTAGE_URL', None)
        self.lookback = os.getenv('LOOKBACK', 8)