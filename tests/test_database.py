import sqlite3
import pytest
import pandas as pd
import tempfile
import os
from app import TrackingDatabase
import numpy as np
import time

class TestTrackingDatabase:
    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp()
        yield path
        os.close(fd)
        os.unlink(path)

    def test_init_db(self, temp_db):
        db = TrackingDatabase(temp_db)
        with sqlite3.connect(temp_db) as conn:
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analyses'")
            assert c.fetchone() is not None

    def test_save_analysis(self, temp_db):
        db = TrackingDatabase(temp_db)
        test_data = {
            'filename': 'test.mp4',
            'model_name': 'test_model',
            'avg_displacement': 1.23,
            'tracking_coverage': 0.95,
            'temporal_consistency': 0.87,
            'optical_flow': 0.76,
            'avg_track_length': 15.2,
            'max_active_tracks': 42,
            'tracking_score': 0.987,
            'processing_time': 12.34
        }
        db.save_analysis(test_data)
        
        with sqlite3.connect(temp_db) as conn:
            df = pd.read_sql("SELECT * FROM analyses", conn)
            assert len(df) == 1
            assert df.iloc[0]['filename'] == 'test.mp4'
            assert df.iloc[0]['tracking_score'] == 0.987

    def test_get_recent_analyses(self, temp_db):
        
        db = TrackingDatabase(temp_db)
        
        for i in range(3):
            if i > 0:
                time.sleep(1.1)
            test_data = {
                'filename': f'test_{i}.mp4',
                'model_name': 'test_model',
                'avg_displacement': 0,
                'tracking_coverage': 0.5,
                'temporal_consistency': 0.5,
                'optical_flow': 0.5,
                'avg_track_length': 10,
                'max_active_tracks': 10,
                'tracking_score': 0.5,
                'processing_time': 10
            }
            db.save_analysis(test_data)

        df = db.get_recent_analyses(2)
        assert len(df) == 2
        assert df.iloc[0]['filename'] == 'test_2.mp4'
        assert df.iloc[1]['filename'] == 'test_1.mp4'

    def test_empty_db(self, temp_db):
        db = TrackingDatabase(temp_db)
        df = db.get_recent_analyses()
        assert df.empty

    def test_data_types(self, temp_db):
        db = TrackingDatabase(temp_db)
        test_data = {
            'filename': 123,
            'model_name': 456,
            'avg_displacement': '1.23',
            'tracking_coverage': '0.95',
            'temporal_consistency': '0.87',
            'optical_flow': '0.76',
            'avg_track_length': '15.2',
            'max_active_tracks': '42',
            'tracking_score': '0.987',
            'processing_time': '12.34'
        }
        db.save_analysis(test_data)
        
        df = db.get_recent_analyses()
        assert isinstance(df.iloc[0]['filename'], str)
        assert isinstance(df.iloc[0]['model_name'], str)
        assert isinstance(df.iloc[0]['avg_displacement'], float)
        assert isinstance(df.iloc[0]['max_active_tracks'], np.int64)
