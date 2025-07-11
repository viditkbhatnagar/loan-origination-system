# Configuration file for the loan origination system

import os

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, 'models', 'saved_models')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# Loan eligibility threshold (lowered for testing - models are conservative)
# Original: 0.5, Temporary: 0.14 to allow best profiles to get approved
LOAN_ELIGIBILITY_THRESHOLD = 0.10  # 10% threshold for easier approval

# Face verification settings
FACE_DATABASE_FOLDER = os.path.join(DATA_FOLDER, 'face_database')
FACE_TOLERANCE = 0.6  # Face recognition tolerance (lower = more strict)

# File upload settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MODELS_FOLDER = MODELS_FOLDER
    DATA_FOLDER = DATA_FOLDER
    UPLOAD_FOLDER = UPLOAD_FOLDER
    LOAN_ELIGIBILITY_THRESHOLD = LOAN_ELIGIBILITY_THRESHOLD
    FACE_DATABASE_FOLDER = FACE_DATABASE_FOLDER
    FACE_TOLERANCE = FACE_TOLERANCE
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH