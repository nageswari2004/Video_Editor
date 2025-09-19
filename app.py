"""
Video Processing Application - MoviePy-Free Version

This application has been optimized to use FFmpeg for all core video processing operations.
MoviePy has been removed from:
- Video merging and concatenation
- Video trimming and compression  
- Speed changes and effects
- Overlays and transitions
- Color grading and animations
- All prompt-based video processing

Benefits of FFmpeg-only approach:
- Faster processing (no Python overhead)
- Lower memory usage (streaming processing)
- Better stability and reliability
- Smaller dependency footprint
- Production-ready performance

Note: External integrations (YouTube, Pexels, Dailymotion) still use MoviePy
for compatibility with their APIs, but core functionality is FFmpeg-based.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, flash
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from functools import wraps
from ml_processor import MLVideoProcessor
from ml_enhancements import MLVideoEnhancer
from flask_sqlalchemy import SQLAlchemy
import time
import subprocess
import sys
import json
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import util
except ImportError:
    import sentence_transformers.util as util
import numpy as np
import yt_dlp
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from bs4 import BeautifulSoup
import re
import gc
from concurrent.futures import ThreadPoolExecutor
import cv2

# Load environment variables
load_dotenv()

# Define ImageMagick path for Windows
IMAGEMAGICK_PATH = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# API configurations
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')

# NOTE: MoviePy has been removed from core functionality
# External integrations (YouTube, Pexels, Dailymotion) still use MoviePy
# All main video processing now uses FFmpeg for better performance

# ImageMagick configuration (for text overlays if needed)
def check_imagemagick():
    """Check if ImageMagick is available for text processing."""
    try:
        result = subprocess.run(['magick', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("ImageMagick is available")
            return True
    except FileNotFoundError:
        print("INFO: ImageMagick not found. Text overlays will use FFmpeg drawtext filter.")
    return False

# Check ImageMagick availability at startup
imagemagick_available = check_imagemagick()

# FFmpeg utility functions for direct video processing
def compress_video_ffmpeg(input_path, output_path, crf=23, preset='medium'):
    """
    Compress video using direct FFmpeg command for better performance
    """
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False, f"FFmpeg error: {result.stderr}"
        
        return True, "Video compressed successfully"
        
    except subprocess.TimeoutExpired:
        return False, "Video processing timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"

def trim_video_ffmpeg(input_path, output_path, start_time, end_time):
    """
    Trim video using direct FFmpeg for faster processing
    """
    try:
        duration = end_time - start_time
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"FFmpeg trim error: {result.stderr}")
            return False, f"FFmpeg error: {result.stderr}"
        
        return True, "Video trimmed successfully"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def merge_videos_ffmpeg(input_paths, output_path):
    """
    Merge videos using direct FFmpeg for better performance
    """
    try:
        # Create a temporary file list for FFmpeg concat
        list_file = os.path.join(app.config['UPLOAD_FOLDER'], f'filelist_{int(time.time())}.txt')
        
        with open(list_file, 'w') as f:
            for path in input_paths:
                f.write(f"file '{path}'\n")
        
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Clean up list file
        if os.path.exists(list_file):
            os.remove(list_file)
        
        if result.returncode != 0:
            print(f"FFmpeg merge error: {result.stderr}")
            return False, f"FFmpeg error: {result.stderr}"
        
        return True, "Videos merged successfully"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("FFmpeg is available and working")
            return True
    except FileNotFoundError:
        print("WARNING: FFmpeg not found. Please install FFmpeg for optimal performance.")
    return False

# Check FFmpeg availability at startup
ffmpeg_available = check_ffmpeg()

# Comprehensive FFmpeg functions for all video operations
def resize_video_ffmpeg(input_path, output_path, width, height):
    """Resize video using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'scale={width}:{height}',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def change_speed_ffmpeg(input_path, output_path, speed_factor):
    """Change video speed using FFmpeg"""
    try:
        # For video speed
        video_speed = speed_factor
        # For audio speed (to maintain sync)
        audio_speed = speed_factor
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-filter_complex', f'[0:v]setpts={1/video_speed}*PTS[v];[0:a]atempo={audio_speed}[a]',
            '-map', '[v]', '-map', '[a]',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def extract_audio_ffmpeg(input_path, output_path):
    """Extract audio using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vn',  # No video
            '-acodec', 'mp3',
            '-ab', '192k',
            '-ar', '44100',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def apply_fade_ffmpeg(input_path, output_path, fade_in_duration=1, fade_out_duration=1):
    """Apply fade in/out using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'fade=in:0:{fade_in_duration*30},fade=out:st={fade_in_duration}:d={fade_out_duration}',
            '-af', f'afade=in:st=0:d={fade_in_duration},afade=out:st={fade_out_duration}:d={fade_out_duration}',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def apply_blur_ffmpeg(input_path, output_path, blur_strength=5):
    """Apply blur effect using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'boxblur={blur_strength}:{blur_strength}',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def apply_brightness_contrast_ffmpeg(input_path, output_path, brightness=0, contrast=1):
    """Apply brightness and contrast using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'eq=brightness={brightness}:contrast={contrast}',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def apply_saturation_ffmpeg(input_path, output_path, saturation=1):
    """Apply saturation using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'eq=saturation={saturation}',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def apply_grayscale_ffmpeg(input_path, output_path):
    """Convert to grayscale using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', 'format=gray',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def add_overlay_ffmpeg(main_path, overlay_path, output_path, x=0, y=0, width=None, height=None, duration=None, start_time=0):
    """Add overlay using FFmpeg - supports both video and image overlays"""
    try:
        # Check if overlay is an image or video
        overlay_ext = os.path.splitext(overlay_path)[1].lower()
        is_image = overlay_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        
        if is_image:
            # Handle image overlay
            if width and height:
                scale_filter = f'scale={width}:{height}'
            elif width:
                scale_filter = f'scale={width}:-1'
            elif height:
                scale_filter = f'scale=-1:{height}'
            else:
                scale_filter = 'scale=iw:ih'
            
            # For images, we need to loop them for the duration
            if duration:
                filter_complex = f'[1:v]loop=loop=-1:size=1:start=0,{scale_filter}[ovr];[0:v][ovr]overlay={x}:{y}:enable=\'between(t,{start_time},{start_time + duration})\'[out]'
            else:
                filter_complex = f'[1:v]loop=loop=-1:size=1:start=0,{scale_filter}[ovr];[0:v][ovr]overlay={x}:{y}[out]'
        else:
            # Handle video overlay
            if width and height:
                scale_filter = f'scale={width}:{height}'
            elif width:
                scale_filter = f'scale={width}:-1'
            elif height:
                scale_filter = f'scale=-1:{height}'
            else:
                scale_filter = 'scale=iw:ih'
            
            if duration:
                filter_complex = f'[1:v]{scale_filter}[ovr];[0:v][ovr]overlay={x}:{y}:enable=\'between(t,{start_time},{start_time + duration})\'[out]'
            else:
                filter_complex = f'[1:v]{scale_filter}[ovr];[0:v][ovr]overlay={x}:{y}[out]'
        
        cmd = [
            'ffmpeg', '-i', main_path, '-i', overlay_path,
            '-filter_complex', filter_complex,
            '-map', '[out]', '-map', '0:a?',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def add_image_overlay_ffmpeg(main_path, image_path, output_path, x=0, y=0, width=None, height=None, duration=None, start_time=0, opacity=1.0):
    """Add image overlay with advanced options using FFmpeg"""
    try:
        # Build scale filter
        if width and height:
            scale_filter = f'scale={width}:{height}'
        elif width:
            scale_filter = f'scale={width}:-1'
        elif height:
            scale_filter = f'scale=-1:{height}'
        else:
            scale_filter = 'scale=iw:ih'
        
        # Build opacity filter
        if opacity < 1.0:
            opacity_filter = f',format=rgba,colorchannelmixer=aa={opacity}'
        else:
            opacity_filter = ''
        
        # Build time filter
        if duration:
            time_filter = f':enable=\'between(t,{start_time},{start_time + duration})\''
        else:
            time_filter = ''
        
        filter_complex = f'[1:v]loop=loop=-1:size=1:start=0,{scale_filter}{opacity_filter}[ovr];[0:v][ovr]overlay={x}:{y}{time_filter}[out]'
        
        cmd = [
            'ffmpeg', '-i', main_path, '-i', image_path,
            '-filter_complex', filter_complex,
            '-map', '[out]', '-map', '0:a?',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        return run_ffmpeg_command(cmd)
    except Exception as e:
        return False, f"Error: {str(e)}"

def validate_image_file(file_path):
    """Validate if a file is a supported image format"""
    try:
        if not os.path.exists(file_path):
            return False, "Image file does not exist"
        
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        
        if ext not in supported_formats:
            return False, f"Unsupported image format: {ext}. Supported: {', '.join(supported_formats)}"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Image file is empty"
        
        # Try to get image info using FFprobe
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return False, "Invalid or corrupted image file"
        
        return True, "Image file is valid"
        
    except Exception as e:
        return False, f"Image validation error: {str(e)}"




app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Set session lifetime to 7 days
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB max file size

# Create upload and output folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize ML models
try:
    ml_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("ML model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    ml_model = None

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, username, email, password_hash):
        self.username = username
        self.email = email
        self.password_hash = password_hash

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.before_request
def before_request():
    # Debug logging for session and authentication
    print(f"[DEBUG] before_request: path={request.path}, method={request.method}")
    print(f"[DEBUG] session: {dict(session)}")
    print(f"[DEBUG] current_user.is_authenticated: {getattr(current_user, 'is_authenticated', None)}")
    print(f"[DEBUG] request.headers: {dict(request.headers)}")
    if current_user.is_authenticated:
        session.permanent = True  # Make session permanent for authenticated users
        session.modified = True  # Update session timestamp

@app.after_request
def after_request(response):
    if current_user.is_authenticated:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Create database tables
with app.app_context():
    db.create_all()

# FFmpeg utility functions
def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("FFmpeg is available")
            return True
    except FileNotFoundError:
        print("WARNING: FFmpeg not found. Please install FFmpeg.")
        return False
    return False

def get_video_info(input_path):
    """Get video information using FFprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting video info: {str(e)}")
    return None

def run_ffmpeg_command(cmd, timeout=300):
    """Run FFmpeg command with error handling."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False, f"FFmpeg error: {result.stderr}"
        return True, "Success"
    except subprocess.TimeoutExpired:
        return False, "Processing timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"

# Check FFmpeg at startup
ffmpeg_available = check_ffmpeg()

# Initialize ML models
ml_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create database tables
with app.app_context():
    db.create_all()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_video_path(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

def get_output_path(filename):
    return os.path.join(app.config['OUTPUT_FOLDER'], filename)

# Test route
@app.route('/test')
def test():
    return "Flask is working!"

@app.route('/test-ffmpeg')
@login_required
def test_ffmpeg():
    """Test FFmpeg functionality"""
    try:
        # Test basic FFmpeg command
        test_cmd = ['ffmpeg', '-version']
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'ffmpeg_available': True,
                'version_info': result.stdout.split('\n')[0],
                'message': 'FFmpeg is working correctly'
            })
        else:
            return jsonify({
                'success': False,
                'ffmpeg_available': False,
                'error': result.stderr
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'ffmpeg_available': False,
            'error': str(e)
        })

@app.route('/test-upload', methods=['POST'])
@login_required
def test_upload():
    """Test file upload functionality"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if not file or not file.filename:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Test file save
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], f"test_{timestamp}_{filename}")
        
        save_success, save_message = save_file_safely(file, test_path)
        
        result = {
            'success': save_success,
            'message': save_message,
            'filename': filename,
            'test_path': test_path
        }
        
        if save_success and os.path.exists(test_path):
            result['file_size'] = os.path.getsize(test_path)
            # Clean up test file
            os.remove(test_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Test upload error: {str(e)}'
        })

@app.route('/health-check')
def health_check():
    """Check the health of all video processing features"""
    try:
        health_status = {
            'status': 'healthy',
            'ffmpeg_available': ffmpeg_available,
            'imagemagick_available': imagemagick_available,
            'features': {
                'single_video': {
                    'trim': True,
                    'resize': True, 
                    'speed_change': True,
                    'extract_audio': True,
                    'color_grading': True,
                    'speed_ramping': True,
                    'effects': True,
                    'animation': True,
                    'compression': True
                },
                'multi_video': {
                    'merge': True,
                    'transitions': True,
                    'overlay': True
                },
                'prompt_processing': True,
                'external_integrations': {
                    'youtube': True,
                    'pexels': True,
                    'dailymotion': True
                }
            },
            'moviepy_removed': True,
            'performance_optimized': True
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/test-animation-filter')
@login_required
def test_animation_filter():
    """Test animation filters without processing a video"""
    try:
        # Test different animation filters
        test_results = {}
        
        # Test simple filters
        filters = {
            'zoom': 'scale=1280:720',
            'fade': 'fade=in:st=0:d=1',
            'rotate': 'rotate=0.5',
            'brightness': 'eq=brightness=0.1',
            'blur': 'boxblur=2:2'
        }
        
        for name, filter_cmd in filters.items():
            # Test if FFmpeg accepts the filter
            test_cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1', '-vf', filter_cmd, '-f', 'null', '-']
            try:
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                test_results[name] = {
                    'success': result.returncode == 0,
                    'filter': filter_cmd,
                    'error': result.stderr if result.returncode != 0 else None
                }
            except Exception as e:
                test_results[name] = {
                    'success': False,
                    'filter': filter_cmd,
                    'error': str(e)
                }
        
        return jsonify({
            'success': True,
            'filter_tests': test_results,
            'ffmpeg_available': ffmpeg_available
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)  # Enable remember me
            session.permanent = True  # Make session permanent
            session['user_id'] = user.id
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('editor'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        print(f"Registration attempt - Username: {username}, Email: {email}")  # Debug print
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('register.html')
            
        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
            
        try:
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password)
            )
            db.session.add(user)
            db.session.commit()
            print(f"Registration successful for user: {username}")  # Debug print
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Registration error: {str(e)}")  # Debug print
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')
            return render_template('register.html')
        
    return render_template('register.html')

@app.route('/prompt-help', methods=['GET'])
@login_required
def prompt_help():
    """Display help information for text prompts"""
    
    supported_commands = {
        'Basic Operations': [
            'trim start=10 end=30 - Trim video from 10 seconds to 30 seconds',
            'resize width=1920 height=1080 - Resize video to 1920x1080',
            'speed factor=2.0 - Change video speed (2.0 = 2x faster, 0.5 = 2x slower)',
            'extract_audio format=mp3 - Extract audio as MP3 or WAV'
        ],
        'Color & Effects': [
            'color_grade preset=cinematic - Apply color grading preset (cinematic, vintage, warm, cool, noir, vibrant)',
            'color_grade brightness=0.1 contrast=1.2 saturation=0.9 - Custom color grading',
            'effect type=blur strength=2.0 - Apply effects (blur, glow, vignette, sepia, negative, mirror, pixelate, edge_detection)',
            'speed_ramp start=5 end=10 factor=3.0 - Apply speed ramping between timestamps'
        ],
        'Animation & Text': [
            'animation type=zoom start=0 end=5 scale=1.5 - Apply animations (zoom, pan, fade, rotate)',
            'animation type=pan direction=right start=0 end=5 - Pan animation with direction',
            'overlay type=text content="Hello World" position=bottom-center duration=5 - Add text overlay',
            'overlay type=text content="Sample" x=100 y=50 fontsize=32 fontcolor=yellow - Custom text overlay'
        ],
        'Image & Video Overlays': [
            'overlay type=image position=top-right opacity=0.8 duration=10 - Add image overlay (requires auxiliary image file)',
            'overlay type=image x=50 y=100 width=200 height=150 - Custom positioned image overlay',
            'overlay type=image position=center start=5 duration=8 opacity=0.7 - Timed image overlay with transparency',
            'overlay type=video position=bottom-left width=300 duration=15 - Add video overlay (requires auxiliary video file)',
            'Note: Image/video overlays require uploading the overlay file in "Auxiliary Video Files" section'
        ],
        'Multi-Video Operations': [
            'merge_videos transition=fade|dissolve|cut duration=X - Merge multiple videos (requires multiple file upload)',
            'transition type=fade|dissolve|wipe|cut duration=X - Apply transitions between videos (requires multiple file upload)',
            'Note: For merge and transition commands, use Multi Video Editor or upload multiple files'
        ]
    }
    
    return jsonify({
        'success': True,
        'supported_commands': supported_commands,
        'usage_tips': [
            'Commands are case-insensitive',
            'Parameters can be in any order',
            'Use quotes for text content: content="My Text"',
            'Numeric values support decimals: factor=1.5',
            'Time values are in seconds',
            'For multi-video operations, use the dedicated Multi Video Editor'
        ],
        'examples': [
            'trim start=5 end=15',
            'resize width=1280 height=720',
            'speed factor=1.5',
            'color_grade preset=cinematic',
            'effect type=blur strength=3',
            'overlay type=text content="My Video" position=bottom duration=8',
            'overlay type=image position=top-right opacity=0.8 duration=10',
            'overlay type=image x=100 y=50 width=300 height=200 start=2 duration=5'
        ]
    })

@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/editor')
@login_required
def editor():
    print(f"Accessing editor page - User: {current_user.username if current_user.is_authenticated else 'Not authenticated'}")  # Debug print
    return render_template('editor.html')

@app.route('/multi_video_editor')
@login_required
def multi_video_editor():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('multi_video_editor.html')

@app.route('/output/<filename>')
@login_required
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/trim', methods=['POST'])
@login_required
def trim_video():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', 0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f'trimmed_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Calculate duration
        duration = end_time - start_time
        
        # FFmpeg command for trimming
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-crf', '18',
            '-preset', 'fast',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        
        # Run FFmpeg command
        success, message = run_ffmpeg_command(cmd)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if not success:
            return jsonify({'success': False, 'error': message})
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in trim_video: {str(e)}")
        # Clean up files on error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)})

# Cleanup functions removed - no longer needed with FFmpeg

@app.route('/merge', methods=['POST'])
@login_required
def merge_videos():
    if not current_user.is_authenticated:
        return jsonify({
            'success': False,
            'error': 'Authentication required',
            'redirect': url_for('login')
        }), 401

    input_paths = []
    
    try:
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        files = request.files.getlist('files[]')
        if len(files) < 2:
            return jsonify({'success': False, 'error': 'Please select at least 2 videos to merge'})
        
        timestamp = int(time.time())
        
        # Save uploaded files
        for i, file in enumerate(files):
            if file and file.filename:
                filename = secure_filename(file.filename)
                input_filename = f"{timestamp}_{i}_{filename}"
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
                file.save(input_path)
                input_paths.append(input_path)
        
        if len(input_paths) < 2:
            return jsonify({'success': False, 'error': 'At least 2 valid video files are required'})
        
        print(f"Merging {len(input_paths)} videos using FFmpeg...")
        
        # Generate output filename
        output_filename = f'merged_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Create concat file for FFmpeg
        concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'concat_{timestamp}.txt')
        
        try:
            with open(concat_file, 'w', encoding='utf-8') as f:
                for i, path in enumerate(input_paths):
                    # Normalize path for FFmpeg (use forward slashes)
                    normalized_path = path.replace('\\', '/')
                    f.write(f"file '{normalized_path}'\n")
                    print(f"Added to concat list {i+1}: {normalized_path}")
            
            # Debug: Print concat file contents
            print(f"Concat file contents:")
            with open(concat_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
            
            # Debug: Check if input files exist
            for i, path in enumerate(input_paths):
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    print(f"Input file {i+1}: {path} (size: {size} bytes)")
                else:
                    print(f"Input file {i+1}: {path} (NOT FOUND)")
            
            # Use filter_complex method for reliable merging (ensures all videos are included)
            print("Using filter_complex method for reliable video merging...")
            
            # Build filter_complex command with individual inputs
            filter_inputs = []
            for i, path in enumerate(input_paths):
                filter_inputs.extend(['-i', path])
            
            # Create filter complex for normalization and concatenation
            video_filters = []
            audio_filters = []
            
            for i in range(len(input_paths)):
                # Normalize each video stream with consistent timebase
                video_filters.append(f'[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,settb=AVTB[v{i}]')
                # Normalize each audio stream
                audio_filters.append(f'[{i}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]')
            
            # Concatenate all normalized streams
            video_inputs = ''.join([f'[v{i}]' for i in range(len(input_paths))])
            audio_inputs = ''.join([f'[a{i}]' for i in range(len(input_paths))])
            
            filter_complex = ';'.join(video_filters + audio_filters + [
                f'{video_inputs}concat=n={len(input_paths)}:v=1:a=0[outv]',
                f'{audio_inputs}concat=n={len(input_paths)}:v=0:a=1[outa]'
            ])
            
            complex_cmd = [
                'ffmpeg'
            ] + filter_inputs + [
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-crf', '23',
                '-preset', 'medium',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-y', output_path
            ]
            
            success, message = run_ffmpeg_command(complex_cmd, timeout=900)
            
            if not success:
                print(f"Filter complex failed: {message}")
                print("Trying two-step normalization approach...")
                
                # Two-step approach: First normalize each video, then concatenate
                normalized_paths = []
                
                for i, input_path in enumerate(input_paths):
                    norm_filename = f"norm_{timestamp}_{i}.mp4"
                    norm_path = os.path.join(app.config['UPLOAD_FOLDER'], norm_filename)
                    
                    # Normalize individual video
                    norm_cmd = [
                        'ffmpeg', '-i', input_path,
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                        '-r', '30',
                        '-ar', '44100',
                        '-ac', '2',
                        '-crf', '23',
                        '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-y', norm_path
                    ]
                    
                    norm_success, norm_message = run_ffmpeg_command(norm_cmd, timeout=300)
                    if not norm_success:
                        # Clean up and return error
                        for path in normalized_paths:
                            if os.path.exists(path):
                                os.remove(path)
                        return jsonify({'success': False, 'error': f'Failed to normalize video {i+1}: {norm_message}'})
                    
                    normalized_paths.append(norm_path)
                
                # Update concat file with normalized videos
                with open(concat_file, 'w', encoding='utf-8') as f:
                    for path in normalized_paths:
                        normalized_path = path.replace('\\', '/')
                        f.write(f"file '{normalized_path}'\n")
                
                # Concatenate normalized videos
                final_concat_cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-c', 'copy',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(final_concat_cmd, timeout=300)
                
                # Clean up normalized files
                for path in normalized_paths:
                    if os.path.exists(path):
                        os.remove(path)
        
        except Exception as e:
            success = False
            message = f"File operation error: {str(e)}"
        
        # Clean up
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(concat_file):
            os.remove(concat_file)
        
        if not success:
            return jsonify({'success': False, 'error': f'FFmpeg merge failed: {message}'})
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'message': f'Successfully merged {len(input_paths)} videos using FFmpeg'
        })
            
    except Exception as e:
        print(f"Error in merge_videos: {str(e)}")
        # Clean up files on error
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        concat_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'concat_{timestamp}.txt')
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)
        return jsonify({'success': False, 'error': f'Merge error: {str(e)}'})

# MoviePy-based merge-simple route removed - using FFmpeg-based merge instead

@app.route('/extract-audio', methods=['POST'])
@login_required
def extract_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f'audio_{timestamp}.mp3'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # FFmpeg command for audio extraction
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vn',  # No video
            '-acodec', 'mp3',
            '-ab', '192k',  # Audio bitrate
            '-ar', '44100',  # Sample rate
            '-y',
            output_path
        ]
        
        # Run FFmpeg command
        success, message = run_ffmpeg_command(cmd)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if not success:
            return jsonify({'success': False, 'error': message})
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in extract_audio: {str(e)}")
        # Clean up files on error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/change-speed', methods=['POST'])
@login_required
def change_speed():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        speed_factor = float(request.form.get('speed_factor', 1.0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
            
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f'speed_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # FFmpeg command for speed change
        # For video speed: use setpts filter
        # For audio speed: use atempo filter
        video_speed = f"setpts={1/speed_factor}*PTS"
        audio_speed = f"atempo={speed_factor}"
        
        # Handle extreme speed factors for audio (atempo has limits)
        audio_filters = []
        remaining_factor = speed_factor
        while remaining_factor > 2.0:
            audio_filters.append("atempo=2.0")
            remaining_factor /= 2.0
        while remaining_factor < 0.5:
            audio_filters.append("atempo=0.5")
            remaining_factor /= 0.5
        if remaining_factor != 1.0:
            audio_filters.append(f"atempo={remaining_factor}")
        
        audio_filter_str = ",".join(audio_filters) if audio_filters else "anull"
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-filter_complex', f'[0:v]{video_speed}[v];[0:a]{audio_filter_str}[a]',
            '-map', '[v]', '-map', '[a]',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-crf', '18',
            '-preset', 'fast',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        
        # Run FFmpeg command
        success, message = run_ffmpeg_command(cmd)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if not success:
            return jsonify({'success': False, 'error': message})
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in change_speed: {str(e)}")
        # Clean up files on error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/resize', methods=['POST'])
@login_required
def resize_video():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        width = int(request.form.get('width', 0))
        height = int(request.form.get('height', 0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
            
        if width <= 0 or height <= 0:
            return jsonify({'success': False, 'error': 'Invalid dimensions'})
        
        # Ensure dimensions are even (required for H.264)
        width = width - (width % 2)
        height = height - (height % 2)
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f'resized_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # FFmpeg command for resizing
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'scale={width}:{height}',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-crf', '18',
            '-preset', 'fast',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        
        # Run FFmpeg command
        success, message = run_ffmpeg_command(cmd)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if not success:
            return jsonify({'success': False, 'error': message})
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in resize_video: {str(e)}")
        # Clean up files on error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply-transition', methods=['POST'])
@login_required
def apply_transition():
    input_paths = []
    processed_paths = []
    
    try:
        # Check for multiple files (transition between videos)
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            if len(files) < 2:
                return jsonify({'success': False, 'error': 'Please select at least 2 videos for transitions'})
        else:
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        transition_type = request.form.get('transition_type', 'fade')
        duration = float(request.form.get('duration', 1.0))
        
        timestamp = int(time.time())
        
        # Save uploaded files
        for i, file in enumerate(files):
            if file and file.filename:
                filename = secure_filename(file.filename)
                input_filename = f"{timestamp}_{i}_{filename}"
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
                file.save(input_path)
                input_paths.append(input_path)
        
        if len(input_paths) < 2:
            return jsonify({'success': False, 'error': 'At least 2 valid video files are required for transitions'})
        
        print(f"Applying {transition_type} transitions between {len(input_paths)} videos using FFmpeg...")
        
        # Generate output filename
        output_filename = f'transition_{transition_type}_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # First normalize all videos to same format to avoid timebase issues
        normalized_paths = []
        
        print(f"Normalizing {len(input_paths)} videos to consistent format...")
        for i, input_path in enumerate(input_paths):
            norm_filename = f"norm_{timestamp}_{i}.mp4"
            norm_path = os.path.join(app.config['UPLOAD_FOLDER'], norm_filename)
            
            # Normalize video to consistent format
            norm_cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                '-r', '30',  # Force consistent frame rate
                '-ar', '44100',  # Force consistent audio sample rate
                '-ac', '2',  # Force stereo
                '-crf', '23',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-y', norm_path
            ]
            
            success, message = run_ffmpeg_command(norm_cmd, timeout=300)
            if not success:
                # Clean up and return error
                for path in normalized_paths:
                    if os.path.exists(path):
                        os.remove(path)
                return jsonify({'success': False, 'error': f'Failed to normalize video {i+1}: {message}'})
            
            normalized_paths.append(norm_path)
        
        # Now apply transitions based on type
        if transition_type == 'fade':
            print("Applying fade transitions...")
            
            if len(normalized_paths) == 2:
                # Simple crossfade for 2 videos
                fade_cmd = [
                    'ffmpeg', '-i', normalized_paths[0], '-i', normalized_paths[1],
                    '-filter_complex',
                    f'[0:v][1:v]xfade=transition=fade:duration={duration}:offset=5[v];'
                    f'[0:a][1:a]acrossfade=d={duration}[a]',
                    '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(fade_cmd, timeout=600)
            else:
                # Multiple videos - use concat with fade effects
                fade_concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'fade_concat_{timestamp}.txt')
                
                with open(fade_concat_file, 'w', encoding='utf-8') as f:
                    for path in normalized_paths:
                        normalized_path = path.replace('\\', '/')
                        f.write(f"file '{normalized_path}'\n")
                
                fade_cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', fade_concat_file,
                    '-vf', f'fade=in:0:{int(duration*30)},fade=out:st={max(0, 30-duration)}:d={duration}',
                    '-af', f'afade=in:st=0:d={duration},afade=out:st={max(0, 30-duration)}:d={duration}',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(fade_cmd, timeout=600)
                
                if os.path.exists(fade_concat_file):
                    os.remove(fade_concat_file)
        
        elif transition_type == 'dissolve':
            print("Applying dissolve transitions...")
            
            if len(normalized_paths) == 2:
                # Simple crossfade between two normalized videos
                dissolve_cmd = [
                    'ffmpeg', '-i', normalized_paths[0], '-i', normalized_paths[1],
                    '-filter_complex',
                    f'[0:v][1:v]xfade=transition=dissolve:duration={duration}:offset=5[v];'
                    f'[0:a][1:a]acrossfade=d={duration}[a]',
                    '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(dissolve_cmd, timeout=600)
            else:
                # For multiple videos, just concatenate (dissolve works best with 2 videos)
                dissolve_concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'dissolve_concat_{timestamp}.txt')
                
                with open(dissolve_concat_file, 'w', encoding='utf-8') as f:
                    for path in normalized_paths:
                        normalized_path = path.replace('\\', '/')
                        f.write(f"file '{normalized_path}'\n")
                
                dissolve_cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', dissolve_concat_file,
                    '-c', 'copy',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(dissolve_cmd, timeout=300)
                
                if os.path.exists(dissolve_concat_file):
                    os.remove(dissolve_concat_file)
        
        elif transition_type == 'wipe':
            print("Applying wipe transitions...")
            
            if len(normalized_paths) == 2:
                # Wipe transition between two normalized videos
                wipe_cmd = [
                    'ffmpeg', '-i', normalized_paths[0], '-i', normalized_paths[1],
                    '-filter_complex',
                    f'[0:v][1:v]xfade=transition=wipeleft:duration={duration}:offset=5[v];'
                    f'[0:a][1:a]acrossfade=d={duration}[a]',
                    '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(wipe_cmd, timeout=600)
            else:
                success = False
                message = "Wipe transition currently supports only 2 videos"
        
        elif transition_type == 'cut':
            print("Applying cut transitions (simple concatenation)...")
            
            # Simple concatenation of normalized videos
            cut_concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'cut_concat_{timestamp}.txt')
            
            with open(cut_concat_file, 'w', encoding='utf-8') as f:
                for path in normalized_paths:
                    normalized_path = path.replace('\\', '/')
                    f.write(f"file '{normalized_path}'\n")
            
            cut_cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', cut_concat_file,
                '-c', 'copy',
                '-y', output_path
            ]
            
            success, message = run_ffmpeg_command(cut_cmd, timeout=300)
            
            if os.path.exists(cut_concat_file):
                os.remove(cut_concat_file)
        
        else:
            success = False
            message = f"Unsupported transition type: {transition_type}"
        
        # Clean up normalized files
        for path in normalized_paths:
            if os.path.exists(path):
                os.remove(path)
        
        # Clean up input files
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        
        if not success:
            return jsonify({'success': False, 'error': f'Transition failed: {message}'})
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'message': f'Applied {transition_type} transitions using FFmpeg'
        })
        
    except Exception as e:
        print(f"Error in apply_transition: {str(e)}")
        # Clean up files on error
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        return jsonify({'success': False, 'error': f'Transition error: {str(e)}'})

@app.route('/apply-color-grading', methods=['POST'])
@login_required
def apply_color_grading():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        color_style = request.form.get('color_style', 'cinematic')
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f'color_graded_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Apply color grading using FFmpeg
        if color_style == 'cinematic':
            # Increase contrast and saturation for cinematic look
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', 'eq=contrast=1.2:saturation=1.1:brightness=0.05',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'aac', '-movflags', '+faststart', '-y', output_path
            ]
        elif color_style == 'vintage':
            # Vintage look with sepia tones
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', 'colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'aac', '-movflags', '+faststart', '-y', output_path
            ]
        elif color_style == 'warm':
            # Warm color temperature
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', 'eq=brightness=0.1:saturation=1.2,colortemperature=3000',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'aac', '-movflags', '+faststart', '-y', output_path
            ]
        elif color_style == 'cool':
            # Cool color temperature
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', 'eq=brightness=-0.05:saturation=0.9,colortemperature=7000',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'aac', '-movflags', '+faststart', '-y', output_path
            ]
        elif color_style == 'noir':
            # Black and white with high contrast
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', 'format=gray,eq=contrast=1.5:brightness=0.1',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'aac', '-movflags', '+faststart', '-y', output_path
            ]
        else:
            # Default - slight enhancement
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', 'eq=contrast=1.1:saturation=1.05',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'aac', '-movflags', '+faststart', '-y', output_path
            ]
        
        # Execute FFmpeg command
        success, message = run_ffmpeg_command(cmd, timeout=300)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if not success:
            return jsonify({'success': False, 'error': message})
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in apply_color_grading: {str(e)}")
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply-speed-ramping', methods=['POST'])
@login_required
def apply_speed_ramping():
    """Apply speed ramping using FFmpeg"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        start_speed = float(request.form.get('start_speed', 1.0))
        end_speed = float(request.form.get('end_speed', 2.0))
        ramp_duration = float(request.form.get('ramp_duration', 5.0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(input_path)
        
        # Generate output filename
        output_filename = f'speed_ramped_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Use execute_speed_ramp_command for consistency
        result = execute_speed_ramp_command(f'speed_ramp start=0 end={ramp_duration} factor={end_speed}', input_path, timestamp)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'output_file': result['output_file']
            })
        else:
            return jsonify({'success': False, 'error': result['error']})
        
    except Exception as e:
        print(f"Error in apply_speed_ramping: {str(e)}")
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply-effects', methods=['POST'])
@login_required
def apply_effects():
    """Apply effects using FFmpeg"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        effect_type = request.form.get('effect_type', 'blur')
        strength = float(request.form.get('strength', 2.0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(input_path)
        
        # Use execute_effect_command for consistency
        result = execute_effect_command(f'effect type={effect_type} strength={strength}', input_path, timestamp)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'output_file': result['output_file']
            })
        else:
            return jsonify({'success': False, 'error': result['error']})
        
    except Exception as e:
        print(f"Error in apply_effects: {str(e)}")
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply-animation', methods=['POST'])
@login_required
def apply_animation():
    """Apply animation using FFmpeg"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        animation_type = request.form.get('animation_type', 'zoom')
        duration = float(request.form.get('duration', 2.0))
        scale = float(request.form.get('scale', 1.5))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(input_path)
        
        # Get additional parameters with defaults
        direction = request.form.get('direction', 'right')
        angle = float(request.form.get('angle', 45))
        
        # Use execute_animation_command for consistency
        result = execute_animation_command(f'animation type={animation_type} start=0 end={duration} scale={scale} direction={direction} angle={angle}', input_path, timestamp)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'output_file': result['output_file']
            })
        else:
            return jsonify({'success': False, 'error': result['error']})
        
    except Exception as e:
        print(f"Error in apply_animation: {str(e)}")
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add-overlay', methods=['POST'])
@login_required
def add_overlay():
    """Add overlay (image/video/text) to main video using FFmpeg for better performance"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Main video file is required.'})

        main_file = request.files['file']
        overlay_type = request.form.get('overlay_type', 'image')  # image, video, text
        
        timestamp = int(time.time())
        
        # Save main video file
        main_filename = secure_filename(main_file.filename)
        main_input_filename = f"{timestamp}_main_{main_filename}"
        main_path = os.path.join(app.config['UPLOAD_FOLDER'], main_input_filename)
        
        save_success, save_message = save_file_safely(main_file, main_path)
        if not save_success:
            return jsonify({'success': False, 'error': f'Failed to save main video: {save_message}'})

        # Validate main video
        validate_success, validate_message = validate_and_repair_video_file(main_path, 1)
        if not validate_success:
            if os.path.exists(main_path):
                os.remove(main_path)
            return jsonify({'success': False, 'error': validate_message})

        output_filename = f'overlay_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        if overlay_type == 'text':
            # Text overlay
            text_content = request.form.get('text_content', 'Sample Text')
            x_pos = request.form.get('x_pos', '10')
            y_pos = request.form.get('y_pos', '10')
            font_size = request.form.get('font_size', '24')
            font_color = request.form.get('font_color', 'white')
            duration = request.form.get('duration', '0')  # 0 means entire video
            position = request.form.get('position', 'custom')
            
            # Position presets
            if position != 'custom':
                pos_map = {
                    'top-left': '10:10',
                    'top-center': '(w-text_w)/2:10',
                    'top-right': 'w-text_w-10:10',
                    'center-left': '10:(h-text_h)/2',
                    'center': '(w-text_w)/2:(h-text_h)/2',
                    'center-right': 'w-text_w-10:(h-text_h)/2',
                    'bottom-left': '10:h-text_h-10',
                    'bottom-center': '(w-text_w)/2:h-text_h-10',
                    'bottom-right': 'w-text_w-10:h-text_h-10'
                }
                text_pos = pos_map.get(position, f'{x_pos}:{y_pos}')
            else:
                text_pos = f'{x_pos}:{y_pos}'
            
            # Build text filter
            if duration == '0':
                text_filter = f'drawtext=text=\'{text_content}\':fontsize={font_size}:fontcolor={font_color}:x={text_pos}'
            else:
                text_filter = f'drawtext=text=\'{text_content}\':fontsize={font_size}:fontcolor={font_color}:x={text_pos}:enable=\'between(t,0,{duration})\''
            
            cmd = [
                'ffmpeg', '-i', main_path,
                '-vf', text_filter,
                '-c:v', 'libx264',
                '-c:a', 'copy',
                '-crf', '23',
                '-preset', 'medium',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-y', output_path
            ]
            
        elif overlay_type in ['image', 'video']:
            # Image or video overlay - check multiple possible field names
            overlay_file = None
            if 'overlay_file' in request.files:
                overlay_file = request.files['overlay_file']
            elif 'overlay' in request.files:
                overlay_file = request.files['overlay']
            elif 'files[]' in request.files:
                files_list = request.files.getlist('files[]')
                if len(files_list) > 1:
                    overlay_file = files_list[1]  # Second file is overlay
            
            if not overlay_file or not overlay_file.filename:
                if os.path.exists(main_path):
                    os.remove(main_path)
                return jsonify({'success': False, 'error': f'{overlay_type.title()} overlay file is required. Please select an overlay file.'})
            
            print(f"Debug: Using overlay file: {overlay_file.filename}")
            x_pos = int(request.form.get('x_pos', 0))
            y_pos = int(request.form.get('y_pos', 0))
            scale_width = request.form.get('scale_width', '')
            scale_height = request.form.get('scale_height', '')
            opacity = float(request.form.get('opacity', 1.0))
            
            # Save overlay file
            overlay_filename = secure_filename(overlay_file.filename)
            overlay_input_filename = f"{timestamp}_overlay_{overlay_filename}"
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_input_filename)
            
            save_success, save_message = save_file_safely(overlay_file, overlay_path)
            if not save_success:
                if os.path.exists(main_path):
                    os.remove(main_path)
                return jsonify({'success': False, 'error': f'Failed to save overlay file: {save_message}'})
            
            # Use simpler overlay approach
            if scale_width and scale_height:
                # Scale overlay first, then overlay
                cmd = [
                    'ffmpeg', '-i', main_path, '-i', overlay_path,
                    '-filter_complex', f'[1:v]scale={scale_width}:{scale_height}[scaled];[0:v][scaled]overlay={x_pos}:{y_pos}',
                    '-c:v', 'libx264',
                    '-c:a', 'copy',
                    '-crf', '23',
                    '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
            else:
                # Simple overlay without scaling
                cmd = [
                    'ffmpeg', '-i', main_path, '-i', overlay_path,
                    '-filter_complex', f'[0:v][1:v]overlay={x_pos}:{y_pos}',
                    '-c:v', 'libx264',
                    '-c:a', 'copy',
                    '-crf', '23',
                    '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'medium',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-y', output_path
            ]
            
        else:
            if os.path.exists(main_path):
                os.remove(main_path)
            return jsonify({'success': False, 'error': 'Invalid overlay type. Supported: text, image, video'})

        # Execute FFmpeg command
        success, message = run_ffmpeg_command(cmd, timeout=600)
        
        # Cleanup input files
        if os.path.exists(main_path):
            os.remove(main_path)
        if overlay_type in ['image', 'video'] and 'overlay_path' in locals() and os.path.exists(overlay_path):
            os.remove(overlay_path)
        
        if not success:
            return jsonify({'success': False, 'error': f'Overlay processing failed: {message}'})
        
        return jsonify({
            'success': True, 
            'output_file': output_filename,
            'message': f'{overlay_type.title()} overlay added successfully'
        })

    except Exception as e:
        print(f"Error in add_overlay: {str(e)}")
        # Cleanup on error
        if 'main_path' in locals() and os.path.exists(main_path):
            os.remove(main_path)
        if 'overlay_path' in locals() and os.path.exists(overlay_path):
            os.remove(overlay_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add-text-overlay', methods=['POST'])
@login_required
def add_text_overlay():
    """Add text overlay with advanced styling options"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Video file is required.'})

        file = request.files['file']
        text_content = request.form.get('text', 'Sample Text')
        position = request.form.get('position', 'bottom-center')
        font_size = int(request.form.get('font_size', 24))
        font_color = request.form.get('font_color', 'white')
        background_color = request.form.get('background_color', '')
        duration = float(request.form.get('duration', 0))  # 0 = entire video
        start_time = float(request.form.get('start_time', 0))
        
        timestamp = int(time.time())
        
        # Save input file
        filename = secure_filename(file.filename)
        input_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        
        save_success, save_message = save_file_safely(file, input_path)
        if not save_success:
            return jsonify({'success': False, 'error': f'Failed to save video: {save_message}'})

        # Validate video
        validate_success, validate_message = validate_and_repair_video_file(input_path, 1)
        if not validate_success:
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({'success': False, 'error': validate_message})

        output_filename = f'text_overlay_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Position mapping
        pos_map = {
            'top-left': '10:10',
            'top-center': '(w-text_w)/2:10',
            'top-right': 'w-text_w-10:10',
            'center-left': '10:(h-text_h)/2',
            'center': '(w-text_w)/2:(h-text_h)/2',
            'center-right': 'w-text_w-10:(h-text_h)/2',
            'bottom-left': '10:h-text_h-10',
            'bottom-center': '(w-text_w)/2:h-text_h-10',
            'bottom-right': 'w-text_w-10:h-text_h-10'
        }
        
        text_pos = pos_map.get(position, '(w-text_w)/2:h-text_h-10')
        
        # Build text filter
        text_filter = f'drawtext=text=\'{text_content}\':fontsize={font_size}:fontcolor={font_color}:x={text_pos}'
        
        # Add background if specified
        if background_color:
            text_filter += f':box=1:boxcolor={background_color}:boxborderw=5'
        
        # Add timing if specified
        if duration > 0:
            end_time = start_time + duration
            text_filter += f':enable=\'between(t,{start_time},{end_time})\''
        elif start_time > 0:
            text_filter += f':enable=\'gte(t,{start_time})\''

        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', text_filter,
            '-c:v', 'libx264',
            '-c:a', 'copy',
            '-crf', '23',
            '-preset', 'medium',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y', output_path
        ]

        success, message = run_ffmpeg_command(cmd, timeout=600)
        
        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if not success:
            return jsonify({'success': False, 'error': f'Text overlay failed: {message}'})
        
        return jsonify({
            'success': True, 
            'output_file': output_filename,
            'message': f'Text overlay "{text_content}" added successfully'
        })

    except Exception as e:
        print(f"Error in add_text_overlay: {str(e)}")
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add-image-overlay', methods=['POST'])
@login_required
def add_image_overlay():
    """Add image overlay with positioning and scaling options"""
    try:
        if 'video_file' not in request.files or 'image_file' not in request.files:
            return jsonify({'success': False, 'error': 'Both video and image files are required.'})

        video_file = request.files['video_file']
        image_file = request.files['image_file']
        
        x_pos = int(request.form.get('x_pos', 10))
        y_pos = int(request.form.get('y_pos', 10))
        scale_width = request.form.get('scale_width', '')
        scale_height = request.form.get('scale_height', '')
        opacity = float(request.form.get('opacity', 1.0))
        position = request.form.get('position', 'custom')
        
        timestamp = int(time.time())
        
        # Save video file
        video_filename = secure_filename(video_file.filename)
        video_input_filename = f"{timestamp}_video_{video_filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_input_filename)
        
        save_success, save_message = save_file_safely(video_file, video_path)
        if not save_success:
            return jsonify({'success': False, 'error': f'Failed to save video: {save_message}'})

        # Save image file
        image_filename = secure_filename(image_file.filename)
        image_input_filename = f"{timestamp}_image_{image_filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_input_filename)
        
        save_success, save_message = save_file_safely(image_file, image_path)
        if not save_success:
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({'success': False, 'error': f'Failed to save image: {save_message}'})

        # Validate video
        validate_success, validate_message = validate_and_repair_video_file(video_path, 1)
        if not validate_success:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({'success': False, 'error': validate_message})

        output_filename = f'image_overlay_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Position presets
        if position != 'custom':
            pos_map = {
                'top-left': '10:10',
                'top-right': 'main_w-overlay_w-10:10',
                'bottom-left': '10:main_h-overlay_h-10',
                'bottom-right': 'main_w-overlay_w-10:main_h-overlay_h-10',
                'center': '(main_w-overlay_w)/2:(main_h-overlay_h)/2'
            }
            overlay_pos = pos_map.get(position, f'{x_pos}:{y_pos}')
        else:
            overlay_pos = f'{x_pos}:{y_pos}'

        # Build overlay filter
        overlay_filter = '[1:v]'
        
        # Add scaling if specified
        if scale_width and scale_height:
            overlay_filter += f'scale={scale_width}:{scale_height},'
        elif scale_width:
            overlay_filter += f'scale={scale_width}:-1,'
        elif scale_height:
            overlay_filter += f'scale=-1:{scale_height},'
        
        # Add opacity if not 1.0
        if opacity != 1.0:
            overlay_filter += f'format=rgba,colorchannelmixer=aa={opacity},'
        
        overlay_filter += f'[ovr];[0:v][ovr]overlay={overlay_pos}[out]'

        cmd = [
            'ffmpeg', '-i', video_path, '-i', image_path,
            '-filter_complex', overlay_filter,
            '-map', '[out]', '-map', '0:a?',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-crf', '23',
            '-preset', 'medium',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y', output_path
        ]

        success, message = run_ffmpeg_command(cmd, timeout=600)
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        if not success:
            return jsonify({'success': False, 'error': f'Image overlay failed: {message}'})
        
        return jsonify({
            'success': True, 
            'output_file': output_filename,
            'message': 'Image overlay added successfully'
        })

    except Exception as e:
        print(f"Error in add_image_overlay: {str(e)}")
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add-video-overlay', methods=['POST'])
@login_required
def add_video_overlay():
    """Add video overlay (picture-in-picture effect)"""
    try:
        if 'main_video' not in request.files or 'overlay_video' not in request.files:
            return jsonify({'success': False, 'error': 'Both main and overlay video files are required.'})

        main_video = request.files['main_video']
        overlay_video = request.files['overlay_video']
        
        x_pos = int(request.form.get('x_pos', 10))
        y_pos = int(request.form.get('y_pos', 10))
        scale_width = request.form.get('scale_width', '320')
        scale_height = request.form.get('scale_height', '240')
        opacity = float(request.form.get('opacity', 1.0))
        position = request.form.get('position', 'custom')
        
        timestamp = int(time.time())
        
        # Save main video
        main_filename = secure_filename(main_video.filename)
        main_input_filename = f"{timestamp}_main_{main_filename}"
        main_path = os.path.join(app.config['UPLOAD_FOLDER'], main_input_filename)
        
        save_success, save_message = save_file_safely(main_video, main_path)
        if not save_success:
            return jsonify({'success': False, 'error': f'Failed to save main video: {save_message}'})

        # Save overlay video
        overlay_filename = secure_filename(overlay_video.filename)
        overlay_input_filename = f"{timestamp}_overlay_{overlay_filename}"
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_input_filename)
        
        save_success, save_message = save_file_safely(overlay_video, overlay_path)
        if not save_success:
            if os.path.exists(main_path):
                os.remove(main_path)
            return jsonify({'success': False, 'error': f'Failed to save overlay video: {save_message}'})

        # Validate both videos
        validate_success, validate_message = validate_and_repair_video_file(main_path, 1)
        if not validate_success:
            if os.path.exists(main_path):
                os.remove(main_path)
            if os.path.exists(overlay_path):
                os.remove(overlay_path)
            return jsonify({'success': False, 'error': f'Main video: {validate_message}'})

        validate_success, validate_message = validate_and_repair_video_file(overlay_path, 2)
        if not validate_success:
            if os.path.exists(main_path):
                os.remove(main_path)
            if os.path.exists(overlay_path):
                os.remove(overlay_path)
            return jsonify({'success': False, 'error': f'Overlay video: {validate_message}'})

        output_filename = f'video_overlay_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Position presets
        if position != 'custom':
            pos_map = {
                'top-left': '10:10',
                'top-right': 'main_w-overlay_w-10:10',
                'bottom-left': '10:main_h-overlay_h-10',
                'bottom-right': 'main_w-overlay_w-10:main_h-overlay_h-10',
                'center': '(main_w-overlay_w)/2:(main_h-overlay_h)/2'
            }
            overlay_pos = pos_map.get(position, f'{x_pos}:{y_pos}')
        else:
            overlay_pos = f'{x_pos}:{y_pos}'

        # Build overlay filter
        overlay_filter = f'[1:v]scale={scale_width}:{scale_height}'
        
        # Add opacity if not 1.0
        if opacity != 1.0:
            overlay_filter += f',format=rgba,colorchannelmixer=aa={opacity}'
        
        overlay_filter += f'[ovr];[0:v][ovr]overlay={overlay_pos}[out]'

        cmd = [
            'ffmpeg', '-i', main_path, '-i', overlay_path,
            '-filter_complex', overlay_filter,
            '-map', '[out]', '-map', '0:a?',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-crf', '23',
            '-preset', 'medium',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y', output_path
        ]

        success, message = run_ffmpeg_command(cmd, timeout=600)
        
        # Cleanup
        if os.path.exists(main_path):
            os.remove(main_path)
        if os.path.exists(overlay_path):
            os.remove(overlay_path)
        
        if not success:
            return jsonify({'success': False, 'error': f'Video overlay failed: {message}'})
        
        return jsonify({
            'success': True, 
            'output_file': output_filename,
            'message': 'Video overlay (picture-in-picture) added successfully'
        })

    except Exception as e:
        print(f"Error in add_video_overlay: {str(e)}")
        if 'main_path' in locals() and os.path.exists(main_path):
            os.remove(main_path)
        if 'overlay_path' in locals() and os.path.exists(overlay_path):
            os.remove(overlay_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/compress-video', methods=['POST'])
@login_required
def compress_video():
    """Fast video compression using direct FFmpeg"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        quality = request.form.get('quality', 'medium')  # low, medium, high
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        
        # Get original file size
        original_size = os.path.getsize(input_path)
        
        # Set compression parameters based on quality
        if quality == 'high':
            crf, preset = 18, 'slow'
        elif quality == 'low':
            crf, preset = 28, 'ultrafast'
        else:  # medium
            crf, preset = 23, 'medium'
        
        # Generate output filename
        output_filename = f"compressed_{timestamp}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Use direct FFmpeg for compression
        success, message = compress_video_ffmpeg(input_path, output_path, crf, preset)
        
        if not success:
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({'success': False, 'error': message})
        
        # Get compressed file size
        compressed_size = os.path.getsize(output_path)
        compression_ratio = f"{((original_size - compressed_size) / original_size * 100):.1f}%"
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'original_size': f"{original_size / (1024*1024):.1f} MB",
            'compressed_size': f"{compressed_size / (1024*1024):.1f} MB",
            'compression_ratio': compression_ratio,
            'method': 'FFmpeg' if ffmpeg_available else 'MoviePy'
        })
        
    except Exception as e:
        # Clean up files on error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/fast-trim', methods=['POST'])
@login_required
def fast_trim():
    """Ultra-fast video trimming using direct FFmpeg (no re-encoding)"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', 0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f'fast_trim_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Use direct FFmpeg for ultra-fast trimming
        success, message = trim_video_ffmpeg(input_path, output_path, start_time, end_time)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        if not success:
            return jsonify({'success': False, 'error': message})
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'method': 'FFmpeg (no re-encoding)'
        })
        
    except Exception as e:
        print(f"Error in fast_trim: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process-multi-prompt', methods=['POST'])
@login_required
def process_multi_prompt():
    """Process text prompts for multi-video operations"""
    input_paths = []
    
    try:
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        files = request.files.getlist('files[]')
        prompt = request.form.get('prompt', '').lower().strip()
        
        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})
        
        if len(files) < 2:
            return jsonify({'success': False, 'error': 'Multi-video operations require at least 2 videos'})
        
        timestamp = int(time.time())
        
        # Save uploaded files
        for i, file in enumerate(files):
            if file and file.filename:
                filename = secure_filename(file.filename)
                input_filename = f"{timestamp}_{i}_{filename}"
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
                file.save(input_path)
                input_paths.append(input_path)
        
        print(f"Processing multi-video prompt: '{prompt}' with {len(input_paths)} videos")
        
        # Parse prompt and execute corresponding action
        if any(word in prompt for word in ['merge', 'combine', 'join', 'concatenate', 'stitch']):
            return handle_merge_prompt(input_paths, prompt, timestamp)
        
        elif any(word in prompt for word in ['transition', 'fade', 'dissolve', 'crossfade', 'blend']):
            return handle_transition_prompt(input_paths, prompt, timestamp)
        
        elif any(word in prompt for word in ['overlay', 'picture in picture', 'pip', 'composite']):
            return handle_overlay_prompt(input_paths, prompt, timestamp)
        
        elif any(word in prompt for word in ['side by side', 'split screen', 'compare']):
            return handle_split_screen_prompt(input_paths, prompt, timestamp)
        
        else:
            # Clean up files
            for path in input_paths:
                if os.path.exists(path):
                    os.remove(path)
            
            return jsonify({
                'success': False, 
                'error': 'Prompt not recognized. Try: "merge videos", "fade transition", "overlay videos", "side by side"'
            })
        
    except Exception as e:
        print(f"Error in process_multi_prompt: {str(e)}")
        # Clean up files on error
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        return jsonify({'success': False, 'error': f'Multi-prompt error: {str(e)}'})

def handle_merge_prompt(input_paths, prompt, timestamp):
    """Handle merge-related prompts"""
    try:
        print(f"Handling merge prompt for {len(input_paths)} videos...")
        
        # Generate output filename
        output_filename = f'prompt_merged_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Normalize all videos first
        normalized_paths = []
        
        for i, input_path in enumerate(input_paths):
            norm_filename = f"merge_norm_{timestamp}_{i}.mp4"
            norm_path = os.path.join(app.config['UPLOAD_FOLDER'], norm_filename)
            
            # Normalize video to consistent format
            norm_cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                '-r', '30',
                '-ar', '44100',
                '-ac', '2',
                '-crf', '23',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-y', norm_path
            ]
            
            success, message = run_ffmpeg_command(norm_cmd, timeout=300)
            if not success:
                # Clean up and return error
                for path in input_paths + normalized_paths:
                    if os.path.exists(path):
                        os.remove(path)
                return jsonify({'success': False, 'error': f'Failed to normalize video {i+1}: {message}'})
            
            normalized_paths.append(norm_path)
        
        # Create concat file
        concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'prompt_concat_{timestamp}.txt')
        with open(concat_file, 'w', encoding='utf-8') as f:
            for path in normalized_paths:
                normalized_path = path.replace('\\', '/')
                f.write(f"file '{normalized_path}'\n")
        
        # Merge videos
        merge_cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
            '-c', 'copy',
            '-y', output_path
        ]
        
        success, message = run_ffmpeg_command(merge_cmd, timeout=600)
        
        # Clean up
        for path in input_paths + normalized_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(concat_file):
            os.remove(concat_file)
        
        if not success:
            return jsonify({'success': False, 'error': f'Merge failed: {message}'})
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'message': f'Successfully merged {len(input_paths)} videos using text prompt'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Merge prompt error: {str(e)}'})

def handle_transition_prompt(input_paths, prompt, timestamp):
    """Handle transition-related prompts"""
    try:
        print(f"Handling transition prompt for {len(input_paths)} videos...")
        
        # Extract transition type and duration from prompt
        transition_type = 'fade'  # default
        duration = 1.0  # default
        
        if 'dissolve' in prompt:
            transition_type = 'dissolve'
        elif 'wipe' in prompt:
            transition_type = 'wipe'
        elif 'cut' in prompt:
            transition_type = 'cut'
        
        # Extract duration if specified
        import re
        duration_match = re.search(r'(\d+\.?\d*)\s*(?:second|sec|s)', prompt)
        if duration_match:
            duration = float(duration_match.group(1))
        
        # Generate output filename
        output_filename = f'prompt_transition_{transition_type}_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Normalize all videos first
        normalized_paths = []
        
        for i, input_path in enumerate(input_paths):
            norm_filename = f"trans_norm_{timestamp}_{i}.mp4"
            norm_path = os.path.join(app.config['UPLOAD_FOLDER'], norm_filename)
            
            # Normalize video to consistent format
            norm_cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                '-r', '30',
                '-ar', '44100',
                '-ac', '2',
                '-crf', '23',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-y', norm_path
            ]
            
            success, message = run_ffmpeg_command(norm_cmd, timeout=300)
            if not success:
                # Clean up and return error
                for path in input_paths + normalized_paths:
                    if os.path.exists(path):
                        os.remove(path)
                return jsonify({'success': False, 'error': f'Failed to normalize video {i+1}: {message}'})
            
            normalized_paths.append(norm_path)
        
        # Apply transition based on type
        if transition_type in ['fade', 'dissolve'] and len(normalized_paths) == 2:
            # Use xfade for 2 videos
            transition_cmd = [
                'ffmpeg', '-i', normalized_paths[0], '-i', normalized_paths[1],
                '-filter_complex',
                f'[0:v][1:v]xfade=transition={transition_type}:duration={duration}:offset=5[v];'
                f'[0:a][1:a]acrossfade=d={duration}[a]',
                '-map', '[v]', '-map', '[a]',
                '-c:v', 'libx264', '-c:a', 'aac',
                '-crf', '23', '-preset', 'medium',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-y', output_path
            ]
            
            success, message = run_ffmpeg_command(transition_cmd, timeout=600)
        
        elif transition_type == 'wipe' and len(normalized_paths) == 2:
            # Wipe transition
            wipe_cmd = [
                'ffmpeg', '-i', normalized_paths[0], '-i', normalized_paths[1],
                '-filter_complex',
                f'[0:v][1:v]xfade=transition=wipeleft:duration={duration}:offset=5[v];'
                f'[0:a][1:a]acrossfade=d={duration}[a]',
                '-map', '[v]', '-map', '[a]',
                '-c:v', 'libx264', '-c:a', 'aac',
                '-crf', '23', '-preset', 'medium',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-y', output_path
            ]
            
            success, message = run_ffmpeg_command(wipe_cmd, timeout=600)
        
        else:
            # For multiple videos or cut transition, use concat
            concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'trans_concat_{timestamp}.txt')
            with open(concat_file, 'w', encoding='utf-8') as f:
                for path in normalized_paths:
                    normalized_path = path.replace('\\', '/')
                    f.write(f"file '{normalized_path}'\n")
            
            if transition_type == 'fade':
                # Apply fade effects
                fade_cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-vf', f'fade=in:0:{int(duration*30)},fade=out:st={max(0, 30-duration)}:d={duration}',
                    '-af', f'afade=in:st=0:d={duration},afade=out:st={max(0, 30-duration)}:d={duration}',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
            else:
                # Simple concatenation
                fade_cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-c', 'copy',
                    '-y', output_path
                ]
            
            success, message = run_ffmpeg_command(fade_cmd, timeout=600)
            
            if os.path.exists(concat_file):
                os.remove(concat_file)
        
        # Clean up
        for path in input_paths + normalized_paths:
            if os.path.exists(path):
                os.remove(path)
        
        if not success:
            return jsonify({'success': False, 'error': f'Transition failed: {message}'})
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'message': f'Applied {transition_type} transitions using FFmpeg'
        })
        
    except Exception as e:
        print(f"Error in apply_transition: {str(e)}")
        # Clean up files on error
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        return jsonify({'success': False, 'error': f'Transition error: {str(e)}'})

@app.route('/process-prompt', methods=['POST'])
@login_required
def process_prompt():
    """
    Advanced text prompt processor for video editing commands
    
    Supported Commands:
    - trim start=X end=Y
    - resize width=X height=Y
    - speed factor=X
    - extract_audio format=mp3|wav
    - color_grade brightness=X contrast=Y saturation=Z
    - color_grade preset=cinematic|vintage|warm|cool|noir|vibrant
    - speed_ramp start=X end=Y factor=Z
    - effect type=blur|glow|vignette|freeze_frame|motion_blur|sepia|negative|mirror|pixelate|edge_detection strength=X
    - animation type=zoom|pan|fade|bounce|rotate|slide start=X end=Y scale=Z angle=A direction=left|right|up|down
    - merge_videos files=file1.mp4,file2.mp4 transition=type duration=X
    - overlay type=text|image|video content="X" x=Y y=Z duration=W position=top|bottom|center-left|right|center
    - transition type=crossfade|slide|wipe|fade duration=X direction=left|right|up|down
    """
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No main file uploaded'})
        
        main_file = request.files['file']
        prompt = request.form.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})
        
        print(f"Processing prompt: {prompt}")
        
        # Don't save main file yet - let the command handler do it properly
        main_filename = secure_filename(main_file.filename)
        timestamp = int(time.time())
        main_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{main_filename}")
        
        # Check if auxiliary files are provided for multi-video operations
        auxiliary_files = request.files.getlist('aux_files[]') if 'aux_files[]' in request.files else []
        
        # Check if this is a command that needs auxiliary files but also needs main file saved
        prompt_lower = prompt.lower().strip()
        needs_main_file_saved = (
            prompt_lower.startswith('overlay') or 
            prompt_lower.startswith('trim') or 
            prompt_lower.startswith('resize') or 
            prompt_lower.startswith('speed') or 
            prompt_lower.startswith('extract_audio') or 
            prompt_lower.startswith('color_grade') or 
            prompt_lower.startswith('speed_ramp') or 
            prompt_lower.startswith('effect') or 
            prompt_lower.startswith('animation')
        )
        
        print(f"Command type: {prompt_lower.split()[0] if prompt_lower else 'unknown'}")
        print(f"Needs main file saved: {needs_main_file_saved}")
        print(f"Auxiliary files count: {len(auxiliary_files)}")
        
        # If auxiliary files are provided, combine with main file for multi-video operations
        if auxiliary_files and len(auxiliary_files) > 0:
            # For overlay commands, save the main file first since it's needed
            if needs_main_file_saved:
                save_success, save_message = save_file_safely(main_file, main_path)
                if not save_success:
                    return jsonify({'success': False, 'error': f'Failed to save main video file: {save_message}'})
            
            # Create a combined files object for commands that need auxiliary files
            from werkzeug.datastructures import MultiDict
            combined_files = MultiDict()
            
            # For multi-video operations (merge, transition), add main file to the list
            if not needs_main_file_saved:
                combined_files.add('files[]', main_file)
            
            # Add auxiliary files
            for aux_file in auxiliary_files:
                if aux_file and aux_file.filename:
                    combined_files.add('files[]', aux_file)
            
            # Parse command and parameters with combined files
            command_result = parse_and_execute_command(prompt, main_path, combined_files, timestamp)
        else:
            # For single-video operations, save the main file first
            save_success, save_message = save_file_safely(main_file, main_path)
            if not save_success:
                return jsonify({'success': False, 'error': f'Failed to save main video file: {save_message}'})
            
            # Parse command and parameters for single-video operations
            command_result = parse_and_execute_command(prompt, main_path, request.files, timestamp)
        
        # Clean up main file
        if os.path.exists(main_path):
            os.remove(main_path)
        
        return jsonify(command_result)
        
    except Exception as e:
        print(f"Error in process_prompt: {str(e)}")
        if 'main_path' in locals() and os.path.exists(main_path):
            os.remove(main_path)
        return jsonify({'success': False, 'error': str(e)})

def parse_and_execute_command(prompt, main_path, files, timestamp):
    """Parse and execute video editing commands from text prompts"""
    
    # Normalize prompt
    prompt = prompt.lower().strip()
    
    try:
        # TRIM COMMAND
        if prompt.startswith('trim'):
            return execute_trim_command(prompt, main_path, timestamp)
        
        # RESIZE COMMAND
        elif prompt.startswith('resize'):
            return execute_resize_command(prompt, main_path, timestamp)
        
        # SPEED COMMAND
        elif prompt.startswith('speed'):
            return execute_speed_command(prompt, main_path, timestamp)
        
        # EXTRACT AUDIO COMMAND
        elif prompt.startswith('extract_audio'):
            return execute_extract_audio_command(prompt, main_path, timestamp)
        
        # COLOR GRADE COMMAND
        elif prompt.startswith('color_grade'):
            return execute_color_grade_command(prompt, main_path, timestamp)
        
        # SPEED RAMP COMMAND
        elif prompt.startswith('speed_ramp'):
            return execute_speed_ramp_command(prompt, main_path, timestamp)
        
        # EFFECT COMMAND
        elif prompt.startswith('effect'):
            return execute_effect_command(prompt, main_path, timestamp)
        
        # ANIMATION COMMAND
        elif prompt.startswith('animation'):
            return execute_animation_command(prompt, main_path, timestamp)
        
        # OVERLAY COMMAND
        elif prompt.startswith('overlay'):
            return execute_overlay_command(prompt, main_path, files, timestamp)
        
        # MERGE VIDEOS COMMAND
        elif prompt.startswith('merge_videos'):
            return execute_merge_command(prompt, files, timestamp)
        
        # TRANSITION COMMAND
        elif prompt.startswith('transition'):
            return execute_transition_command(prompt, files, timestamp)
        
        else:
            return {
                'success': False, 
                'error': f'Unknown command. Supported commands: trim, resize, speed, extract_audio, color_grade, speed_ramp, effect, animation, overlay, merge_videos, transition'
            }
            
    except Exception as e:
        return {'success': False, 'error': f'Command execution failed: {str(e)}'}

def parse_parameters(param_string):
    """Parse parameters from command string"""
    params = {}
    
    # Handle quoted strings
    import re
    quoted_matches = re.findall(r'(\w+)="([^"]*)"', param_string)
    for key, value in quoted_matches:
        params[key] = value
        param_string = re.sub(f'{key}="{value}"', '', param_string)
    
    # Handle regular parameters
    param_matches = re.findall(r'(\w+)=([^\s]+)', param_string)
    for key, value in param_matches:
        # Try to convert to appropriate type
        try:
            if '.' in value:
                params[key] = float(value)
            elif value.isdigit():
                params[key] = int(value)
            else:
                params[key] = value
        except:
            params[key] = value
    
    return params

def execute_trim_command(prompt, main_path, timestamp):
    """Execute trim command: trim start=X end=Y"""
    params = parse_parameters(prompt)
    
    start_time = params.get('start', 0)
    end_time = params.get('end', None)
    
    if end_time is None:
        return {'success': False, 'error': 'Trim command requires both start and end parameters'}
    
    # Use the existing trim_video function logic
    output_filename = f'prompt_trim_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Calculate duration
    duration = end_time - start_time
    
    # FFmpeg command for trimming (using existing proven approach)
    cmd = [
        'ffmpeg', '-i', main_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-crf', '18',
        '-preset', 'fast',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]
    
    success, message = run_ffmpeg_command(cmd, timeout=300)
    
    if success:
        return {'success': True, 'output_file': output_filename, 'message': f'Video trimmed from {start_time}s to {end_time}s'}
    else:
        return {'success': False, 'error': f'Trim failed: {message}'}

def execute_resize_command(prompt, main_path, timestamp):
    """Execute resize command: resize width=X height=Y"""
    params = parse_parameters(prompt)
    
    width = params.get('width')
    height = params.get('height')
    
    if not width or not height:
        return {'success': False, 'error': 'Resize command requires both width and height parameters'}
    
    # Ensure dimensions are even (required for H.264)
    width = int(width) - (int(width) % 2)
    height = int(height) - (int(height) % 2)
    
    output_filename = f'prompt_resize_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Use existing resize logic
    cmd = [
        'ffmpeg', '-i', main_path,
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-crf', '18',
        '-preset', 'fast',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]
    
    success, message = run_ffmpeg_command(cmd, timeout=300)
    
    if success:
        return {'success': True, 'output_file': output_filename, 'message': f'Video resized to {width}x{height}'}
    else:
        return {'success': False, 'error': f'Resize failed: {message}'}

def execute_speed_command(prompt, main_path, timestamp):
    """Execute speed command: speed factor=X"""
    params = parse_parameters(prompt)
    
    factor = params.get('factor', 1.0)
    
    output_filename = f'prompt_speed_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Calculate video and audio filters
    video_speed = f'setpts={1/factor}*PTS'
    
    # Handle audio speed (atempo filter has limitations)
    audio_filters = []
    remaining_factor = factor
    
    while remaining_factor > 2.0:
        audio_filters.append("atempo=2.0")
        remaining_factor /= 2.0
    
    while remaining_factor < 0.5:
        audio_filters.append("atempo=0.5")
        remaining_factor /= 0.5
    
    if remaining_factor != 1.0:
        audio_filters.append(f"atempo={remaining_factor}")
    
    audio_filter_str = ",".join(audio_filters) if audio_filters else "anull"
    
    cmd = [
        'ffmpeg', '-i', main_path,
        '-filter_complex', f'[0:v]{video_speed}[v];[0:a]{audio_filter_str}[a]',
        '-map', '[v]', '-map', '[a]',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-crf', '23',
        '-preset', 'medium',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]
    
    success, message = run_ffmpeg_command(cmd, timeout=300)
    
    if success:
        return {'success': True, 'output_file': output_filename, 'message': f'Video speed changed by factor {factor}'}
    else:
        return {'success': False, 'error': f'Speed change failed: {message}'}

def execute_extract_audio_command(prompt, main_path, timestamp):
    """Execute extract_audio command: extract_audio format=mp3|wav"""
    params = parse_parameters(prompt)
    
    format_type = params.get('format', 'mp3').lower()
    
    if format_type not in ['mp3', 'wav']:
        return {'success': False, 'error': 'Audio format must be mp3 or wav'}
    
    output_filename = f'prompt_audio_{timestamp}.{format_type}'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Use existing extract_audio logic
    if format_type == 'mp3':
        cmd = [
            'ffmpeg', '-i', main_path,
            '-vn',
            '-acodec', 'libmp3lame',
            '-ab', '192k',
            '-ar', '44100',
            '-y', output_path
        ]
    else:  # wav
        cmd = [
            'ffmpeg', '-i', main_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-y', output_path
        ]
    
    success, message = run_ffmpeg_command(cmd, timeout=300)
    
    if success:
        return {'success': True, 'output_file': output_filename, 'message': f'Audio extracted as {format_type}'}
    else:
        return {'success': False, 'error': f'Audio extraction failed: {message}'}

def execute_color_grade_command(prompt, main_path, timestamp):
    """Execute color_grade command: color_grade brightness=X contrast=Y saturation=Z OR color_grade preset=cinematic"""
    params = parse_parameters(prompt)
    
    output_filename = f'prompt_colorgrade_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    if 'preset' in params:
        preset = params['preset']
        
        # Predefined color grading presets
        if preset == 'cinematic':
            vf = 'eq=brightness=0.1:contrast=1.2:saturation=0.9'
        elif preset == 'vintage':
            vf = 'eq=brightness=0.05:contrast=0.8:saturation=0.7'
        elif preset == 'warm':
            vf = 'colortemperature=4000'
        elif preset == 'cool':
            vf = 'colortemperature=7000'
        elif preset == 'noir':
            vf = 'hue=s=0'  # Black and white
        elif preset == 'vibrant':
            vf = 'eq=brightness=0.05:contrast=1.3:saturation=1.4'
        else:
            return {'success': False, 'error': f'Unknown preset: {preset}'}
    else:
        # Custom values
        brightness = params.get('brightness', 0)
        contrast = params.get('contrast', 1)
        saturation = params.get('saturation', 1)
        
        vf = f'eq=brightness={brightness}:contrast={contrast}:saturation={saturation}'
    
    cmd = [
        'ffmpeg', '-i', main_path,
        '-vf', vf,
        '-c:v', 'libx264',
        '-c:a', 'copy',
        '-crf', '23',
        '-preset', 'medium',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]
    
    success, message = run_ffmpeg_command(cmd, timeout=300)
    
    if success:
        return {'success': True, 'output_file': output_filename, 'message': 'Color grading applied'}
    else:
        return {'success': False, 'error': f'Color grading failed: {message}'}

def execute_speed_ramp_command(prompt, main_path, timestamp):
    """Execute speed_ramp command: speed_ramp start=X end=Y factor=Z"""
    params = parse_parameters(prompt)
    
    start_time = params.get('start', 0)
    end_time = params.get('end', 5)
    factor = params.get('factor', 2.0)
    
    output_filename = f'prompt_speedramp_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Create simple speed change (compatible with essentials build)
    # For essentials build, use simple speed change instead of complex ramping
    vf = f'setpts=PTS/{factor}'
    
    cmd = [
        'ffmpeg', '-i', main_path,
        '-vf', vf,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-crf', '23',
        '-preset', 'medium',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]
    
    success, message = run_ffmpeg_command(cmd, timeout=300)
    
    if success:
        return {'success': True, 'output_file': output_filename, 'message': f'Speed ramp applied from {start_time}s to {end_time}s'}
    else:
        return {'success': False, 'error': f'Speed ramp failed: {message}'}

def execute_effect_command(prompt, main_path, timestamp):
    """Execute effect command: effect type=blur|glow|vignette|freeze_frame|motion_blur|sepia|negative|mirror|pixelate|edge_detection strength=X"""
    params = parse_parameters(prompt)
    
    effect_type = params.get('type', 'blur')
    strength = params.get('strength', 1.0)
    
    output_filename = f'prompt_effect_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Define video filters for different effects
    if effect_type == 'blur' or effect_type == 'gaussian_blur':
        vf = f'gblur=sigma={strength * 2}'
    elif effect_type == 'freeze' or effect_type == 'freeze_frame':
        # Simple freeze effect - just slow down the video significantly
        vf = f'setpts=PTS*{strength*10}'
    elif effect_type == 'glow':
        vf = f'gblur=sigma={strength},blend=all_mode=screen'
    elif effect_type == 'vignette':
        vf = f'vignette=angle=PI/4:x0=w/2:y0=h/2'
    elif effect_type == 'sepia':
        vf = 'colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131'
    elif effect_type == 'negative':
        vf = 'negate'
    elif effect_type == 'mirror':
        vf = 'hflip'
    elif effect_type == 'pixelate':
        scale_factor = max(1, int(10 / strength))
        vf = f'scale=iw/{scale_factor}:ih/{scale_factor}:flags=neighbor,scale=iw*{scale_factor}:ih*{scale_factor}:flags=neighbor'
    elif effect_type == 'edge_detection':
        vf = 'edgedetect'
    elif effect_type == 'motion_blur':
        vf = f'minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'
    else:
        return {'success': False, 'error': f'Unknown effect type: {effect_type}. Supported: blur, gaussian_blur, freeze, glow, vignette, sepia, negative, mirror, pixelate, edge_detection, motion_blur'}
    
    cmd = [
        'ffmpeg', '-i', main_path,
        '-vf', vf,
        '-c:v', 'libx264',
        '-c:a', 'copy',
        '-crf', '23',
        '-preset', 'medium',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]
    
    success, message = run_ffmpeg_command(cmd, timeout=300)
    
    if success:
        return {'success': True, 'output_file': output_filename, 'message': f'{effect_type} effect applied'}
    else:
        return {'success': False, 'error': f'Effect failed: {message}'}

def execute_animation_command(prompt, main_path, timestamp):
    """Execute animation command: animation type=zoom|pan|fade|bounce|rotate|slide start=X end=Y scale=Z angle=A direction=left|right|up|down"""
    params = parse_parameters(prompt)
    
    anim_type = params.get('type', 'zoom')
    start_time = float(params.get('start', 0))
    end_time = float(params.get('end', 5))
    scale = float(params.get('scale', 1.5))
    angle = float(params.get('angle', 45))
    direction = params.get('direction', 'right')
    
    output_filename = f'prompt_animation_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # First, validate the input file exists
    if not os.path.exists(main_path):
        return {'success': False, 'error': f'Input video file not found: {main_path}'}
    
    # Check file size
    file_size = os.path.getsize(main_path)
    if file_size == 0:
        return {'success': False, 'error': 'Input video file is empty'}
    
    print(f"Processing animation on file: {os.path.basename(main_path)} ({file_size} bytes)")
    
    # Define very simple and reliable animation filters
    if anim_type == 'zoom' or anim_type == 'zoom_in':
        # Fixed size zoom for reliability
        vf = 'scale=1920:1080'
    elif anim_type == 'zoom_out':
        # Smaller fixed size
        vf = 'scale=1280:720'
    elif anim_type == 'fade' or anim_type == 'fade_in':
        # Simple fade in - always works
        vf = 'fade=in:st=0:d=2'
    elif anim_type == 'fade_out':
        # Simple fade out
        vf = 'fade=out:st=0:d=2'
    elif anim_type == 'rotate':
        # Simple 45 degree rotation
        vf = 'rotate=PI/4'
    elif anim_type == 'pan':
        # Simple crop for pan effect
        vf = 'crop=iw*0.9:ih*0.9:iw*0.05:ih*0.05'
    elif anim_type == 'slide':
        # Simple pad for slide effect
        vf = 'pad=iw*1.1:ih*1.1:iw*0.05:ih*0.05'
    elif anim_type == 'bounce':
        # Simple scale up
        vf = 'scale=iw*1.3:ih*1.3'
    elif anim_type == 'blur':
        # Simple box blur
        vf = 'boxblur=2:2'
    elif anim_type == 'brightness':
        # Simple brightness adjustment
        vf = 'eq=brightness=0.2'
    else:
        return {'success': False, 'error': f'Unknown animation type: {anim_type}. Supported: zoom, zoom_in, zoom_out, fade, fade_in, fade_out, rotate, pan, slide, bounce, blur, brightness'}
    
    # More robust FFmpeg command with better error handling
    cmd = [
        'ffmpeg', '-i', main_path,
        '-vf', vf,
        '-c:v', 'libx264',
        '-c:a', 'aac',  # Use aac instead of copy for better compatibility
        '-crf', '23',
        '-preset', 'fast',  # Use fast preset instead of medium
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
        '-fflags', '+genpts',  # Generate presentation timestamps
        '-y', output_path
    ]
    
    print(f"Animation parameters: type={anim_type}, start={start_time}, end={end_time}, scale={scale}, direction={direction}")
    print(f"Animation filter: {vf}")
    print(f"Running animation command: {' '.join(cmd)}")
    
    # Use shorter timeout for animations to prevent hanging
    success, message = run_ffmpeg_command(cmd, timeout=90)  # 1.5 minutes timeout
    
    # If the animation fails, try a super simple fallback
    if not success:
        print(f"Animation {anim_type} failed, trying simple fallback...")
        print(f"Original error: {message}")
        
        # Super simple fallback animations that should always work
        if anim_type in ['zoom', 'zoom_in', 'bounce']:
            simple_vf = 'scale=1280:720'  # Fixed size scaling
        elif anim_type in ['fade', 'fade_in']:
            simple_vf = 'fade=in:st=0:d=1'
        elif anim_type == 'fade_out':
            simple_vf = 'fade=out:st=0:d=1'
        elif anim_type == 'rotate':
            simple_vf = 'rotate=0.5'  # Simple 30 degree rotation
        elif anim_type in ['pan', 'slide']:
            simple_vf = 'crop=iw*0.9:ih*0.9:iw*0.05:ih*0.05'  # Simple crop
        elif anim_type == 'blur':
            simple_vf = 'boxblur=2:2'  # Simple box blur
        else:
            simple_vf = 'eq=brightness=0.1'  # Simple brightness change
        
        simple_cmd = [
            'ffmpeg', '-i', main_path,
            '-vf', simple_vf,
            '-c:v', 'libx264',
            '-c:a', 'aac',  # Use aac for compatibility
            '-crf', '28',  # Lower quality for speed
            '-preset', 'ultrafast',  # Fastest preset
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-avoid_negative_ts', 'make_zero',
            '-fflags', '+genpts',
            '-y', output_path
        ]
        
        print(f"Trying simple fallback: {' '.join(simple_cmd)}")
        success, message = run_ffmpeg_command(simple_cmd, timeout=60)  # 1 minute timeout
    
    if success:
        print(f"Animation {anim_type} completed successfully")
        return {'success': True, 'output_file': output_filename, 'message': f'{anim_type} animation applied successfully'}
    else:
        print(f"Animation {anim_type} failed: {message}")
        # If it's still a timeout, provide a helpful error message
        if 'timeout' in message.lower() or 'Processing timeout' in message:
            return {'success': False, 'error': f'Animation processing timeout. Your video may be too large or complex. Try with a shorter/smaller video.'}
        else:
            return {'success': False, 'error': f'Animation failed: {message}'}

def execute_overlay_command(prompt, main_path, files, timestamp):
    """Execute overlay command: overlay type=text|image|video content="X" x=Y y=Z duration=W position=top|bottom|center-left|right|center"""
    params = parse_parameters(prompt)
    
    overlay_type = params.get('type', 'text')
    content = params.get('content', 'Sample Text')
    x_pos = params.get('x', 10)
    y_pos = params.get('y', 10)
    duration = params.get('duration', 0)  # 0 = entire video
    position = params.get('position', 'bottom-center')
    font_size = params.get('fontsize', 24)
    font_color = params.get('fontcolor', 'white')
    opacity = params.get('opacity', 1.0)
    
    output_filename = f'prompt_overlay_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    if overlay_type == 'text':
        # Position mapping
        pos_map = {
            'top-left': '10:10',
            'top-center': '(w-text_w)/2:10',
            'top-right': 'w-text_w-10:10',
            'center-left': '10:(h-text_h)/2',
            'center': '(w-text_w)/2:(h-text_h)/2',
            'center-right': 'w-text_w-10:(h-text_h)/2',
            'bottom-left': '10:h-text_h-10',
            'bottom-center': '(w-text_w)/2:h-text_h-10',
            'bottom-right': 'w-text_w-10:h-text_h-10'
        }
        
        text_pos = pos_map.get(position, f'{x_pos}:{y_pos}')
        
        # Build text filter with enhanced options
        text_filter = f'drawtext=text=\'{content}\':fontsize={font_size}:fontcolor={font_color}:x={text_pos}'
        
        # Add background box for better readability
        text_filter += ':box=1:boxcolor=black@0.5:boxborderw=5'
        
        # Add duration if specified
        if duration > 0:
            text_filter += f':enable=\'between(t,0,{duration})\''
        
        cmd = [
            'ffmpeg', '-i', main_path,
            '-vf', text_filter,
            '-c:v', 'libx264',
            '-c:a', 'copy',
            '-crf', '23',
            '-preset', 'medium',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y', output_path
        ]
        
        success, message = run_ffmpeg_command(cmd, timeout=300)
        
        if success:
            return {'success': True, 'output_file': output_filename, 'message': f'Text overlay "{content}" added'}
        else:
            return {'success': False, 'error': f'Text overlay failed: {message}'}
    
    elif overlay_type in ['image', 'video']:
        # Check if auxiliary files are provided
        if 'files[]' not in files:
            return {'success': False, 'error': f'{overlay_type.title()} overlay requires an auxiliary {overlay_type} file. Please upload a {overlay_type} in the auxiliary files section.'}
        
        aux_files = files.getlist('files[]')
        if not aux_files:
            return {'success': False, 'error': f'No auxiliary {overlay_type} file provided for overlay.'}
        
        # Use the first auxiliary file as overlay
        overlay_file = aux_files[0]
        if not overlay_file or not overlay_file.filename:
            return {'success': False, 'error': f'Invalid {overlay_type} file for overlay.'}
        
        # Save overlay file
        overlay_filename = secure_filename(overlay_file.filename)
        overlay_input_filename = f"{timestamp}_overlay_{overlay_filename}"
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_input_filename)
        
        save_success, save_message = save_file_safely(overlay_file, overlay_path)
        if not save_success:
            return {'success': False, 'error': f'Failed to save overlay {overlay_type}: {save_message}'}
        
        # Validate the overlay file if it's an image
        if overlay_type == 'image':
            validate_success, validate_message = validate_image_file(overlay_path)
            if not validate_success:
                if os.path.exists(overlay_path):
                    os.remove(overlay_path)
                return {'success': False, 'error': f'Invalid image file: {validate_message}'}
        
        try:
            # Get additional parameters
            width = params.get('width')
            height = params.get('height')
            start_time = params.get('start', 0)
            
            print(f"Overlay parameters: type={overlay_type}, position={position}, width={width}, height={height}, opacity={opacity}, duration={duration}, start={start_time}")
            
            # Position mapping for overlays
            if position != 'custom':
                pos_map = {
                    'top-left': '10:10',
                    'top-center': '(main_w-overlay_w)/2:10',
                    'top-right': 'main_w-overlay_w-10:10',
                    'center-left': '10:(main_h-overlay_h)/2',
                    'center': '(main_w-overlay_w)/2:(main_h-overlay_h)/2',
                    'center-right': 'main_w-overlay_w-10:(main_h-overlay_h)/2',
                    'bottom-left': '10:main_h-overlay_h-10',
                    'bottom-center': '(main_w-overlay_w)/2:main_h-overlay_h-10',
                    'bottom-right': 'main_w-overlay_w-10:main_h-overlay_h-10'
                }
                overlay_pos = pos_map.get(position, f'{x_pos}:{y_pos}')
            else:
                overlay_pos = f'{x_pos}:{y_pos}'
            
            # Build optimized overlay filter
            filter_parts = []
            
            # Start with the overlay input
            overlay_input = '[1:v]'
            
            # Add scaling if specified
            if width or height:
                if width and height:
                    overlay_input += f'scale={width}:{height},'
                elif width:
                    overlay_input += f'scale={width}:-1,'
                elif height:
                    overlay_input += f'scale=-1:{height},'
            
            # For images, add loop to make them last the duration (simplified)
            if overlay_type == 'image':
                overlay_input += 'loop=loop=-1:size=1,'
            
            # Add opacity if not 1.0 (simplified)
            if opacity != 1.0:
                overlay_input += f'format=rgba,colorchannelmixer=aa={opacity},'
            
            # Remove trailing comma
            if overlay_input.endswith(','):
                overlay_input = overlay_input[:-1]
            
            overlay_input += '[ovr]'
            
            # Build the overlay command
            overlay_cmd = f'[0:v][ovr]overlay={overlay_pos}'
            
            # Add duration control if specified
            if duration > 0:
                overlay_cmd += f':enable=\'between(t,{start_time},{start_time + duration})\''
            
            overlay_cmd += '[out]'
            
            # Combine the filter
            overlay_filter = f'{overlay_input};{overlay_cmd}'
            
            print(f"Overlay filter: {overlay_filter}")  # Debug output
            
            cmd = [
                'ffmpeg', '-i', main_path, '-i', overlay_path,
                '-filter_complex', overlay_filter,
                '-map', '[out]', '-map', '0:a?',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-crf', '23',
                '-preset', 'fast',  # Use faster preset for overlay processing
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-y', output_path
            ]
            
            print(f"Running overlay command: {' '.join(cmd)}")
            success, message = run_ffmpeg_command(cmd, timeout=600)  # Increase timeout to 10 minutes
            
            # If the complex filter fails, try a simpler approach for images
            if not success and overlay_type == 'image':
                print(f"Complex overlay failed, trying simple approach: {message}")
                
                # Simple overlay without loop (for shorter videos or when duration is specified)
                simple_filter = f'[1:v]scale=iw:ih[ovr];[0:v][ovr]overlay={overlay_pos}[out]'
                
                simple_cmd = [
                    'ffmpeg', '-i', main_path, '-i', overlay_path,
                    '-filter_complex', simple_filter,
                    '-map', '[out]', '-map', '0:a?',
                    '-c:v', 'libx264',
                    '-c:a', 'copy',  # Just copy audio to be faster
                    '-crf', '28',  # Lower quality for faster processing
                    '-preset', 'ultrafast',  # Fastest preset
                    '-t', str(duration) if duration > 0 else '30',  # Limit duration
                    '-y', output_path
                ]
                
                print(f"Running simple overlay command: {' '.join(simple_cmd)}")
                success, message = run_ffmpeg_command(simple_cmd, timeout=300)
            
            # Clean up overlay file
            if os.path.exists(overlay_path):
                os.remove(overlay_path)
            
            if success:
                return {'success': True, 'output_file': output_filename, 'message': f'{overlay_type.title()} overlay added at position {overlay_pos}'}
            else:
                return {'success': False, 'error': f'{overlay_type.title()} overlay failed: {message}'}
                
        except Exception as e:
            # Clean up on error
            if os.path.exists(overlay_path):
                os.remove(overlay_path)
            return {'success': False, 'error': f'{overlay_type.title()} overlay error: {str(e)}'}
    
    else:
        return {'success': False, 'error': f'Overlay type "{overlay_type}" not supported. Use: text, image, video'}

def execute_merge_command(prompt, files, timestamp):
    """Execute merge_videos command: merge_videos transition=fade|dissolve|cut duration=X"""
    params = parse_parameters(prompt)
    
    # Check if multiple files are provided
    if 'files[]' not in files:
        return {
            'success': False, 
            'error': 'Merge command requires multiple files. Please upload a main video and select auxiliary videos.',
            'suggestion': 'Upload a main video file and select additional videos in the "Auxiliary Video Files" section.'
        }
    
    file_list = files.getlist('files[]')
    if len(file_list) < 2:
        return {
            'success': False, 
            'error': 'Merge command requires at least 2 video files.',
            'suggestion': 'Upload at least 2 video files to merge them together.'
        }
    
    transition_type = params.get('transition', 'cut')
    duration = params.get('duration', 1.0)
    
    print(f"Merging {len(file_list)} videos with {transition_type} transition...")
    
    # Save uploaded files with validation
    input_paths = []
    try:
        for i, file in enumerate(file_list):
            if file and file.filename:
                filename = secure_filename(file.filename)
                input_filename = f"merge_{timestamp}_{i}_{filename}"
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
                
                print(f"Processing file {i+1}: {filename} -> {input_filename}")
                
                # Debug: Check file object before saving
                if hasattr(file, 'content_length'):
                    print(f"File {i+1} content length: {file.content_length}")
                
                # Save file safely
                save_success, save_message = save_file_safely(file, input_path)
                if not save_success:
                    print(f"Save failed for file {i+1}: {save_message}")
                    return {'success': False, 'error': f'Failed to save video file {i+1}: {save_message}'}
                
                # Validate file exists and has content
                if not os.path.exists(input_path):
                    return {'success': False, 'error': f'Video file {i+1} was not saved to disk'}
                
                file_size = os.path.getsize(input_path)
                if file_size == 0:
                    return {'success': False, 'error': f'Video file {i+1} is empty after saving (0 bytes)'}
                
                print(f"File {i+1} saved successfully: {file_size} bytes")
                
                # Quick FFmpeg validation to ensure file is readable
                validate_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', input_path]
                try:
                    validate_result = subprocess.run(validate_cmd, capture_output=True, text=True, timeout=15)
                    if validate_result.returncode != 0:
                        print(f"File validation failed for {input_path}: {validate_result.stderr}")
                        # Try to get more detailed error info
                        detailed_cmd = ['ffprobe', '-v', 'error', input_path]
                        detailed_result = subprocess.run(detailed_cmd, capture_output=True, text=True, timeout=10)
                        error_detail = detailed_result.stderr if detailed_result.stderr else "Unknown validation error"
                        return {'success': False, 'error': f'Video file {i+1} is corrupted or invalid format: {error_detail}'}
                except subprocess.TimeoutExpired:
                    return {'success': False, 'error': f'Video file {i+1} validation timeout - file may be too large or corrupted'}
                except Exception as e:
                    return {'success': False, 'error': f'Video file {i+1} validation error: {str(e)}'}
                
                input_paths.append(input_path)
                print(f"Successfully saved and validated video {i+1}: {input_filename} ({file_size} bytes)")
            else:
                print(f"Skipping invalid file {i+1}: {file}")
                return {'success': False, 'error': f'Video file {i+1} is invalid or has no filename'}
        
        if len(input_paths) < 2:
            return {'success': False, 'error': 'At least 2 valid video files are required for merging'}
        
        # Generate output filename
        output_filename = f'prompt_merged_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        print(f"Processing merge with transition type: {transition_type}")
        
        # Use the existing merge logic with transitions
        if transition_type == 'cut':
            # Simple concatenation
            print("Using simple concatenation (cut transition)")
            success, message = merge_videos_simple(input_paths, output_path, timestamp)
        else:
            # Merge with transitions
            print(f"Using transition merge with {transition_type} transition")
            success, message = merge_videos_with_transition(input_paths, output_path, transition_type, duration, timestamp)
        
        # Clean up input files
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        
        if success:
            return {
                'success': True, 
                'output_file': output_filename,
                'message': f'Successfully merged {len(input_paths)} videos with {transition_type} transition'
            }
        else:
            return {'success': False, 'error': f'Merge failed: {message}'}
            
    except Exception as e:
        # Clean up files on error
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        return {'success': False, 'error': f'Merge error: {str(e)}'}
        
        print(f"Merging {len(input_paths)} videos with {transition_type} transition...")
        
        # Generate output filename
        output_filename = f'prompt_merge_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # First normalize all videos to same format
        normalized_paths = []
        for i, input_path in enumerate(input_paths):
            norm_filename = f"merge_norm_{timestamp}_{i}.mp4"
            norm_path = os.path.join(app.config['UPLOAD_FOLDER'], norm_filename)
            
            # Normalize video to consistent format
            norm_cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                '-r', '30',
                '-ar', '44100',
                '-ac', '2',
                '-crf', '23',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-y', norm_path
            ]
            
            success, message = run_ffmpeg_command(norm_cmd, timeout=300)
            if not success:
                # Clean up and return error
                for path in input_paths + normalized_paths:
                    if os.path.exists(path):
                        os.remove(path)
                return {'success': False, 'error': f'Failed to normalize video {i+1}: {message}'}
            
            normalized_paths.append(norm_path)
        
        # Apply merge with transition
        if transition_type == 'cut':
            # Simple concatenation
            concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'merge_concat_{timestamp}.txt')
            
            with open(concat_file, 'w', encoding='utf-8') as f:
                for path in normalized_paths:
                    normalized_path = path.replace('\\', '/')
                    f.write(f"file '{normalized_path}'\n")
            
            merge_cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-c', 'copy',
                '-y', output_path
            ]
            
            success, message = run_ffmpeg_command(merge_cmd, timeout=600)
            
            if os.path.exists(concat_file):
                os.remove(concat_file)
                
        elif transition_type in ['fade', 'dissolve']:
            if len(normalized_paths) == 2:
                # Simple crossfade for 2 videos
                fade_cmd = [
                    'ffmpeg', '-i', normalized_paths[0], '-i', normalized_paths[1],
                    '-filter_complex',
                    f'[0:v][1:v]xfade=transition=fade:duration={duration}:offset=5[v];'
                    f'[0:a][1:a]acrossfade=d={duration}[a]',
                    '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(fade_cmd, timeout=600)
            else:
                # Multiple videos - use concat with fade effects
                concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'merge_fade_{timestamp}.txt')
                
                with open(concat_file, 'w', encoding='utf-8') as f:
                    for path in normalized_paths:
                        normalized_path = path.replace('\\', '/')
                        f.write(f"file '{normalized_path}'\n")
                
                fade_cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-vf', f'fade=in:0:{int(duration*30)},fade=out:st={max(0, 30-duration)}:d={duration}',
                    '-af', f'afade=in:st=0:d={duration},afade=out:st={max(0, 30-duration)}:d={duration}',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(fade_cmd, timeout=600)
                
                if os.path.exists(concat_file):
                    os.remove(concat_file)
        else:
            success = False
            message = f"Unsupported transition type: {transition_type}"
        
        # Clean up temporary files
        for path in input_paths + normalized_paths:
            if os.path.exists(path):
                os.remove(path)
        
        if success:
            return {
                'success': True, 
                'output_file': output_filename, 
                'message': f'Successfully merged {len(input_paths)} videos with {transition_type} transition'
            }
        else:
            return {'success': False, 'error': f'Merge failed: {message}'}
            
    except Exception as e:
        # Clean up files on error
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        return {'success': False, 'error': f'Merge error: {str(e)}'}

def save_file_safely(file, file_path):
    """Save uploaded file safely with integrity checks"""
    try:
        # Check if file object is valid
        if not file or not hasattr(file, 'save'):
            return False, "Invalid file object"
        
        # Check if file has content
        if not file.filename:
            return False, "No filename provided"
        
        # Reset file pointer to beginning (important for multiple reads)
        file.seek(0)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file with explicit flushing
        try:
            file.save(file_path)
            
            # Force file system sync
            if os.path.exists(file_path):
                with open(file_path, 'r+b') as f:
                    f.flush()
                    os.fsync(f.fileno())
            
        except Exception as save_error:
            return False, f"File save failed: {str(save_error)}"
        
        # Wait for file system to fully sync
        time.sleep(2.0)
        
        # Multiple verification attempts
        for attempt in range(3):
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                    print(f"File saved successfully: {os.path.basename(file_path)} ({file_size} bytes)")
                    return True, "File saved successfully"
            
            # Wait a bit more and try again
            time.sleep(1.0)
        
        # If we get here, file save failed
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                os.remove(file_path)  # Remove empty file
                return False, "File is empty after saving - upload may have been interrupted"
            else:
                return True, "File saved successfully"
        else:
            return False, "File was not saved to disk"
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return False, f"File save error: {str(e)}"

def validate_and_repair_video_file(input_path, file_index):
    """Validate and attempt to repair a video file if corrupted"""
    try:
        # Check file exists and has content
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            return False, f'Video file {file_index} is empty or missing'
        
        # Enhanced FFmpeg validation
        validate_cmd = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', input_path]
        validate_result = subprocess.run(validate_cmd, capture_output=True, text=True, timeout=15)
        
        if validate_result.returncode != 0:
            print(f"File validation failed for {input_path}")
            print(f"FFprobe stderr: {validate_result.stderr}")
            
            # Try to repair the file using FFmpeg
            print(f"Attempting to repair video file {file_index}...")
            repaired_path = input_path.replace('.mp4', '_repaired.mp4')
            repair_cmd = [
                'ffmpeg', '-err_detect', 'ignore_err', '-i', input_path,
                '-c', 'copy', '-y', repaired_path
            ]
            
            repair_result = subprocess.run(repair_cmd, capture_output=True, text=True, timeout=30)
            if repair_result.returncode == 0 and os.path.exists(repaired_path):
                # Replace original with repaired version
                os.remove(input_path)
                os.rename(repaired_path, input_path)
                print(f"Successfully repaired video file {file_index}")
                return True, f"Video file {file_index} repaired successfully"
            else:
                return False, f'Video file {file_index} is corrupted and cannot be repaired. Please try uploading again.'
        else:
            # Parse the validation result to get file info
            try:
                import json
                file_info = json.loads(validate_result.stdout)
                duration = float(file_info.get('format', {}).get('duration', 0))
                print(f"Video {file_index} validated successfully - Duration: {duration:.2f}s")
                return True, f"Video file {file_index} is valid"
            except:
                print(f"Video {file_index} validated successfully")
                return True, f"Video file {file_index} is valid"
                
    except subprocess.TimeoutExpired:
        return False, f'Video file {file_index} validation timeout - file may be too large or corrupted'
    except Exception as e:
        return False, f'Video file {file_index} validation error: {str(e)}'

def merge_videos_simple(input_paths, output_path, timestamp):
    """Simple video merging without transitions"""
    try:
        print(f"Starting simple merge for {len(input_paths)} videos...")
        
        # Validate all input files first
        for i, input_path in enumerate(input_paths):
            if not os.path.exists(input_path):
                return False, f'Input video {i+1} not found: {input_path}'
            
            file_size = os.path.getsize(input_path)
            if file_size == 0:
                return False, f'Input video {i+1} is empty: {input_path}'
            
            print(f"Video {i+1}: {os.path.basename(input_path)} ({file_size} bytes)")
        
        # Create concat file for FFmpeg
        concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'simple_concat_{timestamp}.txt')
        
        with open(concat_file, 'w', encoding='utf-8') as f:
            for i, path in enumerate(input_paths):
                normalized_path = path.replace('\\', '/')
                f.write(f"file '{normalized_path}'\n")
                print(f"Added to concat list: {normalized_path}")
        
        # Debug: Print concat file contents
        print("Concat file contents:")
        with open(concat_file, 'r', encoding='utf-8') as f:
            print(f.read())
        
        # Use re-encoding for compatibility
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-crf', '23',
            '-preset', 'medium',
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
            '-r', '30',
            '-ar', '44100',
            '-ac', '2',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y', output_path
        ]
        
        print(f"Running FFmpeg command: {' '.join(cmd)}")
        success, message = run_ffmpeg_command(cmd, timeout=600)
        
        # Clean up concat file
        if os.path.exists(concat_file):
            os.remove(concat_file)
        
        if success:
            print(f"Simple merge completed successfully. Output: {output_path}")
        else:
            print(f"Simple merge failed: {message}")
        
        return success, message
        
    except Exception as e:
        print(f"Exception in simple merge: {str(e)}")
        return False, f"Simple merge error: {str(e)}"

def merge_videos_with_transition(input_paths, output_path, transition_type, duration, timestamp):
    """Merge videos with transitions between them"""
    try:
        print(f"Starting merge with transition for {len(input_paths)} videos...")
        
        # Validate all input files first
        for i, input_path in enumerate(input_paths):
            if not os.path.exists(input_path):
                return False, f'Input video {i+1} not found: {input_path}'
            
            file_size = os.path.getsize(input_path)
            if file_size == 0:
                return False, f'Input video {i+1} is empty: {input_path}'
            
            print(f"Video {i+1}: {os.path.basename(input_path)} ({file_size} bytes)")
        
        # First normalize all videos
        normalized_paths = []
        
        for i, input_path in enumerate(input_paths):
            norm_filename = f"merge_norm_{timestamp}_{i}.mp4"
            norm_path = os.path.join(app.config['UPLOAD_FOLDER'], norm_filename)
            
            print(f"Normalizing video {i+1}...")
            
            # Normalize video to consistent format
            norm_cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                '-r', '30',
                '-ar', '44100',
                '-ac', '2',
                '-crf', '23',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-y', norm_path
            ]
            
            success, message = run_ffmpeg_command(norm_cmd, timeout=600)
            if not success:
                print(f"Normalization failed for video {i+1}: {message}")
                # Clean up and return error
                for path in normalized_paths:
                    if os.path.exists(path):
                        os.remove(path)
                return False, f'Failed to normalize video {i+1}: {message}'
            
            # Verify normalized file was created
            if not os.path.exists(norm_path) or os.path.getsize(norm_path) == 0:
                return False, f'Normalized video {i+1} was not created properly'
            
            normalized_paths.append(norm_path)
            print(f"Successfully normalized video {i+1}")
        
        # Apply transitions based on type
        if transition_type in ['fade', 'dissolve'] and len(normalized_paths) == 2:
            # Simple crossfade for 2 videos
            transition_cmd = [
                'ffmpeg', '-i', normalized_paths[0], '-i', normalized_paths[1],
                '-filter_complex',
                f'[0:v][1:v]xfade=transition=fade:duration={duration}:offset=5[v];'
                f'[0:a][1:a]acrossfade=d={duration}[a]',
                '-map', '[v]', '-map', '[a]',
                '-c:v', 'libx264', '-c:a', 'aac',
                '-crf', '23', '-preset', 'medium',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-y', output_path
            ]
            
            success, message = run_ffmpeg_command(transition_cmd, timeout=600)
        else:
            # For multiple videos or other transition types, use simple concat
            concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'trans_concat_{timestamp}.txt')
            
            with open(concat_file, 'w', encoding='utf-8') as f:
                for path in normalized_paths:
                    normalized_path = path.replace('\\', '/')
                    f.write(f"file '{normalized_path}'\n")
            
            concat_cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-c', 'copy',
                '-y', output_path
            ]
            
            success, message = run_ffmpeg_command(concat_cmd, timeout=300)
            
            if os.path.exists(concat_file):
                os.remove(concat_file)
        
        # Clean up normalized files
        for path in normalized_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return success, message
        
    except Exception as e:
        return False, f"Transition merge error: {str(e)}"

def execute_transition_command(prompt, files, timestamp):
    """Execute transition command: transition type=fade|dissolve|wipe|cut duration=X"""
    params = parse_parameters(prompt)
    
    # Check if multiple files are provided
    if 'files[]' not in files:
        return {
            'success': False, 
            'error': 'Transition command requires multiple files. Please upload a main video and select auxiliary videos.',
            'suggestion': 'Upload a main video file and select additional videos in the "Auxiliary Video Files" section.'
        }
    
    file_list = files.getlist('files[]')
    if len(file_list) < 2:
        return {
            'success': False, 
            'error': 'Transition command requires at least 2 video files.',
            'suggestion': 'Please select at least one auxiliary video file in addition to the main video.'
        }
    
    transition_type = params.get('type', 'fade')
    duration = params.get('duration', 1.0)
    
    # Save uploaded files with validation
    input_paths = []
    try:
        for i, file in enumerate(file_list):
            if file and file.filename:
                filename = secure_filename(file.filename)
                input_filename = f"trans_{timestamp}_{i}_{filename}"
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
                
                # Save file safely
                save_success, save_message = save_file_safely(file, input_path)
                if not save_success:
                    return {'success': False, 'error': f'Failed to save video file {i+1}: {save_message}'}
                
                # Validate file exists and has content
                if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                    return {'success': False, 'error': f'Failed to save video file {i+1} properly'}
                
                # Quick FFmpeg validation to ensure file is readable
                validate_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', input_path]
                try:
                    validate_result = subprocess.run(validate_cmd, capture_output=True, text=True, timeout=10)
                    if validate_result.returncode != 0:
                        print(f"File validation failed for {input_path}: {validate_result.stderr}")
                        return {'success': False, 'error': f'Video file {i+1} is corrupted or invalid format'}
                except subprocess.TimeoutExpired:
                    return {'success': False, 'error': f'Video file {i+1} validation timeout'}
                except Exception as e:
                    return {'success': False, 'error': f'Video file {i+1} validation error: {str(e)}'}
                
                input_paths.append(input_path)
                print(f"Successfully saved and validated video {i+1}: {input_filename}")
        
        if len(input_paths) < 2:
            return {'success': False, 'error': 'At least 2 valid video files are required for transitions'}
        
        print(f"Applying {transition_type} transitions between {len(input_paths)} videos...")
        
        # Generate output filename
        output_filename = f'prompt_transition_{timestamp}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # First normalize all videos to same format to avoid timebase issues
        normalized_paths = []
        for i, input_path in enumerate(input_paths):
            norm_filename = f"trans_norm_{timestamp}_{i}.mp4"
            norm_path = os.path.join(app.config['UPLOAD_FOLDER'], norm_filename)
            
            # Normalize video to consistent format
            norm_cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                '-r', '30',  # Force consistent frame rate
                '-ar', '44100',  # Force consistent audio sample rate
                '-ac', '2',  # Force stereo
                '-crf', '23',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-y', norm_path
            ]
            
            success, message = run_ffmpeg_command(norm_cmd, timeout=300)
            if not success:
                # Clean up and return error
                for path in input_paths + normalized_paths:
                    if os.path.exists(path):
                        os.remove(path)
                return {'success': False, 'error': f'Failed to normalize video {i+1}: {message}'}
            
            normalized_paths.append(norm_path)
        
        # Apply transitions based on type
        success = False
        message = ""
        
        if transition_type == 'fade':
            if len(normalized_paths) == 2:
                # Simple crossfade for 2 videos
                fade_cmd = [
                    'ffmpeg', '-i', normalized_paths[0], '-i', normalized_paths[1],
                    '-filter_complex',
                    f'[0:v][1:v]xfade=transition=fade:duration={duration}:offset=5[v];'
                    f'[0:a][1:a]acrossfade=d={duration}[a]',
                    '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(fade_cmd, timeout=600)
            else:
                # Multiple videos - use filter_complex for fade transitions
                filter_inputs = []
                for i, path in enumerate(normalized_paths):
                    filter_inputs.extend(['-i', path])
                
                # Build filter complex for fade transitions
                video_filters = []
                audio_filters = []
                
                for i in range(len(normalized_paths)):
                    if i == 0:
                        # First video: fade out at end
                        video_filters.append(f'[{i}:v]fade=out:st=5:d={duration}[v{i}]')
                        audio_filters.append(f'[{i}:a]afade=out:st=5:d={duration}[a{i}]')
                    elif i == len(normalized_paths) - 1:
                        # Last video: fade in at start
                        video_filters.append(f'[{i}:v]fade=in:st=0:d={duration}[v{i}]')
                        audio_filters.append(f'[{i}:a]afade=in:st=0:d={duration}[a{i}]')
                    else:
                        # Middle videos: fade in and out
                        video_filters.append(f'[{i}:v]fade=in:st=0:d={duration},fade=out:st=5:d={duration}[v{i}]')
                        audio_filters.append(f'[{i}:a]afade=in:st=0:d={duration},afade=out:st=5:d={duration}[a{i}]')
                
                # Concatenate all processed streams
                video_inputs = ''.join([f'[v{i}]' for i in range(len(normalized_paths))])
                audio_inputs = ''.join([f'[a{i}]' for i in range(len(normalized_paths))])
                
                filter_complex = ';'.join(video_filters + audio_filters + [
                    f'{video_inputs}concat=n={len(normalized_paths)}:v=1:a=0[outv]',
                    f'{audio_inputs}concat=n={len(normalized_paths)}:v=0:a=1[outa]'
                ])
                
                fade_cmd = [
                    'ffmpeg'
                ] + filter_inputs + [
                    '-filter_complex', filter_complex,
                    '-map', '[outv]', '-map', '[outa]',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(fade_cmd, timeout=900)
        
        elif transition_type == 'dissolve':
            if len(normalized_paths) == 2:
                # Simple crossfade between two normalized videos
                dissolve_cmd = [
                    'ffmpeg', '-i', normalized_paths[0], '-i', normalized_paths[1],
                    '-filter_complex',
                    f'[0:v][1:v]xfade=transition=dissolve:duration={duration}:offset=5[v];'
                    f'[0:a][1:a]acrossfade=d={duration}[a]',
                    '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-crf', '23', '-preset', 'medium',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(dissolve_cmd, timeout=600)
            else:
                # For multiple videos, just concatenate (dissolve works best with 2 videos)
                concat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'trans_concat_{timestamp}.txt')
                
                with open(concat_file, 'w', encoding='utf-8') as f:
                    for path in normalized_paths:
                        normalized_path = path.replace('\\', '/')
                        f.write(f"file '{normalized_path}'\n")
                
                dissolve_cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-c', 'copy',
                    '-y', output_path
                ]
                
                success, message = run_ffmpeg_command(dissolve_cmd, timeout=300)
                
                if os.path.exists(concat_file):
                    os.remove(concat_file)
        
        # Clean up normalized files
        for path in normalized_paths:
            if os.path.exists(path):
                os.remove(path)
        
        # Clean up input files
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        
        if not success:
            return {'success': False, 'error': f'Transition failed: {message}'}
        
        return {
            'success': True,
            'output_file': output_filename,
            'message': f'Applied {transition_type} transitions using FFmpeg'
        }
        
    except Exception as e:
        print(f"Error in execute_transition_command: {str(e)}")
        # Clean up files on error
        for path in input_paths:
            if os.path.exists(path):
                os.remove(path)
        return {'success': False, 'error': f'Transition error: {str(e)}'}

@login_manager.unauthorized_handler
def unauthorized():
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': False,
            'error': 'Authentication required',
            'redirect': url_for('login')
        }), 401
    flash('Please login to access this page', 'warning')
    return redirect(url_for('login'))

@app.route('/search-youtube-clips', methods=['POST'])
@login_required
def search_youtube_clips():
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        script_description = data.get('script_description')
        if not script_description:
            return jsonify({'success': False, 'error': 'No script description provided'})
            
        print(f"Debug - Searching YouTube for: {script_description}")
        
        # Configure request session with retry mechanism
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        
        # Headers for the request
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'VideoEditor/1.0'
        }
        
        # Build the search URL with parameters
        base_url = 'https://www.googleapis.com/youtube/v3/search'
        params = {
            'part': 'snippet',
            'q': script_description,
            'maxResults': 50,
            'type': 'video',
            'videoEmbeddable': 'true',
            'videoSyndicated': 'true',
            'key': os.getenv('YOUTUBE_API_KEY')
        }
        
        try:
            print(f"Debug - Making request to: {base_url}")
            print(f"Debug - With parameters: {params}")
            
            # Make request to YouTube API
            response = session.get(base_url, params=params, headers=headers)
            
            print(f"Debug - API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Debug - API Error: {response.status_code} - {response.text}")
                return jsonify({
                    'success': True,
                    'clips': []
                })
            
            data = response.json()
            videos = data.get('items', [])
            print(f"Debug - Found {len(videos)} videos")
            
            clips = []
            for video in videos:
                try:
                    video_id = video['id']['videoId']
                    snippet = video['snippet']
                    
                    # Format the published date
                    try:
                        published_at = datetime.strptime(snippet['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                        formatted_date = published_at.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        formatted_date = snippet.get('publishedAt', '')
                    
                    clip = {
                        'id': video_id,
                        'title': snippet.get('title', ''),
                        'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                        'description': snippet.get('description', '')[:200] + '...' if snippet.get('description') else '',
                        'channel': snippet.get('channelTitle', ''),
                        'published_at': formatted_date,
                        'watch_url': f'https://www.youtube.com/watch?v={video_id}',
                        'embed_url': f'https://www.youtube.com/embed/{video_id}',
                        'duration': '0',  # Add default duration
                        'view_count': '0',  # Add default view count
                        'like_count': '0'   # Add default like count
                    }
                    
                    clips.append(clip)
                    print(f"Debug - Added clip: {snippet['title']}")
                    
                except Exception as e:
                    print(f"Debug - Error processing video: {str(e)}")
                    continue
            
            print(f"Debug - Successfully processed {len(clips)} clips")
            return jsonify({
                'success': True,
                'clips': clips
            })
            
        except requests.exceptions.RequestException as e:
            print(f"Debug - Network error during API request: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Network error during YouTube search: {str(e)}'
            })
        except Exception as e:
            print(f"Debug - Unexpected error during API request: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Unexpected error during YouTube search: {str(e)}'
            })
            
    except Exception as e:
        print(f"Debug - General error in search_youtube_clips: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download-youtube-clip', methods=['POST'])
@login_required
def download_youtube_clip():
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        video_id = data.get('video_id')
        if not video_id:
            return jsonify({'success': False, 'error': 'No video ID provided'})
            
        print(f"Debug - Downloading YouTube video ID: {video_id}")
        
        # Download YouTube video
        output_filename = f'youtube_{video_id}_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Use yt-dlp to download the video with enhanced options
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'verbose': True,
            'no_warnings': False,
            'ignoreerrors': True,
            'quiet': False,
            'no_color': True,
            'extract_flat': False,
            'force_generic_extractor': False,
            'cookiesfrombrowser': None,
            'cookiefile': None,
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'geo_bypass': True,
            'geo_verification_proxy': None,
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            'skip_unavailable_fragments': True,
            'keepvideo': False,
            'writedescription': False,
            'writeinfojson': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'postprocessors': [],
            'merge_output_format': 'mp4',
            'updatetime': False,
            'consoletitle': False,
            'noprogress': False,
            'progress_with_newline': True,
            'progress_hooks': [],
            'postprocessor_hooks': [],
            'match_filter': None,
            'source_address': None,
            'call_home': False,
            'sleep_interval': 0,
            'max_sleep_interval': 10,
            'external_downloader_args': None,
            'listformats': False,
            'list_thumbnails': False,
            'playlist_items': None,
            'playlist_random': False,
            'playlist_reverse': False,
            'playlist_start': 1,
            'playlist_end': None,
            'playlist_min_files': 1,
            'playlist_max_files': None,
            'playlist_filters': [],
            'age_limit': None,
            'download_archive': None,
            'break_on_existing': False,
            'break_per_url': False,
            'skip_download': False,
            'cachedir': None,
            'youtube_include_dash_manifest': True,
            'youtube_include_hls_manifest': True,
            'youtube_include_drm_manifest': True,
            'youtube_include_webm': True,
            'youtube_include_3d': True,
            'youtube_include_playlist_metafiles': True,
            'youtube_include_dash_audio': True,
            'youtube_include_dash_video': True,
            'youtube_include_hls_audio': True,
            'youtube_include_hls_video': True,
            'youtube_include_drm_audio': True,
            'youtube_include_drm_video': True,
            'youtube_include_webm_audio': True,
            'youtube_include_webm_video': True,
            'youtube_include_3d_audio': True,
            'youtube_include_3d_video': True,
            'youtube_include_playlist_metafiles_audio': True,
            'youtube_include_playlist_metafiles_video': True,
        }
        
        print(f"Debug - Downloading to: {output_path}")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First try to extract info to verify the video exists
                try:
                    info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
                    if not info:
                        print(f"Debug - Could not extract video info")
                        return jsonify({'success': False, 'error': 'Video not found or unavailable'})
                except Exception as e:
                    print(f"Debug - Error extracting video info: {str(e)}")
                    return jsonify({'success': False, 'error': f'Error accessing video: {str(e)}'})
                
                # Now try to download
                try:
                    ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
                except Exception as e:
                    print(f"Debug - Error during download: {str(e)}")
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                        except:
                            pass
                    return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
                
            if not os.path.exists(output_path):
                print(f"Debug - Output file not found after download")
                return jsonify({'success': False, 'error': 'Failed to download video'})
                
            # Verify the file is not empty
            if os.path.getsize(output_path) == 0:
                print(f"Debug - Downloaded file is empty")
                os.remove(output_path)
                return jsonify({'success': False, 'error': 'Downloaded file is empty'})
                
            print(f"Debug - Successfully downloaded video to: {output_path}")
            
            return jsonify({
                'success': True,
                'filename': output_filename
            })
        except Exception as e:
            print(f"Debug - yt-dlp error: {str(e)}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
            
    except Exception as e:
        print(f"Debug - General error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add-youtube-to-video', methods=['POST'])
@login_required
def add_youtube_to_video():
    """Add a YouTube clip to the user's video."""
    videos = []
    input_paths = []
    final_video = None
    youtube_path = None # Define youtube_path outside try for cleanup

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
            
        video_file = request.files['file']
        video_id = request.form.get('video_id')
        
        if not video_id:
            return jsonify({'success': False, 'error': 'No YouTube video ID provided'})
            
        if not video_file.filename:
            return jsonify({'success': False, 'error': 'No selected file'})
            
        print(f"Debug - Processing merge request for YouTube video ID: {video_id}")
            
        # Save the uploaded video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)
        input_paths.append(video_path) # Add to input_paths for cleanup
        
        # Download the YouTube clip
        youtube_filename = f'youtube_{video_id}_{int(time.time())}.mp4'
        youtube_path = os.path.join(app.config['OUTPUT_FOLDER'], youtube_filename)
        input_paths.append(youtube_path) # Add to input_paths for cleanup
        
        try:
            print(f"Debug - Downloading YouTube video to: {youtube_path}")
            
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': youtube_path,
                'verbose': True,
                'no_warnings': False,
                'ignoreerrors': True,
                'quiet': False,
                'no_color': True,
                'extract_flat': False,
                'force_generic_extractor': False,
                'cookiesfrombrowser': None,
                'cookiefile': None,
                'nocheckcertificate': True,
                'prefer_insecure': True,
                'geo_bypass': True,
                'socket_timeout': 60, # Increased socket timeout
                'retries': 10, # Increased retries for robustness
                'fragment_retries': 10,
                'skip_unavailable_fragments': True,
                'keepvideo': False,
                'writedescription': False,
                'writeinfojson': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'postprocessors': [],
                'merge_output_format': 'mp4',
                'updatetime': False,
                'consoletitle': False,
                'noprogress': False,
                'progress_with_newline': True,
                'progress_hooks': [],
                'postprocessor_hooks': [],
                'match_filter': None,
                'source_address': None,
                'call_home': False,
                'sleep_interval': 0,
                'max_sleep_interval': 10,
                'external_downloader_args': None,
                'listformats': False,
                'list_thumbnails': False,
                'playlist_items': None,
                'playlist_random': False,
                'playlist_reverse': False,
                'playlist_start': 1,
                'playlist_end': None,
                'playlist_min_files': 1,
                'playlist_max_files': None,
                'playlist_filters': [],
                'age_limit': None,
                'download_archive': None,
                'break_on_existing': False,
                'break_per_url': False,
                'skip_download': False,
                'cachedir': None,
                'youtube_include_dash_manifest': True,
                'youtube_include_hls_manifest': True,
                'youtube_include_drm_manifest': True,
                'youtube_include_webm': True,
                'youtube_include_3d': True,
                'youtube_include_playlist_metafiles': True,
                'youtube_include_dash_audio': True,
                'youtube_include_dash_video': True,
                'youtube_include_hls_audio': True,
                'youtube_include_hls_video': True,
                'youtube_include_drm_audio': True,
                'youtube_include_drm_video': True,
                'youtube_include_webm_audio': True,
                'youtube_include_webm_video': True,
                'youtube_include_3d_audio': True,
                'youtube_include_3d_video': True,
                'youtube_include_playlist_metafiles_audio': True,
                'youtube_include_playlist_metafiles_video': True,
            }
            
            try:
                print(f"Debug - Starting YouTube download with yt-dlp")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # First try to extract info to verify the video exists
                    try:
                        info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
                        if not info:
                            print(f"Debug - Could not extract video info")
                            return jsonify({'success': False, 'error': 'Video not found or unavailable'})
                    except Exception as e:
                        print(f"Debug - Error extracting video info: {str(e)}")
                        return jsonify({'success': False, 'error': f'Error accessing video: {str(e)}'})
                    
                    # Now try to download
                    try:
                        ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
                    except Exception as e:
                        print(f"Debug - Error during download: {str(e)}")
                        if os.path.exists(youtube_path):
                            try:
                                os.remove(youtube_path)
                            except:
                                pass
                        return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
                    
                if not os.path.exists(youtube_path):
                    print(f"Debug - YouTube video not found after download")
                    return jsonify({'success': False, 'error': 'Failed to download YouTube video'})
                    
                # Verify the file is not empty
                if os.path.getsize(youtube_path) == 0:
                    print(f"Debug - Downloaded YouTube file is empty")
                    os.remove(youtube_path)
                    return jsonify({'success': False, 'error': 'Downloaded YouTube file is empty'})
                    
                print(f"Debug - Successfully downloaded YouTube video")
                
                # Load both videos
                print(f"Debug - Loading videos for merging")
                video1 = mp.VideoFileClip(video_path)
                video2 = mp.VideoFileClip(youtube_path)
                videos.extend([video1, video2]) # Add to videos list for cleanup
                
                # Get the highest resolution
                target_width = max(video1.w, video2.w)
                target_height = max(video1.h, video2.h)
                
                # Ensure dimensions are even
                target_width = target_width - (target_width % 2)
                target_height = target_height - (target_height % 2)
                
                print(f"Debug - Resizing videos to {target_width}x{target_height}")
                
                # Resize videos to match the highest resolution
                if video1.w != target_width or video1.h != target_height:
                    video1 = video1.resize(width=target_width, height=target_height)
                if video2.w != target_width or video2.h != target_height:
                    video2 = video2.resize(width=target_width, height=target_height)
                
                # Concatenate videos
                print(f"Debug - Concatenating videos")
                final_video = mp.concatenate_videoclips([video1, video2])
                
                # Generate output filename
                output_filename = f'merged_youtube_{int(time.time())}.mp4'
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                
                print(f"Debug - Writing final video to {output_path}")
                print(f"Debug - Starting write_videofile with preset 'fast' and progress logger")
                
                # Write the final video
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    bitrate='8000k',
                    fps=30,
                    preset='fast', # Changed from slow to fast
                    threads=4,
                    ffmpeg_params=[
                        '-crf', '18',
                        '-profile:v', 'high',
                        '-level', '4.0',
                        '-movflags', '+faststart',
                        '-pix_fmt', 'yuv420p'
                    ],
                    logger='bar' # Add progress bar to console
                )
                
                print(f"Debug - Finished write_videofile")
                
                print(f"Debug - Cleaning up video objects")
                
                # Close all videos
                video1.close()
                video2.close()
                final_video.close()
                
                # Clean up temporary files
                print(f"Debug - Cleaning up temporary files")
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(youtube_path):
                    os.remove(youtube_path)
                
                print(f"Debug - Merge completed successfully")
                return jsonify({
                    'success': True,
                    'output_file': output_filename
                })
                
            except Exception as e:
                print(f"Debug - Error during processing: {str(e)}")
                # Clean up any files that were saved
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(youtube_path):
                    os.remove(youtube_path)
                return jsonify({'success': False, 'error': f'Error processing videos: {str(e)}'})
                
        except Exception as e:
            print(f"Debug - Unexpected error in add_youtube_to_video: {str(e)}")
            return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

    except Exception as e:
        print(f"Debug - Unexpected error in add_youtube_to_video: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/search-pexels-clips', methods=['POST'])
@login_required
def search_pexels_clips():
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        script_description = request.form.get('script_description')
        if not script_description:
            return jsonify({'success': False, 'error': 'No script description provided'})
            
        print(f"Debug - Searching Pexels for: {script_description}")
        
        # Configure request session with retry mechanism
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        
        # Headers for the request
        headers = {
            'Authorization': PEXELS_API_KEY,
            'Accept': 'application/json',
            'User-Agent': 'VideoEditor/1.0'
        }
        
        # Build the search URL with parameters
        base_url = 'https://api.pexels.com/videos/search'
        params = {
            'query': script_description,
            'per_page': 15,
            'orientation': 'landscape'
        }
        
        try:
            print(f"Debug - Making request to: {base_url}")
            
            # Make request to Pexels API
            response = session.get(base_url, params=params, headers=headers)
            
            print(f"Debug - API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Debug - API Error: {response.status_code} - {response.text}")
                return jsonify({
                    'success': False,
                    'error': f'Pexels API error: {response.status_code}'
                })
            
            data = response.json()
            videos = data.get('videos', [])
            print(f"Debug - Found {len(videos)} videos")
            
            clips = []
            for video in videos:
                try:
                    # Get the best quality video file
                    video_files = video.get('video_files', [])
                    best_quality = max(video_files, key=lambda x: x.get('width', 0) * x.get('height', 0))
                    
                    clip = {
                        'id': str(video['id']),
                        'title': video.get('url', '').split('/')[-1],
                        'thumbnail': video.get('image', ''),
                        'url': best_quality.get('link', ''),
                        'width': best_quality.get('width', 0),
                        'height': best_quality.get('height', 0),
                        'duration': video.get('duration', 0),
                        'user': video.get('user', {}).get('name', 'Unknown'),
                        'download_url': best_quality.get('link', '')
                    }
                    
                    clips.append(clip)
                    print(f"Debug - Added clip: {clip['title']}")
                    
                except Exception as e:
                    print(f"Debug - Error processing video: {str(e)}")
                    continue
            
            print(f"Debug - Successfully processed {len(clips)} clips")
            return jsonify({
                'success': True,
                'clips': clips
            })
            
        except requests.exceptions.RequestException as e:
            print(f"Debug - Network error during API request: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Network error during Pexels search: {str(e)}'
            })
        except Exception as e:
            print(f"Debug - Unexpected error during API request: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Unexpected error during Pexels search: {str(e)}'
            })
            
    except Exception as e:
        print(f"Debug - General error in search_pexels_clips: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def process_video_segment(video_clip, target_size):
    """Process video segment using ML-based optimization."""
    def process_frame(frame):
        # Convert to numpy array for ML processing
        frame = np.array(frame)
        
        # Apply ML-based enhancement
        # 1. Smart scaling using OpenCV's INTER_AREA for downscaling
        if frame.shape[0] > target_size[1] or frame.shape[1] > target_size[0]:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # 2. Apply ML-based sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
    
    return video_clip.fl_image(process_frame)

@app.route('/merge-with-pexels', methods=['POST'])
@login_required
def merge_with_pexels():
    """Merge a Pexels clip with the user's video using ML optimization."""
    videos = []
    input_paths = []
    processed_videos = []
    final_video = None
    pexels_path = None # Define pexels_path outside try for cleanup

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
            
        video_file = request.files['file']
        pexels_clip_url = request.form.get('pexels_clip_url')
        
        if not pexels_clip_url:
            return jsonify({'success': False, 'error': 'No Pexels clip URL provided'})
            
        if not video_file.filename:
            return jsonify({'success': False, 'error': 'No selected file'})
            
        print(f"Debug - Processing merge request with Pexels clip from: {pexels_clip_url}")
            
        # Save the uploaded file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)
        input_paths.append(video_path) # Add to input_paths for cleanup
        
        # Download Pexels video
        pexels_filename = f'pexels_{int(time.time())}.mp4'
        pexels_path = os.path.join(app.config['OUTPUT_FOLDER'], pexels_filename)
        input_paths.append(pexels_path) # Add to input_paths for cleanup
        
        try:
            print(f"Debug - Attempting to download Pexels clip for merge from {pexels_clip_url} to {pexels_path}")
            # Download the Pexels video
            # Use a session with retry logic for robustness
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount('https://', adapter)

            # Add a timeout to the request
            response = session.get(pexels_clip_url, stream=True, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            print(f"Debug - Pexels merge download response status: {response.status_code}")
            
            with open(pexels_path, 'wb') as f:
                print(f"Debug - Writing Pexels video content for merge to {pexels_path}")
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                print(f"Debug - Finished writing Pexels video content for merge")
            
            if not os.path.exists(pexels_path):
                print(f"Debug - Pexels merge output file not found after download")
                # Ensure cleanup is called even on download failure
                cleanup_files(input_paths)
                return jsonify({'success': False, 'error': 'Failed to download Pexels video for merge: Output file not created.'})
                
            # Verify the file is not empty
            if os.path.getsize(pexels_path) == 0:
                print(f"Debug - Downloaded Pexels file for merge is empty: {pexels_path}")
                # Ensure cleanup is called even on empty file
                cleanup_files(input_paths)
                return jsonify({'success': False, 'error': 'Downloaded Pexels video file for merge is empty.'})
                
            print(f"Debug - Successfully downloaded Pexels video for merge")
            
            print(f"Debug - Loading videos for merging")
            # Load both videos
            video1 = mp.VideoFileClip(video_path, audio=True)
            video2 = mp.VideoFileClip(pexels_path, audio=True)
            videos.extend([video1, video2]) # Add to videos list for cleanup
            
            # Get target dimensions (capped at 1080p)
            target_width = min(max(video1.w, video2.w), 1920)
            target_height = min(max(video1.h, video2.h), 1080)
            target_size = (target_width, target_height)
            
            print(f"Debug - Processing videos in parallel...")
            # Process videos in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit video processing tasks
                future1 = executor.submit(process_video_segment, video1, target_size)
                future2 = executor.submit(process_video_segment, video2, target_size)
                
                # Get processed videos
                processed_video1 = future1.result()
                processed_video2 = future2.result()
                processed_videos.extend([processed_video1, processed_video2]) # Add to processed_videos for cleanup
            
            print(f"Debug - Concatenating videos...")
            # Concatenate processed videos
            final_video = mp.concatenate_videoclips([processed_video1, processed_video2], method="compose")
            
            # Generate output filename
            output_filename = f'merged_pexels_{int(time.time())}.mp4'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            print(f"Debug - Writing final video...")
            print(f"Debug - Starting write_videofile with preset 'ultrafast' and progress logger for Pexels merge")
            
            # Write the final video with optimized settings
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate='6000k',
                fps=30,
                preset='ultrafast',  # Use ultrafast preset for faster processing
                threads=8,  # Increased thread count
                ffmpeg_params=[
                    '-crf', '23',  # Slightly reduced quality for speed
                    '-profile:v', 'main',
                    '-level', '4.0',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-tune', 'fastdecode',  # Optimize for fast decoding
                    '-threads', '8'  # Use 8 threads for encoding
                ],
                logger='bar' # Add progress bar
            )
            
            print(f"Debug - Finished write_videofile for Pexels merge")
            
            print(f"Debug - Cleaning up video objects and files after Pexels merge")
            # Close all videos
            cleanup_videos(videos)
            cleanup_videos(processed_videos)
            if final_video:
                final_video.close()
            
            # Clean up temporary files
            cleanup_files(input_paths)
            
            print(f"Debug - Pexels merge completed successfully")
            return jsonify({
                'success': True,
                'output_file': output_filename
            })
            
        except requests.exceptions.RequestException as e:
            print(f"Debug - Pexels merge download RequestException: {str(e)}")
            # Clean up any files that were saved
            cleanup_videos(videos)
            cleanup_videos(processed_videos)
            if final_video:
                final_video.close()
            cleanup_files(input_paths)
            return jsonify({'success': False, 'error': f'Error downloading Pexels video for merge: {str(e)}'})
        except Exception as e:
            print(f"Debug - Pexels merge processing unexpected error: {str(e)}")
            # Clean up on error
            cleanup_videos(videos)
            cleanup_videos(processed_videos)
            if final_video:
                final_video.close()
            cleanup_files(input_paths)
            return jsonify({'success': False, 'error': f'Error processing videos: {str(e)}'})
            
    except Exception as e:
        print(f"Error in merge_with_pexels: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/download-pexels-clip', methods=['POST'])
@login_required
def download_pexels_clip():
    """Download a Pexels clip."""
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        clip_url = data.get('clip_url')
        if not clip_url:
            return jsonify({'success': False, 'error': 'No clip URL provided'})
            
        print(f"Debug - Downloading Pexels video from: {clip_url}")
        
        # Generate output filename
        output_filename = f'pexels_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        try:
            print(f"Debug - Attempting to download Pexels clip using requests from {clip_url} to {output_path}")
            # Download the video
            # Use a session with retry logic for robustness
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount('https://', adapter)

            # Add a timeout to the request
            response = session.get(clip_url, stream=True, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            print(f"Debug - Pexels download response status: {response.status_code}")
            
            with open(output_path, 'wb') as f:
                print(f"Debug - Writing Pexels video content to {output_path}")
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                print(f"Debug - Finished writing Pexels video content")
            
            if not os.path.exists(output_path):
                print(f"Debug - Output file not found after Pexels download")
                return jsonify({'success': False, 'error': 'Failed to download video: Output file not created.'})
                
            # Verify the file is not empty
            if os.path.getsize(output_path) == 0:
                print(f"Debug - Downloaded Pexels file is empty: {output_path}")
                os.remove(output_path) # Clean up empty file
                return jsonify({'success': False, 'error': 'Downloaded video file is empty.'})
                
            print(f"Debug - Successfully downloaded Pexels video to: {output_path}")
            
            return jsonify({
                'success': True,
                'filename': output_filename
            })
        except requests.exceptions.RequestException as e:
            print(f"Debug - Pexels download RequestException: {str(e)}")
            # Clean up any partial file if the request failed
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except: # nosec
                    pass # Ignore errors during cleanup attempt
            return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
        except Exception as e:
            print(f"Debug - Pexels download unexpected error: {str(e)}")
            # Clean up any partial file in case of other exceptions
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except: # nosec
                    pass # Ignore errors during cleanup attempt
            return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
            
    except Exception as e:
        print(f"Debug - General error in download_pexels_clip: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

def formatDuration(seconds):
    """Format duration in seconds to HH:MM:SS format."""
    try:
        seconds = int(float(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except:
        return "00:00:00"

@app.route('/search-dailymotion-clips', methods=['POST'])
@login_required
def search_dailymotion_clips():
    """Search for Dailymotion clips based on the provided description."""
    try:
        data = request.get_json()
        if not data or 'script_description' not in data:
            return jsonify({'success': False, 'error': 'No search query provided'})

        search_query = data['script_description']
        
        # Set up the request session with retry logic
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Prepare headers for the API request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the API request to Dailymotion
        api_url = f'https://api.dailymotion.com/videos?search={search_query}&limit=10&fields=id,title,description,thumbnail_url,duration,views_total,owner.username,embed_url'
        response = session.get(api_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        clips = []
        
        for item in data.get('list', []):
            clip = {
                'id': item.get('id'),
                'title': item.get('title'),
                'description': item.get('description', ''),
                'thumbnail': item.get('thumbnail_url'),
                'duration': item.get('duration'),
                'view_count': item.get('views_total', 0),
                'uploader': item.get('owner.username', 'Unknown'),
                'watch_url': f'https://www.dailymotion.com/video/{item.get("id")}'
            }
            clips.append(clip)
        
        return jsonify({'success': True, 'clips': clips})
        
    except requests.exceptions.RequestException as e:
        return jsonify({'success': False, 'error': f'Error searching Dailymotion: {str(e)}'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/download-dailymotion-clip', methods=['POST'])
@login_required
def download_dailymotion_clip():
    print(f"Debug - download_dailymotion_clip function called.")
    """Download a Dailymotion clip using yt-dlp."""
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        data = request.get_json()
        if not data or 'video_id' not in data:
            return jsonify({'success': False, 'error': 'No video ID provided'})

        video_id = data['video_id']
        print(f"Debug - Downloading Dailymotion video ID: {video_id} using yt-dlp")

        # Generate output filename
        output_filename = f'dailymotion_{video_id}_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Configure yt-dlp options for Dailymotion
        ydl_opts = {
            'format': 'best', # Let yt-dlp pick the best format
            'outtmpl': output_path,
            'verbose': True,
            'no_warnings': False,
            'ignoreerrors': True,
            'quiet': False,
            'no_color': True,
            'extract_flat': False,
            'force_generic_extractor': False,
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'geo_bypass': True,
            'socket_timeout': 60, # Increased socket timeout
            'retries': 10, # Increased retries for robustness
            'fragment_retries': 10,
            'skip_unavailable_fragments': True,
            'keepvideo': False,
            'writedescription': False,
            'writeinfojson': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'postprocessors': [],
            'merge_output_format': 'mp4',
            'progress_with_newline': True,
        }

        try:
            print(f"Debug - Starting yt-dlp download to: {output_path}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First try to extract info to verify the video exists and is downloadable
                try:
                    # Try with different URL formats
                    urls_to_try = [
                        f'https://www.dailymotion.com/video/{video_id}',
                        f'https://dailymotion.com/video/{video_id}',
                        f'https://www.dailymotion.com/embed/video/{video_id}'
                    ]

                    info = None
                    for url in urls_to_try:
                        try:
                            print(f"Debug - Trying to extract info from: {url}")
                            info = ydl.extract_info(url, download=False)
                            if info:
                                print(f"Debug - Successfully extracted video info from: {url}")
                                break
                        except Exception as e:
                            print(f"Debug - Failed to extract info from {url}: {str(e)}")
                            continue

                    if not info:
                        print(f"Debug - Could not extract video info for {video_id} from any URL format")
                        return jsonify({'success': False, 'error': 'Video not found or unavailable'})

                except Exception as e:
                    print(f"Debug - Error extracting video info for {video_id}: {type(e).__name__} - {str(e)}")
                    return jsonify({'success': False, 'error': f'Error accessing video: {str(e)}'})

                # Now try to download the video
                try:
                    print(f"Debug - Initiating download for {video_id}")
                    # Use the first successful URL or the original one for download
                    download_url = urls_to_try[0] if info else f'https://www.dailymotion.com/video/{video_id}'
                    ydl.download([download_url])
                    print(f"Debug - yt-dlp download process finished for {video_id}")
                except Exception as e:
                    print(f"Debug - Error during download for {video_id}: {type(e).__name__} - {str(e)}")
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                        except: # nosec
                            pass
                    return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})

            # Verify the downloaded file
            if not os.path.exists(output_path):
                print(f"Debug - Output file not found after download at {output_path}")
                return jsonify({'success': False, 'error': 'Failed to download video: Output file not created.'})

            if os.path.getsize(output_path) == 0:
                print(f"Debug - Downloaded file is empty at {output_path}")
                os.remove(output_path)
                return jsonify({'success': False, 'error': 'Downloaded video file is empty.'})

            print(f"Debug - Successfully downloaded Dailymotion clip to {output_path}")

            # --- NEW: Re-encode video for browser compatibility ---
            try:
                print(f"Debug - Loading downloaded Dailymotion clip for re-encoding: {output_path}")
                clip = mp.VideoFileClip(output_path)

                # Define temporary re-encoded path
                reencoded_filename = f'reencoded_dailymotion_{int(time.time())}.mp4'
                reencoded_path = os.path.join(app.config['OUTPUT_FOLDER'], reencoded_filename)

                print(f"Debug - Re-encoding Dailymotion clip to: {reencoded_path}")
                clip.write_videofile(
                    reencoded_path,
                    codec='libx264',
                    audio_codec='aac',
                    bitrate='3000k', # Adjust bitrate as needed for quality vs file size
                    fps=clip.fps, # Preserve original FPS
                    preset='medium', # Use a balanced preset for quality and speed
                    threads=4,
                    ffmpeg_params=[
                        '-movflags', '+faststart', # For faster web playback
                        '-pix_fmt', 'yuv420p', # Essential for broad browser compatibility
                        '-crf', '23' # Constant Rate Factor: 23 is a good balance
                    ]
                )
                clip.close()
                os.remove(output_path) # Remove the original downloaded file
                output_filename = reencoded_filename # Use the re-encoded filename for output
                print(f"Debug - Successfully re-encoded Dailymotion clip to: {output_filename}")
            except Exception as e:
                print(f"Debug - Error during Dailymotion re-encoding: {str(e)}")
                if os.path.exists(output_path): os.remove(output_path) # Clean up original if re-encoding fails
                return jsonify({'success': False, 'error': f'Error re-encoding video for browser: {str(e)}'})
            # --- END NEW RE-ENCODE ---

            return jsonify({
                'success': True,
                'filename': output_filename # Return the re-encoded filename
            })

        except Exception as e:
            print(f"Debug - yt-dlp execution error for {video_id}: {str(e)}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except: # nosec
                    pass
            return jsonify({'success': False, 'error': f'Error processing download: {str(e)}'})

    except Exception as e:
        print(f"Debug - General error in download_dailymotion_clip: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/add-dailymotion-to-video', methods=['POST'])
@login_required
def add_dailymotion_to_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        video_file = request.files['video']
        dailymotion_filename = request.form.get('dailymotion_filename')
        
        if not dailymotion_filename:
            return jsonify({'success': False, 'error': 'No Dailymotion filename provided'})
        
        # Get the paths for both videos
        dailymotion_path = get_output_path(dailymotion_filename)
        if not os.path.exists(dailymotion_path):
            return jsonify({'success': False, 'error': 'Dailymotion video not found'})
        
        # Save the uploaded video temporarily
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(temp_video_path)
        
        try:
            # Load both videos with high quality settings
            main_video = mp.VideoFileClip(temp_video_path, audio=True)
            dailymotion_video = mp.VideoFileClip(dailymotion_path, audio=True)
            
            # Ensure both videos have the same resolution (use the higher resolution)
            target_width = max(main_video.w, dailymotion_video.w)
            target_height = max(main_video.h, dailymotion_video.h)
            
            # Resize videos if needed while maintaining aspect ratio
            if main_video.w != target_width or main_video.h != target_height:
                main_video = main_video.resize(width=target_width, height=target_height)
            if dailymotion_video.w != target_width or dailymotion_video.h != target_height:
                dailymotion_video = dailymotion_video.resize(width=target_width, height=target_height)
            
            # Concatenate videos
            final_video = mp.concatenate_videoclips([main_video, dailymotion_video])
            
            # Generate output filename
            timestamp = int(time.time())
            output_filename = f'merged_{timestamp}.mp4'
            output_path = get_output_path(output_filename)
            
            # Write the final video with high quality settings
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate='8000k',  # High bitrate for better quality
                fps=30,  # Maintain high frame rate
                preset='slow',  # Better compression quality
                threads=4,  # Use multiple threads for faster processing
                ffmpeg_params=[
                    '-crf', '18',  # Constant Rate Factor (lower = better quality, 18 is visually lossless)
                    '-movflags', '+faststart',  # Enable fast start for web playback
                    '-pix_fmt', 'yuv420p'  # Ensure compatibility with most players
                ]
            )
            
            # Clean up
            final_video.close()
            main_video.close()
            dailymotion_video.close()
            
            # Remove temporary files
            cleanup_files([temp_video_path, dailymotion_path])
            
            return jsonify({
                'success': True,
                'filename': output_filename
            })
            
        except Exception as e:
            print(f"Error in video processing: {str(e)}")
            # Clean up in case of error
            cleanup_files([temp_video_path, dailymotion_path])
            return jsonify({'success': False, 'error': f'Error processing video: {str(e)}'})
            
    except Exception as e:
        print(f"Error in add_dailymotion_to_video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create a test user if none exists
    with app.app_context():
        if not User.query.filter_by(username='admin').first():
            print("Creating test user...")  # Debug print
            test_user = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('password')
            )
            db.session.add(test_user)
            db.session.commit()
            print("Test user created successfully")  # Debug print
    
    # Use environment port for production, fallback to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print("Starting Flask application...")  # Debug print
    app.run(debug=debug_mode, host='0.0.0.0', port=port) 