from flask import Flask, render_template_string, request, jsonify, send_file
import fitz  # PyMuPDF
from PIL import Image, ImageStat
import os
import json
from datetime import datetime
from pathlib import Path
import uuid
import io
import zipfile
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'converted'
HISTORY_FILE = 'conversion_history.json'
DPI = 350  # 300-400 DPI range
TILE_SIZE = 1024  # 1024x1024 tiles
OVERLAP = 128  # 128 pixel overlap (12.5% of tile size)
BLANK_THRESHOLD = 0.98  # Consider tile blank if 98% or more is white/uniform

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def is_blank_tile(image, threshold=0.98):
    """
    Detect if a tile is blank/empty or just contains border frames.
    Returns True if the tile is considered blank or is just a border.
    
    Uses multiple methods to detect blank tiles:
    1. Check if most pixels are near white
    2. Check standard deviation (low variance = uniform/blank)
    3. Check edge detection (no edges = blank)
    4. Detect border frames (edges only at perimeter)
    """
    # Convert to numpy array for faster processing
    img_array = np.array(image)
    
    # Method 1: Check percentage of near-white pixels
    # Consider pixels with RGB values > 240 as "white"
    white_pixels = np.all(img_array > 240, axis=2)
    white_percentage = np.sum(white_pixels) / (image.width * image.height)
    
    if white_percentage > threshold:
        return True
    
    # Method 2: Check standard deviation (variance)
    # Low std dev means uniform color (likely blank)
    std_dev = np.std(img_array)
    if std_dev < 5:  # Very low variation
        return True
    
    # Method 3: Check for content using edge detection
    # Convert to grayscale for edge detection
    gray = image.convert('L')
    gray_array = np.array(gray)
    
    # Simple edge detection using gradient
    edges_h = np.abs(np.diff(gray_array, axis=0))
    edges_v = np.abs(np.diff(gray_array, axis=1))
    
    # Count significant edges (difference > 30)
    significant_edges = (np.sum(edges_h > 30) + np.sum(edges_v > 30))
    edge_density = significant_edges / (image.width * image.height)
    
    # If very few edges, likely blank
    if edge_density < 0.001:  # Less than 0.1% of pixels have edges
        return True
    
    # Method 4: Detect border frames
    # Check if edges are concentrated only at the perimeter (border frame detection)
    if is_border_frame(gray_array, edges_h, edges_v):
        return True
    
    return False

def is_border_frame(gray_array, edges_h, edges_v):
    """
    Detect if the tile only contains a border frame with no meaningful content inside.
    Returns True if it's just a border frame.
    """
    height, width = gray_array.shape
    
    # Define border region (outer 10% of the image on each side)
    border_thickness = max(int(min(height, width) * 0.1), 10)
    
    # Create masks for border and center regions
    border_mask_h = np.zeros_like(edges_h, dtype=bool)
    border_mask_v = np.zeros_like(edges_v, dtype=bool)
    center_mask_h = np.zeros_like(edges_h, dtype=bool)
    center_mask_v = np.zeros_like(edges_v, dtype=bool)
    
    # Define border regions (top, bottom, left, right)
    # For horizontal edges
    border_mask_h[:border_thickness, :] = True  # Top
    border_mask_h[-border_thickness:, :] = True  # Bottom
    border_mask_h[:, :border_thickness] = True  # Left
    border_mask_h[:, -border_thickness:] = True  # Right
    
    # For vertical edges
    border_mask_v[:border_thickness, :] = True  # Top
    border_mask_v[-border_thickness:, :] = True  # Bottom
    border_mask_v[:, :border_thickness] = True  # Left
    border_mask_v[:, -border_thickness:] = True  # Right
    
    # Center is everything that's not border
    center_mask_h = ~border_mask_h
    center_mask_v = ~border_mask_v
    
    # Count edges in border vs center
    border_edges = np.sum((edges_h > 30) & border_mask_h) + np.sum((edges_v > 30) & border_mask_v)
    center_edges = np.sum((edges_h > 30) & center_mask_h) + np.sum((edges_v > 30) & center_mask_v)
    
    # If there are edges but they're almost all in the border region, it's a frame
    total_edges = border_edges + center_edges
    
    if total_edges > 0:
        border_ratio = border_edges / total_edges
        # If more than 80% of edges are in border and center has very few edges
        if border_ratio > 0.8 and center_edges < (width * height * 0.002):
            return True
    
    # Additional check: look for rectangular border pattern
    # Check if the edges form a rectangle at the perimeter
    if detect_rectangular_border(gray_array, border_thickness):
        return True
    
    return False

def detect_rectangular_border(gray_array, border_thickness):
    """
    Detect if there's a rectangular border/frame pattern.
    Returns True if a rectangular border is detected with minimal interior content.
    """
    height, width = gray_array.shape
    
    # Extract the border strips
    top_strip = gray_array[:border_thickness, :]
    bottom_strip = gray_array[-border_thickness:, :]
    left_strip = gray_array[:, :border_thickness]
    right_strip = gray_array[:, -border_thickness:]
    
    # Extract the center region
    center = gray_array[border_thickness:-border_thickness, border_thickness:-border_thickness]
    
    if center.size == 0:
        return False
    
    # Calculate variance in border vs center
    border_std = np.mean([
        np.std(top_strip),
        np.std(bottom_strip),
        np.std(left_strip),
        np.std(right_strip)
    ])
    
    center_std = np.std(center)
    
    # If border has variation (edges/lines) but center is uniform, it's likely a frame
    if border_std > 15 and center_std < 8:
        # Also check if center is mostly white
        center_mean = np.mean(center)
        if center_mean > 240:  # Center is white/blank
            return True
    
    return False

def tile_image(image, tile_size=1024, overlap=128):
    """
    Tile a large image into 1024x1024 windows with overlap.
    Returns list of tuples: (tile_image, row, col)
    """
    width, height = image.size
    tiles = []
    
    # Calculate step size (tile_size - overlap)
    step = tile_size - overlap
    
    # Calculate number of tiles needed
    cols = max(1, (width - overlap + step - 1) // step)
    rows = max(1, (height - overlap + step - 1) // step)
    
    for row in range(rows):
        for col in range(cols):
            # Calculate tile boundaries
            left = col * step
            top = row * step
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            
            # Adjust if we're at the edge and tile would be too small
            if right - left < tile_size and col > 0:
                left = max(0, right - tile_size)
            if bottom - top < tile_size and row > 0:
                top = max(0, bottom - tile_size)
            
            # Crop the tile
            tile = image.crop((left, top, right, bottom))
            
            # If tile is smaller than tile_size (edge case), pad it
            if tile.size != (tile_size, tile_size):
                padded = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
                padded.paste(tile, (0, 0))
                tile = padded
            
            tiles.append((tile, row, col))
    
    return tiles

def optimize_for_roboflow(image):
    """Optimize image for Roboflow training"""
    # Convert to RGB if necessary (removes alpha channel)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def pdf_to_images(pdf_path, dpi=300):
    """Convert PDF to images using PyMuPDF"""
    images = []
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    # Calculate zoom factor for desired DPI
    # PyMuPDF default is 72 DPI, so zoom = desired_dpi / 72
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    # Convert each page to image
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Render page to pixmap at specified DPI
        pix = page.get_pixmap(matrix=mat)
        
        # Convert pixmap to PIL Image
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        images.append(image)
    
    pdf_document.close()
    return images

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        
        # Save uploaded PDF
        pdf_path = os.path.join(UPLOAD_FOLDER, f"{conversion_id}.pdf")
        file.save(pdf_path)
        
        # Convert PDF to images using PyMuPDF
        images = pdf_to_images(pdf_path, dpi=DPI)
        
        # Create output directory for this conversion
        output_dir = os.path.join(OUTPUT_FOLDER, conversion_id)
        os.makedirs(output_dir, exist_ok=True)
        
        converted_files = []
        total_tiles = 0
        blank_tiles_filtered = 0
        
        for i, image in enumerate(images):
            # Optimize image for Roboflow
            optimized_image = optimize_for_roboflow(image)
            
            # Tile the image
            tiles = tile_image(optimized_image, tile_size=TILE_SIZE, overlap=OVERLAP)
            
            # Save each tile (skip blank tiles)
            for tile_img, row, col in tiles:
                # Check if tile is blank
                if is_blank_tile(tile_img, threshold=BLANK_THRESHOLD):
                    blank_tiles_filtered += 1
                    continue  # Skip this tile
                
                output_filename = f"page_{i+1}_tile_r{row}_c{col}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                tile_img.save(output_path, 'JPEG', quality=95, dpi=(DPI, DPI))
                
                converted_files.append({
                    'filename': output_filename,
                    'path': output_path,
                    'size': os.path.getsize(output_path),
                    'page': i + 1,
                    'tile_row': row,
                    'tile_col': col
                })
                total_tiles += 1
        
        # Add to history
        history = load_history()
        history_entry = {
            'id': conversion_id,
            'original_filename': file.filename,
            'timestamp': datetime.now().isoformat(),
            'page_count': len(images),
            'tile_count': total_tiles,
            'blank_filtered': blank_tiles_filtered,
            'files': converted_files
        }
        history.insert(0, history_entry)  # Most recent first
        save_history(history)
        
        # Clean up uploaded PDF
        os.remove(pdf_path)
        
        return jsonify({
            'success': True,
            'conversion_id': conversion_id,
            'page_count': len(images),
            'tile_count': total_tiles,
            'blank_filtered': blank_tiles_filtered,
            'files': converted_files
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    history = load_history()
    return jsonify(history)

@app.route('/preview/<conversion_id>/<filename>')
def preview_image(conversion_id, filename):
    """Serve image for preview (not as attachment)"""
    file_path = os.path.join(OUTPUT_FOLDER, conversion_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    return jsonify({'error': 'File not found'}), 404

@app.route('/download/<conversion_id>/<filename>')
def download_file(conversion_id, filename):
    file_path = os.path.join(OUTPUT_FOLDER, conversion_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/download_zip/<conversion_id>')
def download_zip(conversion_id):
    """Download all files from a conversion as a ZIP"""
    conversion_dir = os.path.join(OUTPUT_FOLDER, conversion_id)
    
    if not os.path.exists(conversion_dir):
        return jsonify({'error': 'Conversion not found'}), 404
    
    # Get the original filename from history
    history = load_history()
    original_filename = None
    for item in history:
        if item['id'] == conversion_id:
            original_filename = item['original_filename']
            break
    
    # Create zip filename
    if original_filename:
        zip_name = f"{os.path.splitext(original_filename)[0]}_tiles.zip"
    else:
        zip_name = f"conversion_{conversion_id}.zip"
    
    # Create ZIP file in memory
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add all files from the conversion directory
        for filename in os.listdir(conversion_dir):
            file_path = os.path.join(conversion_dir, filename)
            if os.path.isfile(file_path):
                zf.write(file_path, filename)
    
    memory_file.seek(0)
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_name
    )

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF to Image Converter - Roboflow Training</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        .spec-info {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        
        .spec-info h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .spec-info ul {
            list-style: none;
            color: #666;
        }
        
        .spec-info li {
            margin-bottom: 5px;
            padding-left: 20px;
            position: relative;
        }
        
        .spec-info li:before {
            content: "✓";
            position: absolute;
            left: 0;
            color: #667eea;
            font-weight: bold;
        }
        
        .no-admin-badge {
            background: #4caf50;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            display: inline-block;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .drop-zone {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 60px 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            background: #f8f9ff;
        }
        
        .drop-zone.drag-over {
            background: #e8ebff;
            border-color: #764ba2;
            transform: scale(1.02);
        }
        
        .drop-zone-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .drop-zone-text {
            font-size: 1.2em;
            color: #667eea;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .drop-zone-subtext {
            color: #666;
            font-size: 0.9em;
        }
        
        #fileInput {
            display: none;
        }
        
        .progress-container {
            display: none;
            margin-top: 30px;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .status-text {
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        
        .success-message {
            background: #4caf50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
            display: none;
        }
        
        .error-message {
            background: #f44336;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
            display: none;
        }
        
        .history-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .history-item {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        
        .history-item:hover {
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .history-title {
            font-weight: 600;
            color: #333;
            font-size: 1.1em;
            flex: 1;
        }
        
        .history-header-right {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .history-date {
            color: #999;
            font-size: 0.85em;
        }
        
        .history-info {
            color: #666;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        
        .file-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }
        
        .file-item {
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9em;
            position: relative;
            cursor: pointer;
        }
        
        .file-item:hover {
            background: #e8e8e8;
        }
        
        .hover-preview {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            margin-bottom: 10px;
            display: none;
            z-index: 1000;
            pointer-events: none;
        }
        
        .file-item:hover .hover-preview {
            display: block;
        }
        
        .hover-preview img {
            max-width: 300px;
            max-height: 300px;
            border: 3px solid #667eea;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            background: white;
        }
        
        .hover-preview-arrow {
            width: 0;
            height: 0;
            border-left: 10px solid transparent;
            border-right: 10px solid transparent;
            border-top: 10px solid #667eea;
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
        }
        
        .download-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            font-size: 0.85em;
            transition: background 0.3s ease;
        }
        
        .download-btn:hover {
            background: #764ba2;
        }
        
        .download-all-btn {
            background: #4caf50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            font-size: 0.9em;
            transition: background 0.3s ease;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        
        .download-all-btn:hover {
            background: #45a049;
        }
        
        .collapse-toggle {
            background: none;
            border: none;
            color: #667eea;
            cursor: pointer;
            font-size: 1.3em;
            padding: 5px;
            margin-right: 10px;
            transition: transform 0.3s ease;
            font-weight: bold;
        }
        
        .collapse-toggle.collapsed {
            transform: rotate(-90deg);
        }
        
        .collapsible-content {
            max-height: 2000px;
            overflow: hidden;
            transition: max-height 0.3s ease, opacity 0.3s ease;
            opacity: 1;
        }
        
        .collapsible-content.collapsed {
            max-height: 0;
            opacity: 0;
        }
        
        .image-preview-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 10000;
            align-items: center;
            justify-content: center;
        }
        
        .image-preview-modal.active {
            display: flex;
        }
        
        .preview-content {
            position: relative;
            max-width: 90%;
            max-height: 90%;
        }
        
        .preview-content img {
            max-width: 100%;
            max-height: 85vh;
            border-radius: 8px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }
        
        .preview-close {
            position: absolute;
            top: -40px;
            right: 0;
            background: white;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 24px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.3s ease;
        }
        
        .preview-close:hover {
            background: #f0f0f0;
        }
        
        .preview-filename {
            text-align: center;
            color: white;
            margin-top: 15px;
            font-size: 0.9em;
        }
        
        .preview-icon {
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖼️ PDF to Image Converter</h1>
        <p class="subtitle">Convert PDFs to Roboflow-ready training images at 350 DPI</p>
        
        <div class="card">
            <div class="no-admin-badge">✓ No Admin Rights Required - Uses Pure Python</div>
            
            <div class="spec-info">
                <h3>Conversion Specifications</h3>
                <ul>
                    <li>Resolution: 350 DPI (300-400 DPI range)</li>
                    <li>Format: JPEG (optimized for Roboflow)</li>
                    <li>Tile size: 1024×1024 pixels</li>
                    <li>Overlap: 128 pixels between tiles (12.5%)</li>
                    <li>Auto-filter: Blank tiles automatically removed</li>
                    <li>Color space: RGB (alpha channels removed)</li>
                    <li>Powered by PyMuPDF - No external dependencies needed</li>
                </ul>
            </div>
            
            <div class="drop-zone" id="dropZone">
                <div class="drop-zone-icon">📄</div>
                <div class="drop-zone-text">Drag & Drop PDF File Here</div>
                <div class="drop-zone-subtext">or click to browse</div>
            </div>
            
            <input type="file" id="fileInput" accept=".pdf">
            
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill">0%</div>
                </div>
                <div class="status-text" id="statusText">Processing...</div>
            </div>
            
            <div class="success-message" id="successMessage"></div>
            <div class="error-message" id="errorMessage"></div>
        </div>
        
        <div class="card history-section">
            <h2>📋 Conversion History</h2>
            <div id="historyList">
                <p style="text-align: center; color: #999;">No conversions yet</p>
            </div>
        </div>
    </div>
    
    <div class="image-preview-modal" id="previewModal">
        <div class="preview-content">
            <button class="preview-close" id="closePreview">×</button>
            <img id="previewImage" src="" alt="Preview">
            <div class="preview-filename" id="previewFilename"></div>
        </div>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const statusText = document.getElementById('statusText');
        const successMessage = document.getElementById('successMessage');
        const errorMessage = document.getElementById('errorMessage');
        const historyList = document.getElementById('historyList');
        const previewModal = document.getElementById('previewModal');
        const previewImage = document.getElementById('previewImage');
        const previewFilename = document.getElementById('previewFilename');
        const closePreview = document.getElementById('closePreview');
        
        loadHistory();
        
        closePreview.addEventListener('click', () => {
            previewModal.classList.remove('active');
        });
        
        previewModal.addEventListener('click', (e) => {
            if (e.target === previewModal) {
                previewModal.classList.remove('active');
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && previewModal.classList.contains('active')) {
                previewModal.classList.remove('active');
            }
        });
        
        function showPreview(imageSrc, filename) {
            previewImage.src = imageSrc;
            previewFilename.textContent = filename;
            previewModal.classList.add('active');
        }
        
        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                showError('Please select a PDF file');
                return;
            }
            
            uploadFile(file);
        }
        
        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            successMessage.style.display = 'none';
            errorMessage.style.display = 'none';
            progressContainer.style.display = 'block';
            progressFill.style.width = '0%';
            progressFill.textContent = '0%';
            statusText.textContent = 'Uploading PDF...';
            
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 5;
                if (progress <= 90) {
                    progressFill.style.width = progress + '%';
                    progressFill.textContent = progress + '%';
                }
            }, 200);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                clearInterval(progressInterval);
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                progressFill.style.width = '100%';
                progressFill.textContent = '100%';
                statusText.textContent = 'Conversion complete!';
                
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    let message = `Successfully converted ${data.page_count} page(s) into ${data.tile_count} tiles!`;
                    if (data.blank_filtered > 0) {
                        message += ` (${data.blank_filtered} blank tiles filtered)`;
                    }
                    showSuccess(message);
                    loadHistory();
                    fileInput.value = '';
                }, 1000);
            })
            .catch(error => {
                clearInterval(progressInterval);
                progressContainer.style.display = 'none';
                showError('Error: ' + error.message);
                fileInput.value = '';
            });
        }
        
        function loadHistory() {
            fetch('/history')
            .then(response => response.json())
            .then(history => {
                if (history.length === 0) {
                    historyList.innerHTML = '<p style="text-align: center; color: #999;">No conversions yet</p>';
                    return;
                }
                
                historyList.innerHTML = history.map(item => {
                    const date = new Date(item.timestamp);
                    const dateStr = date.toLocaleString();
                    
                    const filesHtml = item.files.map(file => `
                        <div class="file-item">
                            <div class="hover-preview">
                                <img src="/preview/${item.id}/${file.filename}" alt="${file.filename}">
                                <div class="hover-preview-arrow"></div>
                            </div>
                            <span onclick="showPreview('/preview/${item.id}/${file.filename}', '${file.filename}')">
                                <span class="preview-icon">👁️</span>${file.filename}
                            </span>
                            <a href="/download/${item.id}/${file.filename}" class="download-btn" onclick="event.stopPropagation()">Download</a>
                        </div>
                    `).join('');
                    
                    let infoText = `${item.page_count} page(s) • ${item.tile_count} tiles • 1024×1024px • 350 DPI`;
                    if (item.blank_filtered && item.blank_filtered > 0) {
                        infoText += ` • ${item.blank_filtered} blank tiles filtered`;
                    }
                    
                    return `
                        <div class="history-item">
                            <div class="history-header">
                                <button class="collapse-toggle collapsed" onclick="toggleCollapse(this)">▼</button>
                                <div class="history-title">${item.original_filename}</div>
                                <div class="history-header-right">
                                    <div class="history-date">${dateStr}</div>
                                    <a href="/download_zip/${item.id}" class="download-all-btn">
                                        📦 Download All (ZIP)
                                    </a>
                                </div>
                            </div>
                            <div class="collapsible-content collapsed">
                                <div class="history-info">
                                    ${infoText}
                                </div>
                                <div class="file-list">
                                    ${filesHtml}
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');
            })
            .catch(error => {
                console.error('Error loading history:', error);
            });
        }
        
        function toggleCollapse(button) {
            const content = button.parentElement.nextElementSibling;
            button.classList.toggle('collapsed');
            content.classList.toggle('collapsed');
        }
        
        function showSuccess(message) {
            successMessage.textContent = message;
            successMessage.style.display = 'block';
            setTimeout(() => {
                successMessage.style.display = 'none';
            }, 5000);
        }
        
        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            setTimeout(() => {
                errorMessage.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)