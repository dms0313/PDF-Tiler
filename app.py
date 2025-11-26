from flask import Flask, render_template_string, request, jsonify, send_file
import fitz  # PyMuPDF
from PIL import Image
import os
import json
from datetime import datetime
from pathlib import Path
import uuid
import io
import zipfile
from io import BytesIO
import numpy as np
import shutil

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'converted'
HISTORY_FILE = 'conversion_history.json'
DPI = 350  # 300-400 DPI range
TILE_SIZE = 1024  # 1024 tiles
OVERLAP = 80  # 80 pixel overlap (12.5% of tile size)
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
    """Quickly detect if a tile is blank, empty, or only contains borders.

    The heavy pixel analysis runs on a downscaled version of the image to
    significantly reduce the processing time while keeping the heuristics
    accurate enough for tile filtering.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Downscale for faster analysis while keeping structure information
    if max(image.size) > 256:
        working_image = image.resize((256, 256), Image.BILINEAR)
    else:
        working_image = image

    img_array = np.asarray(working_image, dtype=np.uint8)

    # Method 1: Check percentage of near-white pixels
    white_pixels = np.all(img_array >= 245, axis=2)
    white_percentage = white_pixels.mean()
    if white_percentage > threshold:
        return True

    # Convert to grayscale once for the remaining checks
    gray_array = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    # Method 2: Low variance implies blank/uniform tiles
    if gray_array.std() < 5:
        return True

    # Method 3: Simple edge detection using gradients
    edges_h = np.abs(np.diff(gray_array, axis=0))
    edges_v = np.abs(np.diff(gray_array, axis=1))
    significant_edges = np.count_nonzero(edges_h > 30) + np.count_nonzero(edges_v > 30)
    edge_density = significant_edges / (gray_array.shape[0] * gray_array.shape[1])
    if edge_density < 0.001:
        return True

    # Method 4: Detect border-only frames
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

def tile_image(image, tile_size=640, overlap=80):
    """
    Tile a large image into 640x640 windows with overlap.
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

@app.route('/preview_pdf', methods=['POST'])
def preview_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    try:
        preview_id = str(uuid.uuid4())
        preview_dir = os.path.join(OUTPUT_FOLDER, 'previews', preview_id)
        os.makedirs(preview_dir, exist_ok=True)

        pdf_path = os.path.join(preview_dir, 'original.pdf')
        file.save(pdf_path)

        # Generate low-res images for preview
        images = pdf_to_images(pdf_path, dpi=72)

        preview_files = []
        for i, image in enumerate(images):
            thumbnail = image.resize((100, 150), Image.LANCZOS)
            preview_filename = f"page_{i+1}.jpg"
            preview_path = os.path.join(preview_dir, preview_filename)
            thumbnail.save(preview_path, 'JPEG', quality=80)
            preview_files.append(preview_filename)

        return jsonify({
            'success': True,
            'preview_id': preview_id,
            'preview_files': preview_files,
            'page_count': len(images)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        preview_id = request.form.get('preview_id')
        selected_pages = json.loads(request.form.get('selected_pages', '[]'))

        if not preview_id or not selected_pages:
            return jsonify({'error': 'Missing preview ID or selected pages'}), 400

        original_pdf_path = os.path.join(OUTPUT_FOLDER, 'previews', preview_id, 'original.pdf')
        if not os.path.exists(original_pdf_path):
            return jsonify({'error': 'Original PDF not found'}), 400

        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        
        # Use the already uploaded PDF
        pdf_path = original_pdf_path
        
        # Convert PDF to images using PyMuPDF
        images = pdf_to_images(pdf_path, dpi=DPI)
        
        # Create output directory for this conversion
        output_dir = os.path.join(OUTPUT_FOLDER, conversion_id)
        os.makedirs(output_dir, exist_ok=True)
        
        converted_files = []
        total_tiles = 0
        blank_tiles_filtered = 0
        
        for i, image in enumerate(images):
            page_num = i + 1
            if page_num not in selected_pages:
                continue

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
                
                output_filename = f"page_{page_num}_tile_r{row}_c{col}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                tile_img.save(output_path, 'JPEG', quality=95, dpi=(DPI, DPI))
                
                converted_files.append({
                    'filename': output_filename,
                    'path': output_path,
                    'size': os.path.getsize(output_path),
                    'page': page_num,
                    'tile_row': row,
                    'tile_col': col
                })
                total_tiles += 1
        
        # Add to history
        history = load_history()
        history_entry = {
            'id': conversion_id,
            'original_filename': "Selected Pages",
            'timestamp': datetime.now().isoformat(),
            'page_count': len(selected_pages),
            'tile_count': total_tiles,
            'blank_filtered': blank_tiles_filtered,
            'files': converted_files
        }
        history.insert(0, history_entry)  # Most recent first
        save_history(history)
        
        # Clean up preview folder
        preview_dir = os.path.join(OUTPUT_FOLDER, 'previews', preview_id)
        if os.path.exists(preview_dir):
            shutil.rmtree(preview_dir)
        
        return jsonify({
            'success': True,
            'conversion_id': conversion_id,
            'page_count': len(selected_pages),
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

@app.route('/preview_image/<preview_id>/<filename>')
def preview_page_image(preview_id, filename):
    """Serve preview image"""
    file_path = os.path.join(OUTPUT_FOLDER, 'previews', preview_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    return jsonify({'error': 'File not found'}), 404

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
    <title>PDF to PNG Converter - Client-Side</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .badge {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
            margin-bottom: 30px;
        }

        .upload-area:hover {
            background: #f0f2ff;
            border-color: #764ba2;
        }
        
        .upload-area.dragover {
            background: #e8ebff;
            border-color: #764ba2;
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.2em;
            color: #333;
            margin-bottom: 10px;
        }
        
        .upload-hint {
            color: #666;
            font-size: 0.95em;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .options {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .option-group {
            display: flex;
            flex-direction: column;
        }
        
        label {
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            font-size: 0.95em;
        }
        
        select {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            background: white;
            cursor: pointer;
            transition: border-color 0.3s;
        }

        select:hover {
            border-color: #667eea;
        }

        select:focus {
            outline: none;
            border-color: #667eea;
        }

        .convert-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }

        .convert-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .convert-btn:active {
            transform: translateY(0);
        }
        
        .convert-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .file-info {
            background: #f0f2ff;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }
        
        .file-info.show {
            display: block;
        }

        .file-name {
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        
        .file-details {
            color: #666;
            font-size: 0.9em;
        }
        
        .progress {
            display: none;
            margin-bottom: 20px;
        }
        
        .progress.show {
            display: block;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.85em;
        }
        
        .results {
            display: none;
        }
        
        .results.show {
            display: block;
        }
        
        .result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f8f9ff;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        
        .result-info {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .result-preview {
            width: 60px;
            height: 60px;
            border-radius: 8px;
            object-fit: cover;
            border: 2px solid #e0e0e0;
        }
        
        .result-details {
            display: flex;
            flex-direction: column;
        }
        
        .result-name {
            font-weight: 600;
            color: #333;
            margin-bottom: 3px;
        }
        
        .result-size {
            color: #666;
            font-size: 0.85em;
        }
        
        .download-btn {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: #764ba2;
            transform: translateY(-2px);
        }
        
        .download-all-btn {
            width: 100%;
            padding: 16px;
            background: #10b981;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }
        
        .download-all-btn:hover {
            background: #059669;
            transform: translateY(-2px);
        }
        
        .message {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }

        .message.show {
            display: block;
        }

        .message.error {
            background: #fee;
            color: #c33;
            border: 1px solid #fcc;
        }

        .message.success {
            background: #efe;
            color: #3c3;
            border: 1px solid #cfc;
        }

        .footer {
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
        }

        .footer a {
            color: #667eea;
            text-decoration: none;
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 25px;
            }
            h1 {
                font-size: 2em;
            }
            .options {
                grid-template-columns: 1fr;
            }
            .upload-area {
                padding: 40px 15px;
            }
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }
        
        .preview-section {
            margin-top: 30px;
        }

        .preview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .preview-header h3 {
            color: #333;
        }

        .select-all-container {
            display: flex;
            align-items: center;
        }
        
        .preview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 15px;
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            background: #f8f9ff;
            border-radius: 10px;
        }

        .preview-item {
            position: relative;
            cursor: pointer;
        }
        
        .preview-item img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            transition: border-color 0.3s;
        }

        .preview-item .page-checkbox {
            position: absolute;
            top: 8px;
            right: 8px;
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        
        .preview-item .page-number {
            position: absolute;
            bottom: 8px;
            left: 8px;
            background: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8em;
        }

        .preview-item.selected img {
            border-color: #667eea;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>📄 PDF to PNG<span class="badge">No Upload!</span></h1>
        <p class="subtitle">Convert PDFs to images right in your browser - 100% client-side, no server needed!</p>

        <div class="message" id="message"></div>

        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📁</div>
            <div class="upload-text">Drop your PDF here or click to browse</div>
            <div class="upload-hint">Maximum file size: 50 MB</div>
            <input type="file" id="fileInput" accept=".pdf">
        </div>

        <div class="file-info" id="fileInfo">
            <div class="file-name" id="fileName"></div>
            <div class="file-details" id="fileDetails"></div>
        </div>

        <div class="options">
            <div class="option-group">
                <label for="dpiSelect">Quality (DPI)</label>
                <select id="dpiSelect">
                    <option value="1">Low (72 DPI) - Fastest</option>
                    <option value="1.5">Medium (108 DPI)</option>
                    <option value="2" selected>High (144 DPI)</option>
                    <option value="3">Very High (216 DPI)</option>
                    <option value="4">Maximum (288 DPI) - Slowest</option>
                </select>
            </div>

            <div class="option-group">
                <label for="formatSelect">Output Format</label>
                <select id="formatSelect">
                    <option value="png" selected>PNG (Best Quality)</option>
                    <option value="jpeg">JPEG (Smaller Size)</option>
                </select>
            </div>
        </div>

        <button class="convert-btn" id="convertBtn" disabled>
            Preview Pages
        </button>

        <div class="preview-section" id="previewSection" style="display: none;">
            <div class="preview-header">
                <h3>Select pages to tile</h3>
                <div class="select-all-container">
                    <input type="checkbox" id="selectAllCheckbox" checked>
                    <label for="selectAllCheckbox">Select All</label>
                </div>
            </div>
            <div class="preview-grid" id="previewGrid"></div>
        </div>

        <button class="convert-btn" id="tileBtn" style="display: none;">
            Tile Selected Pages
        </button>

        <div class="progress" id="progress">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill">0%</div>
            </div>
        </div>

        <div class="results" id="results">
            <h3 style="margin-bottom: 15px; color: #333;">Converted Images:</h3>
            <div id="resultsList"></div>
            <button class="download-all-btn" id="downloadAllBtn">
                📦 Download All as ZIP
            </button>
        </div>

        <div class="footer">
            Made with ❤️ using <a href="https://mozilla.github.io/pdf.js/" target="_blank">PDF.js</a>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <script>
        // Configure PDF.js worker
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileDetails = document.getElementById('fileDetails');
        const convertBtn = document.getElementById('convertBtn');
        const tileBtn = document.getElementById('tileBtn');
        const previewSection = document.getElementById('previewSection');
        const previewGrid = document.getElementById('previewGrid');
        const selectAllCheckbox = document.getElementById('selectAllCheckbox');
        const progress = document.getElementById('progress');
        const progressFill = document.getElementById('progressFill');
        const results = document.getElementById('results');
        const resultsList = document.getElementById('resultsList');
        const downloadAllBtn = document.getElementById('downloadAllBtn');
        const message = document.getElementById('message');

        let currentPDF = null;
        let convertedImages = [];

        // Upload area click
        uploadArea.addEventListener('click', () => fileInput.click());

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type === 'application/pdf') {
                handleFile(file);
            } else {
                showMessage('Please drop a PDF file', 'error');
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleFile(file);
            }
        });

        // Handle file
        async function handleFile(file) {
            if (file.size > 50 * 1024 * 1024) {
                showMessage('File too large! Maximum size is 50 MB', 'error');
                return;
            }

            currentPDF = file;
            fileName.textContent = file.name;

            const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
            fileDetails.textContent = `Size: ${fileSizeMB} MB`;

            fileInfo.classList.add('show');
            convertBtn.disabled = false;
            convertBtn.textContent = 'Preview Pages';
            results.classList.remove('show');
            previewSection.style.display = 'none';
            tileBtn.style.display = 'none';
            hideMessage();

            // Get page count
            try {
                const arrayBuffer = await file.arrayBuffer();
                const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
                fileDetails.textContent = `Size: ${fileSizeMB} MB • Pages: ${pdf.numPages}`;
            } catch (error) {
                console.error('Error reading PDF:', error);
            }
        }

        // Convert button
        convertBtn.addEventListener('click', () => {
            if (currentPDF) {
                previewPDF(currentPDF);
            }
        });

        async function previewPDF(file) {
            convertBtn.disabled = true;
            convertBtn.innerHTML = '<span class="loading-spinner"></span> Generating previews...';
            hideMessage();

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/preview_pdf', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                displayPreviews(data.preview_id, data.preview_files);
                convertBtn.style.display = 'none';
                tileBtn.style.display = 'block';

            } catch (error) {
                showMessage('Error generating previews: ' + error.message, 'error');
                convertBtn.disabled = false;
                convertBtn.textContent = 'Preview Pages';
            }
        }

        function updateProgress(current, total) {
            const percent = Math.round((current / total) * 100);
            progressFill.style.width = percent + '%';
            progressFill.textContent = `${percent}% (${current}/${total})`;
        }

        function addResultItem(image, pageNum) {
            const item = document.createElement('div');
            item.className = 'result-item';

            item.innerHTML = `
                <div class="result-info">
                    <img src="${image.url}" class="result-preview" alt="Page ${pageNum}">
                    <div class="result-details">
                        <div class="result-name">${image.name}</div>
                        <div class="result-size">${image.size} KB</div>
                    </div>
                </div>
                <button class="download-btn" onclick="downloadImage(${convertedImages.length - 1})">
                    Download
                </button>
            `;

            resultsList.appendChild(item);
        }

        function downloadImage(index) {
            const image = convertedImages[index];
            const link = document.createElement('a');
            link.href = image.url;
            link.download = image.name;
            link.click();
        }

        // Download all as ZIP
        downloadAllBtn.addEventListener('click', async() => {
            if (convertedImages.length === 0) return;

            downloadAllBtn.innerHTML = '<span class="loading-spinner"></span> Creating ZIP...';
            downloadAllBtn.disabled = true;

            try {
                const zip = new JSZip();

                convertedImages.forEach(image => {
                    zip.file(image.name, image.blob);
                });

                const zipBlob = await zip.generateAsync({
                    type: 'blob'
                });
                const zipUrl = URL.createObjectURL(zipBlob);

                const baseName = currentPDF.name.replace('.pdf', '');
                const link = document.createElement('a');
                link.href = zipUrl;
                link.download = `${baseName}_converted.zip`;
                link.click();

                downloadAllBtn.innerHTML = '📦 Download All as ZIP';
                downloadAllBtn.disabled = false;
                showMessage('ZIP file downloaded successfully!', 'success');

            } catch (error) {
                console.error('ZIP creation error:', error);
                showMessage('Error creating ZIP file', 'error');
                downloadAllBtn.innerHTML = '📦 Download All as ZIP';
                downloadAllBtn.disabled = false;
            }
        });

        function showMessage(text, type) {
            message.textContent = text;
            message.className = 'message show ' + type;
        }

        function hideMessage() {
            message.classList.remove('show');
        }

        // Make downloadImage available globally
        window.downloadImage = downloadImage;

        async function tilePages(previewId, selectedPages) {
            tileBtn.disabled = true;
            tileBtn.innerHTML = '<span class="loading-spinner"></span> Tiling...';
            hideMessage();
            progress.classList.add('show');
            updateProgress(0, 1);

            const formData = new FormData();
            formData.append('preview_id', previewId);
            formData.append('selected_pages', JSON.stringify(selectedPages));

            // Simulate progress
            let simulatedProgress = 0;
            const progressInterval = setInterval(() => {
                simulatedProgress = Math.min(simulatedProgress + 5, 95);
                updateProgress(simulatedProgress, 100);
            }, 200);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                clearInterval(progressInterval);

                if (data.error) {
                    throw new Error(data.error);
                }

                updateProgress(100, 100);
                showMessage(`Successfully tiled ${data.tile_count} images!`, 'success');
                loadHistory(); // Assuming loadHistory is defined
                resetUI();

            } catch (error) {
                clearInterval(progressInterval);
                showMessage('Error tiling pages: ' + error.message, 'error');
                tileBtn.disabled = false;
                tileBtn.textContent = 'Tile Selected Pages';
                progress.classList.remove('show');
            }
        }

        function resetUI() {
            fileInfo.classList.remove('show');
            previewSection.style.display = 'none';
            tileBtn.style.display = 'none';
            convertBtn.style.display = 'block';
            convertBtn.disabled = true;
            convertBtn.textContent = 'Select a PDF file first';
            progress.classList.remove('show');
            fileInput.value = '';
            currentPDF = null;
        }

        let currentPreviewId = null;

        function displayPreviews(previewId, files) {
            currentPreviewId = previewId;
            previewGrid.innerHTML = '';
            files.forEach((file, index) => {
                const pageNum = index + 1;
                const item = document.createElement('div');
                item.className = 'preview-item selected';
                item.dataset.pageNum = pageNum;
                item.innerHTML = `
                    <img src="/preview_image/${previewId}/${file}" alt="Page ${pageNum}">
                    <input type="checkbox" class="page-checkbox" checked>
                    <div class="page-number">${pageNum}</div>
                `;
                previewGrid.appendChild(item);
            });

            previewSection.style.display = 'block';
            selectAllCheckbox.checked = true;
        }

        previewGrid.addEventListener('click', (e) => {
            const item = e.target.closest('.preview-item');
            if (item) {
                const checkbox = item.querySelector('.page-checkbox');
                checkbox.checked = !checkbox.checked;
                item.classList.toggle('selected');
            }
        });

        selectAllCheckbox.addEventListener('change', (e) => {
            const isChecked = e.target.checked;
            const checkboxes = previewGrid.querySelectorAll('.page-checkbox');
            checkboxes.forEach(checkbox => {
                checkbox.checked = isChecked;
                checkbox.closest('.preview-item').classList.toggle('selected', isChecked);
            });
        });

        tileBtn.addEventListener('click', () => {
            const selectedPages = [];
            const checkboxes = previewGrid.querySelectorAll('.page-checkbox:checked');
            checkboxes.forEach(cb => {
                selectedPages.push(parseInt(cb.closest('.preview-item').dataset.pageNum));
            });

            if (selectedPages.length === 0) {
                showMessage('Please select at least one page to tile.', 'error');
                return;
            }

            tilePages(currentPreviewId, selectedPages);
        });

    </script>
</body>

</html>
'''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
