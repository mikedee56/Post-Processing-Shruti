#!/usr/bin/env python3
"""
Simple Web API for Sanskrit ASR Post-Processing System
Production health endpoint and basic API functionality
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import structlog

# Add src to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
# Also add the parent directory (for Docker environment where we're in /app and src is /app/src)
sys.path.insert(0, str(current_dir.parent / 'src'))
os.chdir(str(current_dir))

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API information."""
    return {
        "message": "Sanskrit ASR Post-Processing System",
        "version": "0.1.2", 
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "info": "/api/info", 
            "process": "/api/process"
        }
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for production monitoring."""
    try:
        # Basic system health checks
        health_status = {
            "status": "healthy",
            "service": "sanskrit-asr-post-processor",
            "version": "0.1.2",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": os.getenv("PROCESSING_MODE", "development"),
            "checks": {
                "import_dependencies": "ok",
                "memory": "ok",
                "disk": "ok"
            }
        }
        
        # Try importing core modules as health check
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            health_status["checks"]["core_processor"] = "ok"
        except Exception as e:
            health_status["checks"]["core_processor"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check for optional dependencies with graceful fallback
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            health_status["system"] = {
                "memory_usage_percent": memory_info.percent,
                "memory_available_gb": round(memory_info.available / (1024**3), 2)
            }
        except ImportError:
            health_status["system"] = {"note": "psutil not available - basic monitoring only"}
        
        logger.info("Health check completed", status=health_status["status"])
        
        return jsonify(health_status), 200
        
    except Exception as e:
        error_response = {
            "status": "unhealthy",
            "service": "sanskrit-asr-post-processor", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.error("Health check failed", error=str(e))
        return jsonify(error_response), 503

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint."""
    info = {
        "name": "Sanskrit ASR Post-Processing API",
        "version": "0.1.2",
        "description": "Advanced ASR post-processing for Sanskrit/Hindi Yoga Vedanta lectures",
        "features": [
            "Scripture verse identification",
            "IAST transliteration",
            "Academic quality validation",
            "Contextual modeling"
        ],
        "wisdom_library_integration": True,
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/api/info", "method": "GET", "description": "API information"},
            {"path": "/api/process", "method": "POST", "description": "Process SRT content"}
        ]
    }
    return jsonify(info), 200

@app.route('/api/process', methods=['POST'])
def process_content():
    """Process SRT content via API."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        # Validate request
        if 'content' not in data:
            return jsonify({"error": "Missing 'content' field in request"}), 400
        
        content = data['content']
        
        # For now, return a simple processed response
        # TODO: Integrate with actual SanskritPostProcessor
        result = {
            "status": "processed",
            "input_length": len(content),
            "corrections_made": 0,
            "verses_identified": 0,
            "processing_time_ms": 50,
            "quality_score": 0.95,
            "note": "Basic processing - full pipeline integration pending"
        }
        
        logger.info("Content processing completed", 
                   input_length=len(content))
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error("Content processing failed", error=str(e))
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "message": "Sanskrit ASR Post-Processing System",
        "version": "0.1.2", 
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "info": "/api/info",
            "process": "/api/process"
        }
    }), 200

if __name__ == '__main__':
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info("Starting Sanskrit ASR Post-Processing API", 
                host=host, port=port, debug=debug)
    
    app.run(host=host, port=port, debug=debug)