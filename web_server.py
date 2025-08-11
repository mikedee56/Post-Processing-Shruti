import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import http.server
import socketserver
import cgi
from pathlib import Path
from src.post_processors.sanskrit_post_processor import SanskritPostProcessor

PORT = 8000
PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "raw_srts")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_srts")
WEB_DIR = os.path.join(BASE_DIR, "web")
PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "raw_srts")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_srts")
WEB_DIR = os.path.join(BASE_DIR, "web")

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        
        # Serve files from PROCESSED_DIR if the path starts with it
        processed_dir_for_path = PROCESSED_DIR.replace('\\', '/')
        if self.path.startswith('/' + processed_dir_for_path):
            # We need to construct the full path to the file on the local filesystem
            # The path in the URL is relative to the root, so we join it with the current working directory
            fs_path = os.path.join(os.getcwd(), self.path.lstrip('/'))
            if os.path.exists(fs_path):
                # Use the parent class to handle serving the file
                # We need to temporarily change the directory for SimpleHTTPRequestHandler to work correctly
                # This is a bit of a hack, but it's the simplest way with the built-in server
                # We'll serve from the root directory to make the path calculation straightforward
                super().__init__(*self.args, directory=os.getcwd(), **self.kwargs)
                self.path = self.path # The path is already correct
                return http.server.SimpleHTTPRequestHandler.do_GET(self)
            else:
                self.send_error(404, "File not found")
                return

        super().do_GET()


    def do_POST(self):
        if self.path == '/upload':
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']}
            )

            if 'srt_file' not in form:
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'No file uploaded. Please select a file and try again.')
                return

            file_item = form['srt_file']

            if file_item.filename:
                # Sanitize filename to prevent directory traversal attacks
                filename = os.path.basename(file_item.filename)
                
                # Ensure the filename is not empty after sanitization
                if not filename:
                    self.send_response(400)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b'Invalid filename.')
                    return

                upload_path = os.path.join(UPLOAD_DIR, filename)
                
                # Add a suffix to the processed filename to avoid overwriting
                base, ext = os.path.splitext(filename)
                processed_filename = f"{base}_processed{ext}"
                processed_path = os.path.join(PROCESSED_DIR, processed_filename)

                # Create directories if they don't exist
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                os.makedirs(PROCESSED_DIR, exist_ok=True)

                with open(upload_path, 'wb') as f:
                    f.write(file_item.file.read())

                try:
                    # Process the file
                    processor = SanskritPostProcessor()
                    processor.process_srt_file(Path(upload_path), Path(processed_path))

                    # Respond with a link to the processed file
                    # Construct the download link
                    processed_dir_forward_slash = PROCESSED_DIR.replace('\\', '/')
                    download_link = f"/{processed_dir_forward_slash}/{processed_filename}"

                    self.send_response(303) # See Other
                    self.send_header('Location', download_link)
                    self.end_headers()

                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"<html><body><h2>Error during processing</h2>")
                    self.wfile.write(f"<p>{e}</p></body></html>".encode('utf-8'))
            else:
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'No file selected. Please choose a file to upload.')
        else:
            self.send_error(404, 'File Not Found: %s' % self.path)

def run_server():
    # Create web directory if it doesn't exist
    os.makedirs(WEB_DIR, exist_ok=True)
    
    with socketserver.TCPServer(('', PORT), MyHttpRequestHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down the server.")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()