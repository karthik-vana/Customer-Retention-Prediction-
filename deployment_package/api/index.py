from http.server import BaseHTTPRequestHandler
import sys
import os
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Flask app
from app.app import app

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Create a WSGI-like environment
        environ = {
            'REQUEST_METHOD': 'GET',
            'PATH_INFO': self.path,
            'QUERY_STRING': urlparse(self.path).query,
            'SERVER_NAME': 'vercel',
            'SERVER_PORT': '443',
            'wsgi.url_scheme': 'https',
            'wsgi.input': None,
            'wsgi.errors': sys.stderr,
        }
        
        # Handle the request with Flask
        with app.test_client() as client:
            response = client.get(self.path)
            
            self.send_response(response.status_code)
            for key, value in response.headers:
                if key.lower() not in ['content-length', 'transfer-encoding']:
                    self.send_header(key, value)
            self.send_header('Content-Type', response.content_type or 'text/html')
            self.end_headers()
            self.wfile.write(response.data)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        with app.test_client() as client:
            response = client.post(
                self.path,
                data=post_data,
                content_type=self.headers.get('Content-Type', 'application/json')
            )
            
            self.send_response(response.status_code)
            for key, value in response.headers:
                if key.lower() not in ['content-length', 'transfer-encoding']:
                    self.send_header(key, value)
            self.send_header('Content-Type', response.content_type or 'application/json')
            self.end_headers()
            self.wfile.write(response.data)
