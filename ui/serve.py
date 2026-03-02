"""
Simple HTTP server to serve the UI.
Run this to access the user-friendly interface.
"""
import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 3000
DIRECTORY = Path(__file__).parent

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

def serve_ui():
    """Start the UI server."""
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("=" * 60)
        print("🌐 User Interface Server Started!")
        print("=" * 60)
        print(f"\n✅ Open in browser: http://localhost:{PORT}")
        print(f"✅ API running at: http://localhost:8000")
        print("\n📝 Instructions:")
        print("   1. The page will open automatically")
        print("   2. Fill in customer info or use quick examples")
        print("   3. Click 'Predict Churn Risk' to see results")
        print("\n⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        print()
        
        # Open browser automatically
        webbrowser.open(f"http://localhost:{PORT}")
        
        httpd.serve_forever()

if __name__ == "__main__":
    serve_ui()
