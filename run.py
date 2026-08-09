import os
from app import app

if __name__ == '__main__':
    # Use environment variables if set, otherwise defaults
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    
    print(f"Starting SPEAR Flask application on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
