"""
EvoCode Web Interface - Command Line Entry Point
This module provides a command-line entry point to run the web interface.
"""

import argparse
import os
from .app import run_app

def main():
    """
    Command-line entry point for running the EvoCode web interface.
    """
    parser = argparse.ArgumentParser(description='Run the EvoCode web interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host IP to listen on (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['FLASK_DEBUG'] = str(args.debug)
    os.environ['PORT'] = str(args.port)
    
    # Output startup message
    print(f"Starting EvoCode web interface on http://{args.host}:{args.port}")
    if args.debug:
        print("Running in DEBUG mode - not recommended for production!")
    
    # Run the app
    run_app()

if __name__ == '__main__':
    main() 