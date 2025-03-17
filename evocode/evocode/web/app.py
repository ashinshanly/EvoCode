"""
EvoCode Web Interface - Flask Application
This module defines the Flask application that serves as the web interface for the EvoCode library.
"""

import os
import sys
import logging
import traceback
from flask import Flask, render_template, Blueprint, send_from_directory, request
from flask_socketio import SocketIO, join_room, leave_room

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evocode_web.log')
    ]
)

# Create Flask-SocketIO instance
socketio = SocketIO()

def create_app(test_config=None):
    """
    Create and configure the Flask application
    
    Args:
        test_config: Configuration to use for testing (default: None)
        
    Returns:
        Flask application instance
    """
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Configure app
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB max upload
        TEMPLATES_AUTO_RELOAD=True,
        DEBUG=os.environ.get('FLASK_DEBUG', 'False') == 'True',
    )
    
    # Override config with test config if provided
    if test_config is not None:
        app.config.from_mapping(test_config)
    
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(app.instance_path, 'temp')
    try:
        os.makedirs(temp_dir)
    except OSError:
        pass
    
    # Initialize extensions
    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading", logger=True, engineio_logger=True)
    
    # Set up socket.io event handlers
    @socketio.on('connect')
    def handle_connect():
        client_id = request.sid
        app.logger.info(f"Client connected: {client_id}")
        # Send connection acknowledgment
        socketio.emit('connection_status', {
            'status': 'connected',
            'sid': client_id,
            'timestamp': __import__('time').time()
        }, to=client_id)
    
    @socketio.on('disconnect')
    def handle_disconnect():
        app.logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('subscribe')
    def handle_subscribe(data):
        client_id = request.sid
        process_id = data.get('process_id')
        app.logger.info(f"Client {client_id} subscribing to process: {process_id}")
        
        if not process_id:
            app.logger.warning(f"Client {client_id} attempted to subscribe without a process_id")
            socketio.emit('error', {
                'message': 'No process_id provided for subscription'
            }, room=client_id)
            return
            
        try:
            join_room(process_id)
            app.logger.info(f"Client {client_id} joined room {process_id}")
            
            # Confirm subscription
            socketio.emit('subscribed', {
                'process_id': process_id,
                'timestamp': __import__('time').time()
            }, room=client_id)
            
            # Also emit global test message to verify connectivity
            socketio.emit('debug_message', {
                'message': f'Client {client_id} subscribed to {process_id}',
                'timestamp': __import__('time').time()
            })
        except Exception as e:
            app.logger.error(f"Error during subscription: {str(e)}")
            app.logger.error(traceback.format_exc())
            socketio.emit('error', {
                'message': f'Subscription error: {str(e)}'
            }, room=client_id)
    
    @socketio.on('ping')
    def handle_ping(data, callback=None):
        client_id = request.sid
        app.logger.debug(f"Ping received from client {client_id}")
        
        # Prepare response data with server timestamp
        response = {
            'server_time': __import__('time').time(),
            'received_data': data
        }
        
        # Send response via callback or emit
        if callback:
            callback(response)
        else:
            socketio.emit('pong', response, room=client_id)
    
    @socketio.on_error()
    def handle_socket_error(e):
        app.logger.error(f"Socket.IO error: {str(e)}")
        app.logger.error(traceback.format_exc())
        socketio.emit('server_error', {
            'message': f'Server error: {str(e)}',
            'timestamp': __import__('time').time()
        }, room=request.sid)
    
    # Register blueprints
    from .routes import api_bp
    app.register_blueprint(api_bp)
    
    # Create a main blueprint for non-API routes
    main_bp = Blueprint('main', __name__)
    
    @main_bp.route('/')
    def index():
        """Render the main page"""
        return render_template('index.html')
    
    @main_bp.route('/favicon.ico')
    def favicon():
        """Serve the favicon"""
        return send_from_directory(
            os.path.join(app.root_path, 'static'),
            'favicon.ico', 
            mimetype='image/vnd.microsoft.icon'
        )
    
    # Register the main blueprint
    app.register_blueprint(main_bp)
    
    # Custom template filters
    @app.template_filter('now')
    def template_now(format_string='%Y'):
        """Return the current date/time formatted according to format_string"""
        from datetime import datetime
        return datetime.now().strftime(format_string)
    
    # Custom error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors"""
        return render_template('error.html', error=e, code=404), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors"""
        app.logger.error(f"Server error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return render_template('error.html', error=e, code=500), 500
    
    # Return the app
    return app

def run_app():
    """Run the Flask application with Flask-SocketIO"""
    app = create_app()
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True, use_reloader=False)
    
if __name__ == '__main__':
    run_app() 