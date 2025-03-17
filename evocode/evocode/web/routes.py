"""
EvoCode Web Interface - API Routes
This module defines the API routes for the EvoCode web interface.
"""

import os
import uuid
import time
import threading
import traceback
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from .code_runner import CodeEvolutionRunner

# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Store active evolution processes
active_processes = {}

@api_bp.route('/submit', methods=['POST'])
def submit_code():
    """
    API endpoint to submit code for optimization.
    
    Request data:
        - code: Source code to optimize
        - function_name: Name of the function to optimize
        - metrics: Optimization metrics (comma-separated)
        - generations: Number of generations to evolve
        - population_size: Size of the population
        - mutation_rate: Probability of mutation
        - file: Optional file upload with source code
        
    Returns:
        JSON with process_id and function_name
    """
    try:
        # Get form data
        source_code = request.form.get('code', '')
        function_name = request.form.get('function_name', '')
        metrics = request.form.get('metrics', 'speed').split(',')
        
        # Validate and convert numeric parameters
        try:
            generations = int(request.form.get('generations', 20))
            if generations <= 0 or generations > 100:
                return jsonify({"error": "Generations must be between 1 and 100"}), 400
        except ValueError:
            return jsonify({"error": "Invalid value for generations parameter"}), 400
            
        try:
            population_size = int(request.form.get('population_size', 50))
            if population_size <= 0 or population_size > 200:
                return jsonify({"error": "Population size must be between 1 and 200"}), 400
        except ValueError:
            return jsonify({"error": "Invalid value for population_size parameter"}), 400
            
        try:
            mutation_rate = float(request.form.get('mutation_rate', 0.3))
            if mutation_rate <= 0 or mutation_rate > 1.0:
                return jsonify({"error": "Mutation rate must be between 0 and 1.0"}), 400
        except ValueError:
            return jsonify({"error": "Invalid value for mutation_rate parameter"}), 400
        
        # Check if a file was uploaded
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file.filename.endswith('.py'):
                # Read the file contents
                try:
                    source_code = file.read().decode('utf-8')
                except UnicodeDecodeError:
                    return jsonify({"error": "Unable to decode the uploaded file. Please ensure it's a valid text file."}), 400
                
                # If no function name is provided, try to find one in the code
                if not function_name:
                    import re
                    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', source_code)
                    if match:
                        function_name = match.group(1)
            else:
                return jsonify({"error": "Uploaded file must have a .py extension"}), 400
        
        # Validate input
        if not source_code:
            return jsonify({"error": "No code provided"}), 400
            
        if not function_name:
            return jsonify({"error": "No function name specified"}), 400
            
        # Validate Python syntax
        import ast
        try:
            ast.parse(source_code)
        except SyntaxError as e:
            return jsonify({
                "error": f"Python syntax error: {str(e)}",
                "line": e.lineno,
                "column": e.offset,
                "text": e.text
            }), 400
        
        # Validate function exists in the code
        try:
            tree = ast.parse(source_code)
            function_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    function_found = True
                    break
                    
            if not function_found:
                return jsonify({"error": f"Function '{function_name}' not found in the provided code"}), 400
        except Exception as e:
            current_app.logger.error(f"Error validating function: {str(e)}")
            current_app.logger.error(traceback.format_exc())
        
        # Generate a unique ID for this process
        process_id = str(uuid.uuid4())
        
        # Create a runner for the evolution process
        runner = CodeEvolutionRunner(
            code=source_code,
            function_name=function_name,
            optimization_metric=metrics[0],  # Use the first metric if multiple are provided
            generations=generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            socketio=current_app.extensions['socketio'],
            process_id=process_id  # Pass process_id to the constructor
        )
        
        # Store the runner
        active_processes[process_id] = runner
        
        # Start the evolution process in a separate thread
        thread = threading.Thread(target=runner.start_evolution)
        thread.daemon = True
        thread.start()
        
        current_app.logger.info(f"Started evolution process {process_id} for function '{function_name}'")
        
        # Return the process ID
        response = {
            "process_id": process_id,
            "function_name": function_name,
            "message": "Evolution process started"
        }
        
        # Also emit a global event to notify about the new process
        try:
            socketio = current_app.extensions['socketio']
            socketio.emit('new_process', {
                'process_id': process_id,
                'function_name': function_name,
                'timestamp': time.time()
            })
        except Exception as e:
            current_app.logger.error(f"Error emitting new process event: {str(e)}")
            
        return jsonify(response), 202
        
    except Exception as e:
        current_app.logger.error(f"Error in submit_code: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@api_bp.route('/status/<process_id>', methods=['GET'])
def get_status(process_id):
    """
    Get the status of an evolution process.
    
    Args:
        process_id: The ID of the process to check
        
    Returns:
        JSON response with the status of the process
    """
    try:
        if process_id not in active_processes:
            return jsonify({"error": "Process not found"}), 404
        
        # Get the runner for this process
        runner = active_processes[process_id]
        
        # Get the status
        status = runner.get_status()
        
        return jsonify(status), 200
    
    except Exception as e:
        current_app.logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/result/<process_id>', methods=['GET'])
def get_result(process_id):
    """
    Get the results of an evolution process.
    
    Args:
        process_id: The ID of the process to check
        
    Returns:
        JSON response with the results of the process
    """
    try:
        if process_id not in active_processes:
            return jsonify({"error": "Process not found"}), 404
        
        # Get the runner for this process
        runner = active_processes[process_id]
        
        # Get the results
        result = runner.get_result()
        
        # Log the result for debugging
        current_app.logger.debug(f"Result for process {process_id}: {result.keys() if isinstance(result, dict) else 'not a dict'}")
        
        # Check if the process is complete directly from the runner
        if runner.is_complete:
            # Don't call _cleanup here as it might delete files still needed
            # Just log that we got the final result
            current_app.logger.debug(f"Got final result for completed process {process_id}")
            
            # We could schedule cleanup for later if needed
            # runner._cleanup()  # This is the correct method name
        
        return jsonify(result), 200
    
    except Exception as e:
        current_app.logger.error(f"Error getting result: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_bp.route('/cancel/<process_id>', methods=['POST'])
def cancel_process(process_id):
    """
    Cancel an evolution process.
    
    Args:
        process_id: The ID of the process to cancel
        
    Returns:
        JSON response with a confirmation message
    """
    try:
        if process_id not in active_processes:
            return jsonify({"error": "Process not found"}), 404
        
        # Get the runner for this process
        runner = active_processes[process_id]
        
        # Cancel the evolution process
        runner.cancel()
        
        # Return success
        return jsonify({
            "message": "Evolution process cancelled",
            "process_id": process_id
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error cancelling process: {str(e)}")
        return jsonify({"error": str(e)}), 500 