"""
Code runner for the EvoCode web interface.
"""

import os
import sys
import time
import importlib.util
import tempfile
import inspect
import traceback
import json
from typing import Dict, Any, List, Optional, Callable
import threading
import logging
import ast
import math

from .. import EvoCode
from ..utils import create_test_suite, benchmark
from ..engine import EvolutionEngine
from ..fitness import FitnessEvaluator
from ..visualization import FunctionVisualizer
from ..mutation import CodeMutation

logger = logging.getLogger(__name__)


class CodeEvolutionRunner:
    """Handles the evolution of code for the web interface."""
    
    def __init__(self, 
                 code: str, 
                 function_name: str,
                 optimization_metric: str = "speed",
                 generations: int = 20,
                 population_size: int = 30,
                 mutation_rate: float = 0.2,
                 timeout: float = 1.0,
                 process_id: str = None,
                 socketio = None):
        """
        Initialize the code evolution runner.
        
        Args:
            code: Source code containing the function to optimize
            function_name: Name of the function to optimize
            optimization_metric: What to optimize for ("speed" or "accuracy")
            generations: Number of generations to evolve
            population_size: Size of the population in each generation
            mutation_rate: Probability of mutation (0.0 to 1.0)
            timeout: Maximum execution time per test case in seconds
            process_id: Unique ID for this evolution process
            socketio: Flask-SocketIO instance for real-time updates
        """
        self.code = code
        self.function_name = function_name
        self.optimization_metric = optimization_metric
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.timeout = timeout
        self.process_id = process_id
        self.socketio = socketio
        
        self.temp_dir = tempfile.mkdtemp(prefix='evocode_')
        self.module_path = os.path.join(self.temp_dir, f"{function_name}.py")
        
        # Status tracking
        self.is_running = False
        self.is_complete = False
        self.is_cancelled = False
        self.start_time = None
        self.end_time = None
        self.current_generation = 0
        self.best_fitness = float('-inf')
        self.function_obj = None
        self.evolved_function = None
        self.test_cases = []
        self.original_source = ""
        self.evolved_source = ""
        self.evolution_stats = []
        self.error = None
        self.performance_comparison = {}
        self.current_mutations = []
        
        # Event to signal cancellation
        self.cancel_event = threading.Event()
    
    def load_function(self) -> Callable:
        """
        Load the target function from the provided code.
        
        Returns:
            The loaded function object
        """
        try:
            # Write the code to a file
            with open(self.module_path, 'w') as f:
                f.write(self.code)
            
            # Load the module
            module_name = os.path.basename(self.module_path)[:-3]  # Remove .py
            spec = importlib.util.spec_from_file_location(module_name, self.module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the function
            if not hasattr(module, self.function_name):
                raise ValueError(f"Function '{self.function_name}' not found in the provided code")
            
            function_obj = getattr(module, self.function_name)
            if not callable(function_obj):
                raise ValueError(f"'{self.function_name}' is not a callable function")
            
            # Get the source code of the function
            self.original_source = inspect.getsource(function_obj)
            
            return function_obj
        
        except Exception as e:
            logger.error(f"Error loading function: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def generate_test_cases(self, func: Callable) -> List[Dict[str, Any]]:
        """
        Generate test cases for the function.
        
        Args:
            func: The function to generate test cases for
            
        Returns:
            List of test cases
        """
        # Try to extract test cases from docstring or generate automatic ones
        try:
            # First attempt: Look for test cases in the function's docstring
            docstring = inspect.getdoc(func)
            if docstring and "test cases" in docstring.lower():
                # Parse docstring for test cases (this is a simplified approach)
                # In a real implementation, you would need more sophisticated parsing
                test_cases = []
                # For now, we'll just generate automatic test cases
            
            # Generate automatic test cases based on function signature
            try:
                test_cases = create_test_suite(func, num_cases=10)
                
                # If test_cases is empty, create some simple test cases
                if not test_cases:
                    self._emit_status("generating_test_cases", {
                        "message": "Automatic test case generation failed, creating simple test cases..."
                    })
                    
                    # Create some simple test cases based on function signature
                    signature = inspect.signature(func)
                    test_cases = self._create_fallback_test_cases(func, signature)
            except Exception as e:
                # Log the specific error
                logger.error(f"Error in test case generation: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Try to create fallback test cases
                self._emit_status("generating_test_cases", {
                    "message": f"Error generating test cases: {str(e)}. Creating fallback test cases..."
                })
                
                # Create simple test cases based on function signature
                signature = inspect.signature(func)
                test_cases = self._create_fallback_test_cases(func, signature)
            
            if not test_cases:
                raise ValueError("Could not generate test cases for the function. Please check that the function signature is not too complex and try again.")
            
            return test_cases
        
        except Exception as e:
            logger.error(f"Error generating test cases: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Could not generate test cases for the function: {str(e)}. Please check that the function has valid parameters and return values.")
    
    def _create_fallback_test_cases(self, func: Callable, signature: inspect.Signature) -> List[Dict[str, Any]]:
        """
        Create basic fallback test cases when automatic generation fails.
        
        Args:
            func: The function to create test cases for
            signature: The function's signature
            
        Returns:
            List of simple test cases
        """
        test_cases = []
        try:
            # Get parameter names
            param_names = list(signature.parameters.keys())
            
            # If the function has no parameters, create a simple test case
            if not param_names:
                # Call the function with no arguments to get a baseline result
                try:
                    result = func()
                    test_cases.append({
                        "args": [],
                        "kwargs": {},
                        "expected": result
                    })
                except Exception:
                    # If calling with no arguments fails, just return an empty test case
                    test_cases.append({
                        "args": [],
                        "kwargs": {},
                        "expected": None
                    })
                return test_cases
            
            # For functions with parameters, create simple test cases with basic values
            basic_values = [0, 1, -1, 100, -100, 0.5, -0.5, "", "test", [], [1, 2, 3], {}, {"key": "value"}]
            
            # Try each basic value for the first parameter
            for value in basic_values[:5]:  # Limit to first 5 values for simplicity
                args = [value] + [0] * (len(param_names) - 1)  # Fill remaining params with 0
                try:
                    result = func(*args)
                    test_cases.append({
                        "args": args,
                        "kwargs": {},
                        "expected": result
                    })
                    
                    # If we've got at least 3 test cases, that's enough for a basic test
                    if len(test_cases) >= 3:
                        break
                except Exception:
                    # If this value doesn't work, try the next one
                    continue
            
            # If we couldn't create any test cases, try string values for all parameters
            if not test_cases:
                try:
                    args = ["test"] * len(param_names)
                    result = func(*args)
                    test_cases.append({
                        "args": args,
                        "kwargs": {},
                        "expected": result
                    })
                except Exception:
                    # Last resort: just create a dummy test case
                    test_cases.append({
                        "args": [0] * len(param_names),
                        "kwargs": {},
                        "expected": None  # We don't know the expected result
                    })
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Error creating fallback test cases: {str(e)}")
            # Return at least one dummy test case to avoid breaking the evolution
            return [{
                "args": [],
                "kwargs": {},
                "expected": None
            }]
    
    def start_evolution(self):
        """Start the evolution process."""
        self.is_running = True
        self.start_time = time.time()
        logger.info(f"Starting evolution for function {self.function_name} with process ID {self.process_id}")
        
        self._emit_status("evolution_started", {
            "process_id": self.process_id,
            "function_name": self.function_name,
            "start_time": self.start_time,
        })
        
        try:
            # Load the function
            logger.info("Loading function...")
            self._emit_status("loading_function", {
                "message": f"Loading function '{self.function_name}'..."
            })
            self.function_obj = self.load_function()
            logger.info(f"Function loaded successfully: {self.function_name}")
            
            # Generate test cases
            logger.info("Generating test cases...")
            self._emit_status("generating_test_cases", {
                "message": "Generating test cases..."
            })
            self.test_cases = self.generate_test_cases(self.function_obj)
            logger.info(f"Generated {len(self.test_cases)} test cases")
            
            self._emit_status("test_cases_generated", {
                "test_cases": self._format_test_cases(self.test_cases),
                "count": len(self.test_cases)
            })
            
            # Create a custom observer to track evolution progress
            logger.info("Setting up evolution...")
            observer = self._create_evolution_observer()
            
            # Start the evolution
            logger.info(f"Starting evolution over {self.generations} generations with population size {self.population_size}")
            self._emit_status("evolution_in_progress", {
                "message": f"Evolving function over {self.generations} generations...",
                "generations": self.generations,
                "population_size": self.population_size
            })
            
            # Create a fitness evaluator
            logger.info("Creating fitness evaluator...")
            fitness_evaluator = FitnessEvaluator(
                test_cases=self.test_cases,
                optimization_metric=self.optimization_metric,
                timeout=self.timeout
            )
            
            # Create an evolution engine with our observer
            logger.info("Creating evolution engine...")
            try:
                engine = EvolutionEngine(
                    initial_function=self.function_obj,
                    fitness_evaluator=fitness_evaluator,
                    population_size=self.population_size,
                    generations=self.generations,
                    mutation_rate=self.mutation_rate,
                    elite_size=2,
                    max_workers=1  # Use single process to avoid pickling issues
                )
                logger.info("Evolution engine created successfully")
            except Exception as e:
                logger.error(f"Error creating evolution engine: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            # Store the original engine's evolve method
            original_evolve = engine.evolve
            
            # Override the evolve method to add our observer logic
            def evolve_with_observer():
                logger.info("Starting evolution process...")
                
                # Initialize population
                logger.info("Initializing population...")
                try:
                    engine._initialize_population()
                    logger.info(f"Population initialized with {len(engine.current_population)} individuals")
                except Exception as e:
                    logger.error(f"Error initializing population: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
                
                # Initialize tracking variables
                best_so_far = None
                best_fitness_so_far = float('-inf')
                valid_population_count = 0
                
                # Initialize stats arrays if empty
                engine.generation_stats = []
                self.evolution_stats = []
                
                # Initial population - use sources instead of function objects
                try:
                    # Create functions from sources for display only
                    sample_functions = []
                    for source in engine.current_population[:5]:  # Limit to 5 samples
                        try:
                            func = CodeMutation.source_to_function(source, engine.function_globals)
                            if func:
                                sample_functions.append(func)
                        except Exception as e:
                            logger.warning(f"Error converting source to function: {str(e)}")
                            
                    self._emit_mutation_update(sample_functions)
                except Exception as e:
                    logger.error(f"Error emitting initial mutations: {str(e)}")
                
                # Evolve for the specified number of generations
                for generation in range(engine.generations):
                    logger.info(f"Starting generation {generation + 1}/{engine.generations}")
                    
                    # Check if cancelled
                    if self.cancel_event.is_set():
                        logger.info("Evolution cancelled by user")
                        self._emit_status("evolution_cancelled", {
                            "message": "Evolution process was cancelled"
                        })
                        # Return best so far if we found one
                        return best_so_far if best_so_far is not None else self.function_obj
                    
                    self.current_generation = generation + 1
                    
                    # Evaluate fitness
                    try:
                        logger.info("Evaluating population fitness...")
                        fitness_scores = engine._evaluate_population()
                        
                        if not fitness_scores:
                            logger.warning("No fitness scores returned from evaluation")
                            fitness_scores = [float('-inf')] * len(engine.current_population)
                            
                        # Check if we got some valid scores
                        valid_scores = [s for s in fitness_scores if s > float('-inf')]
                        if valid_scores:
                            min_fitness = min(valid_scores)
                            max_fitness = max(valid_scores)
                            logger.info(f"Fitness evaluation complete. Valid scores range: {min_fitness} to {max_fitness}")
                            valid_population_count += 1
                        else:
                            logger.warning("No valid fitness scores in this generation")
                            
                        # If we've had 3 generations with no valid population, abort
                        if generation >= 2 and valid_population_count == 0:
                            logger.error("No valid fitness scores in the first 3 generations. Aborting evolution.")
                            self._emit_status("evolution_error", {
                                "message": "Could not evolve function: no valid fitness scores in multiple generations",
                                "traceback": "The evolution process could not produce valid function variants."
                            })
                            return self.function_obj
                    except Exception as e:
                        logger.error(f"Error evaluating population fitness: {str(e)}")
                        logger.error(traceback.format_exc())
                        # Try to continue with empty fitness scores
                        fitness_scores = [float('-inf')] * len(engine.current_population)
                    
                    # Calculate statistics
                    valid_scores = [s for s in fitness_scores if s > float('-inf')]
                    if valid_scores:
                        max_fitness = max(valid_scores)
                        avg_fitness = sum(valid_scores) / len(valid_scores)
                    else:
                        max_fitness = float('-inf')
                        avg_fitness = float('-inf')
                    
                    # Update best function if we found a better one
                    if valid_scores:
                        best_idx = fitness_scores.index(max(valid_scores))
                        
                        # Initialize best_fitness if needed
                        if not hasattr(engine, 'best_fitness') or engine.best_fitness is None:
                            engine.best_fitness = float('-inf')
                            
                        # Check if this is better than best so far
                        if max(valid_scores) > engine.best_fitness:
                            engine.best_fitness = max(valid_scores)
                            best_source = engine.current_population[best_idx]
                            
                            try:
                                best_func = CodeMutation.source_to_function(best_source, engine.function_globals)
                                
                                if best_func:
                                    engine.best_function = best_func
                                    engine.best_source = best_source
                                    best_so_far = best_func  # Save for return value
                                    best_fitness_so_far = engine.best_fitness
                                    logger.info(f"New best fitness found: {engine.best_fitness}")
                                    
                                    # Store the evolved source
                                    self.evolved_source = best_source
                            except Exception as e:
                                logger.error(f"Error creating function from best source: {str(e)}")
                    
                    # Make sure best_fitness is initialized
                    if not hasattr(engine, 'best_fitness') or engine.best_fitness is None:
                        engine.best_fitness = max_fitness
                    
                    # Add to stats
                    stats = {
                        "generation": generation + 1,
                        "max_fitness": max_fitness,
                        "avg_fitness": avg_fitness,
                        "best_fitness": engine.best_fitness if hasattr(engine, 'best_fitness') else max_fitness
                    }
                    
                    # Add stats to both the engine and our local collection
                    if not hasattr(engine, 'generation_stats'):
                        engine.generation_stats = []
                    engine.generation_stats.append(stats)
                    self.evolution_stats.append(stats)
                    
                    # Log stats for debugging
                    logger.info(f"Stats updated - generation: {stats['generation']}, max_fitness: {stats['max_fitness']}, avg_fitness: {stats['avg_fitness']}")
                    
                    # Record best fitness for status
                    self.best_fitness = engine.best_fitness if hasattr(engine, 'best_fitness') else max_fitness
                    
                    # Emit progress update
                    try:
                        self._emit_status("generation_complete", {
                            "generation": generation + 1,
                            "generations": engine.generations,
                            "max_fitness": max_fitness,
                            "avg_fitness": avg_fitness,
                            "best_fitness": self.best_fitness,
                            "progress": (generation + 1) / engine.generations * 100
                        })
                    except Exception as e:
                        logger.error(f"Error emitting generation progress: {str(e)}")
                    
                    # Generate the next population
                    try:
                        logger.info("Creating next generation...")
                        engine._create_next_generation(fitness_scores)
                        logger.info(f"Next generation created with {len(engine.current_population)} individuals")
                    except Exception as e:
                        logger.error(f"Error creating next generation: {str(e)}")
                        logger.error(traceback.format_exc())
                        # If creation fails severely, use the original function
                        if not engine.current_population:
                            engine.current_population = [self.original_source] * engine.population_size
                    
                    # Emit mutation update - convert sources to functions for display
                    try:
                        sample_functions = []
                        for source in engine.current_population[:5]:  # Limit to 5 samples
                            try:
                                if isinstance(source, str):  # Make sure it's a string
                                    func = CodeMutation.source_to_function(source, engine.function_globals)
                                    if func:
                                        sample_functions.append(func)
                            except Exception as e:
                                logger.warning(f"Error converting source to function: {str(e)}")
                                
                        self._emit_mutation_update(sample_functions)
                    except Exception as e:
                        logger.error(f"Error emitting mutation update: {str(e)}")
                    
                    # Update visualization data after each generation
                    try:
                        self._emit_visualization_data()
                    except Exception as e:
                        logger.error(f"Error emitting visualization data: {str(e)}")
                    
                    # Small delay to prevent flooding the client
                    time.sleep(0.1)
                
                logger.info("Evolution complete. Returning best function.")
                
                # Make sure we have a valid best function
                if best_so_far is not None:
                    return best_so_far
                elif hasattr(engine, 'best_function') and engine.best_function:
                    return engine.best_function
                elif hasattr(engine, 'best_source') and engine.best_source:
                    try:
                        best_func = CodeMutation.source_to_function(engine.best_source, engine.function_globals)
                        if best_func:
                            return best_func
                    except Exception:
                        pass
                
                # If no good function found, return the original
                logger.warning("No valid best function found. Returning original function.")
                return self.function_obj
            
            # Replace the evolve method
            engine.evolve = evolve_with_observer
            
            # Run the evolution
            logger.info("Starting evolution...")
            try:
                self.evolved_function = engine.evolve()
                logger.info("Evolution process completed successfully")
            except Exception as e:
                logger.error(f"Error during evolution: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            # If we don't have a valid evolved function, use the original
            if not self.evolved_function:
                logger.warning("No valid evolved function. Using original function.")
                self.evolved_function = self.function_obj
                self.evolved_source = self.original_source
                
            # Benchmark and compare
            logger.info("Benchmarking functions...")
            self._emit_status("benchmarking", {
                "message": "Benchmarking original and evolved functions..."
            })
            
            try:
                self.performance_comparison = self._benchmark_functions()
                improvement = self.performance_comparison.get('speed_improvement_percentage', 0)
                logger.info(f"Benchmark complete. Performance improvement: {improvement}%")
            except Exception as e:
                logger.error(f"Error benchmarking functions: {str(e)}")
                logger.error(traceback.format_exc())
                self.performance_comparison = {
                    "error": str(e),
                    "speed_improvement_percentage": 0
                }
            
            # Emit final update
            self.is_complete = True
            self.end_time = time.time()
            logger.info(f"Evolution complete. Total time: {self.end_time - self.start_time:.2f} seconds")
            
            self._emit_status("evolution_complete", {
                "message": "Evolution complete",
                "total_time": self.end_time - self.start_time,
                "generations_completed": self.current_generation,
                "best_fitness": self.best_fitness,
                "performance_improvement": self.performance_comparison.get("speed_improvement_percentage", 0)
            })
            
            # Generate final visualizations
            try:
                self._generate_visualizations()
            except Exception as e:
                logger.error(f"Error generating visualizations: {str(e)}")
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error in evolution process: {str(e)}")
            logger.error(traceback.format_exc())
            self._emit_status("evolution_error", {
                "message": f"Error: {str(e)}",
                "traceback": traceback.format_exc()
            })
        
        finally:
            self.is_running = False
            logger.info("Evolution process thread completed")
    
    def _create_evolution_observer(self):
        """Create an observer to track evolution progress."""
        # This would be an implementation of an observer pattern
        # For simplicity, we'll just use callbacks in the evolve method override
        pass
    
    def _emit_status(self, event_type: str, data: Dict[str, Any]):
        """
        Emit a status update via SocketIO.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if self.socketio:
            try:
                # Add timestamp and process ID
                data['timestamp'] = time.time()
                data['process_id'] = self.process_id
                
                # Log the event emission for debugging
                logger.info(f"Emitting event: {event_type} with data: {str(data)[:200]}...")
                
                # Emit to the room for this process
                self.socketio.emit(event_type, data, room=self.process_id)
                
                # For critical events, also emit a global event without room restriction
                if event_type in ['evolution_started', 'evolution_complete', 'evolution_error', 'evolution_cancelled']:
                    self.socketio.emit(f"global_{event_type}", {
                        'process_id': self.process_id,
                        'timestamp': time.time(),
                        'message': data.get('message', f"Event: {event_type}")
                    })
            except Exception as e:
                logger.error(f"Error emitting socket.io event: {str(e)}")
                logger.error(traceback.format_exc())
    
    def _emit_mutation_update(self, population: List[Callable]):
        """
        Emit an update with the current population's mutations.
        
        Args:
            population: Current population of function variants
        """
        try:
            # Get source code diffs for a sample of the population
            mutation_data = []
            
            # Log the incoming population for debugging
            valid_funcs = [f for f in population if f is not None]
            logger.info(f"Processing mutation update with {len(valid_funcs)}/{len(population)} valid functions")
            
            if not valid_funcs:
                logger.warning("No valid functions in population for mutation update")
                # Try to generate a new set of mutations directly from the original function
                try:
                    if self.function_obj:
                        logger.info("Attempting to generate direct mutations from original function")
                        original_source = inspect.getsource(self.function_obj)
                        mutations = CodeMutation.mutate_function(original_source, 0.2)
                        
                        # Try to create functions from these mutations
                        for i, mut_source in enumerate(mutations[:3]):  # Limit to 3 attempts
                            try:
                                if CodeMutation.validate_python_syntax(mut_source):
                                    func_globals = self.function_obj.__globals__.copy()
                                    func = CodeMutation.source_to_function(mut_source, func_globals)
                                    if func is not None:
                                        valid_funcs.append(func)
                                        logger.info(f"Successfully created direct mutation {i}")
                            except Exception as e:
                                logger.warning(f"Failed to create direct mutation {i}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error generating direct mutations: {str(e)}")
            
            # Limit to a reasonable sample size but try to ensure we have something
            sample_size = min(5, len(valid_funcs)) if valid_funcs else 0
            for i, func in enumerate(valid_funcs[:sample_size]):
                try:
                    # Skip None values
                    if func is None:
                        continue
                        
                    source = inspect.getsource(func)
                    # Extract the function body for display
                    try:
                        parsed = ast.parse(source)
                        func_def = parsed.body[0]
                        
                        # Get the function name
                        func_name = func_def.name if hasattr(func_def, 'name') else func.__name__
                        
                        # Get only the function body without the decorator and def line
                        func_body = "\n".join(
                            line for line in source.split("\n")
                            if not line.strip().startswith("@")
                        )
                        
                        # Add a comment showing the mutation type if we can detect it
                        mutation_type = "Unknown Mutation"
                        if "sorted(" in func_body and "sorted(" not in self.original_source:
                            mutation_type = "Algorithm Optimization"
                        elif len(func_body.split("\n")) < len(self.original_source.split("\n")):
                            mutation_type = "Code Simplification"
                        elif len(func_body.split("\n")) > len(self.original_source.split("\n")):
                            mutation_type = "Logic Extension"
                        
                        # Add a descriptive comment
                        func_body = f"# {mutation_type}\n{func_body}"
                        
                    except Exception as parse_error:
                        logger.warning(f"Error parsing function body: {str(parse_error)}")
                        func_body = source
                        func_name = func.__name__
                    
                    # Validate that we have valid source code
                    if not func_body or not func_body.strip():
                        logger.warning(f"Empty function body for mutation {i}")
                        continue
                    
                    mutation_data.append({
                        "id": i,
                        "source": func_body,
                        "name": func_name
                    })
                    logger.debug(f"Added mutation {i} with name {func_name}")
                    
                except Exception as e:
                    logger.warning(f"Error getting source for mutation {i}: {str(e)}")
            
            # If we have no valid mutations, create more informative fallback mutations
            if not mutation_data:
                logger.warning("No valid mutations could be processed. Creating informative fallbacks.")
                
                # Create a couple of different simple mutations as examples
                mutation_data.append({
                    "id": 0,
                    "source": f"# Example mutation: Parameter validation\ndef {self.function_name}(*args, **kwargs):\n    # Added input validation\n    if len(args) == 0:\n        return None\n    \n    # Original function body would be here\n    result = None\n    # ... processing logic ...\n    return result",
                    "name": f"{self.function_name}_validation"
                })
                
                mutation_data.append({
                    "id": 1,
                    "source": f"# Example mutation: Caching results\ndef {self.function_name}(*args, **kwargs):\n    # Added caching mechanism\n    cache_key = str(args) + str(kwargs)\n    if hasattr({self.function_name}, '_cache') and cache_key in {self.function_name}._cache:\n        return {self.function_name}._cache[cache_key]\n    \n    # Original function logic would compute the result\n    result = None\n    # ... processing logic ...\n    \n    # Cache the result\n    if not hasattr({self.function_name}, '_cache'):\n        {self.function_name}._cache = {{}}\n    {self.function_name}._cache[cache_key] = result\n    return result",
                    "name": f"{self.function_name}_cached"
                })
            
            self.current_mutations = mutation_data
            
            # Emit the mutation data
            self._emit_status("mutation_update", {
                "mutations": mutation_data,
                "population_size": len(population)
            })
            
            logger.info(f"Emitted {len(mutation_data)} mutations for display")
            
        except Exception as e:
            logger.error(f"Error emitting mutation update: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _benchmark_functions(self) -> Dict[str, Any]:
        """
        Benchmark the original and evolved functions.
        
        Returns:
            Dictionary with benchmark results
        """
        logger.debug(f"Starting benchmark comparison for {self.function_name}")
        
        # Check if we have both functions
        if self.function_obj is None:
            logger.error(f"Original function is None for {self.function_name}")
            return {
                "error": "Original function is not available",
                "original_speed": 0,
                "evolved_speed": 0,
                "speed_improvement_percentage": 0,
                "original_accuracy": 0,
                "evolved_accuracy": 0,
                "accuracy_improvement": 0
            }
            
        if self.evolved_function is None:
            logger.error(f"Evolved function is None for {self.function_name}")
            return {
                "error": "Evolved function is not available",
                "original_speed": 0,
                "evolved_speed": 0,
                "speed_improvement_percentage": 0,
                "original_accuracy": 0,
                "evolved_accuracy": 0,
                "accuracy_improvement": 0
            }
        
        try:
            # Use the EvoCode.compare_functions method to benchmark
            result = EvoCode.compare_functions(
                self.function_obj,
                self.evolved_function,
                self.test_cases
            )
            
            logger.debug(f"Benchmark results for {self.function_name}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error benchmarking functions: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return a structured error response with default values
            return {
                "error": f"Error benchmarking functions: {str(e)}",
                "original_speed": 0,
                "evolved_speed": 0,
                "speed_improvement_percentage": 0,
                "original_accuracy": 0,
                "evolved_accuracy": 0,
                "accuracy_improvement": 0
            }
    
    def _emit_visualization_data(self):
        """Emit visualization data for the evolution chart."""
        try:
            if not self.evolution_stats:
                logger.warning("No evolution stats available for visualization")
                # Send empty data to initialize the chart
                visualization_data = {
                    "generations": [0],
                    "max_fitness": [0],
                    "avg_fitness": [0]
                }
                
                self._emit_status("visualization_data", {
                    "chart_data": visualization_data
                })
                return
                
            # Convert to format for chart.js
            generations = [s.get("generation", i+1) for i, s in enumerate(self.evolution_stats)]
            
            # Extract and clean fitness values
            max_fitness = []
            avg_fitness = []
            
            for s in self.evolution_stats:
                # Get the values with fallbacks
                max_val = s.get("max_fitness", float('-inf'))
                avg_val = s.get("avg_fitness", float('-inf'))
                
                # Clean the values for the chart
                if max_val == float('-inf') or max_val is None or not math.isfinite(max_val):
                    max_val = 0.0  # Default to zero for invalid values
                else:
                    max_val = float(max_val)  # Ensure it's a float
                    
                if avg_val == float('-inf') or avg_val is None or not math.isfinite(avg_val):
                    avg_val = 0.0  # Default to zero for invalid values
                else:
                    avg_val = float(avg_val)  # Ensure it's a float
                
                max_fitness.append(max_val)
                avg_fitness.append(avg_val)
            
            # Log sample of the data we're sending
            data_points = len(generations)
            logger.info(f"Sending visualization data with {data_points} data points")
            
            if data_points > 0:
                sample_index = min(5, data_points - 1)
                logger.info(f"Sample data - Generation: {generations[sample_index]}, Max: {max_fitness[sample_index]}, Avg: {avg_fitness[sample_index]}")
            
            # Create the data dictionary
            visualization_data = {
                "generations": generations,
                "max_fitness": max_fitness,
                "avg_fitness": avg_fitness
            }
            
            # Also emit to the room
            self._emit_status("visualization_data", {
                "chart_data": visualization_data
            })
            
            # Also emit globally to ensure it's received
            self.socketio.emit("visualization_data", {
                "chart_data": visualization_data
            })
            
            logger.info("Visualization data emitted successfully")
        except Exception as e:
            logger.error(f"Error emitting visualization data: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _generate_visualizations(self):
        """Generate visualizations for the evolution process."""
        try:
            # Ensure we emit using both standard and global methods
            self._emit_visualization_data()
            
            # Also emit a direct event with performance comparison
            if self.performance_comparison and self.socketio:
                try:
                    performance_data = {
                        "original_speed": self.performance_comparison.get("original_speed", 0),
                        "evolved_speed": self.performance_comparison.get("evolved_speed", 0),
                        "original_accuracy": self.performance_comparison.get("original_accuracy", 0),
                        "evolved_accuracy": self.performance_comparison.get("evolved_accuracy", 0),
                        "speed_improvement": self.performance_comparison.get("speed_improvement_percentage", 0),
                        "accuracy_improvement": self.performance_comparison.get("accuracy_improvement", 0)
                    }
                    
                    self._emit_status("performance_data", {
                        "performance": performance_data
                    })
                    
                    # Also emit globally
                    self.socketio.emit("global_performance_data", {
                        "process_id": self.process_id,
                        "performance": performance_data
                    })
                except Exception as e:
                    logger.error(f"Error emitting performance data: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _format_test_cases(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format test cases for display in the UI.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Formatted test cases
        """
        formatted = []
        for i, test_case in enumerate(test_cases):
            # Truncate or format the test case for display
            args = test_case.get("args", [])
            kwargs = test_case.get("kwargs", {})
            expected = test_case.get("expected")
            
            # Format args and expected value for display
            formatted_args = ', '.join([repr(arg) for arg in args])
            for key, value in kwargs.items():
                formatted_args += f", {key}={repr(value)}"
            
            formatted.append({
                "id": i + 1,
                "args": formatted_args,
                "expected": repr(expected)
            })
        
        return formatted
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the evolution process.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "process_id": self.process_id,
            "function_name": self.function_name,
            "is_running": self.is_running,
            "is_complete": self.is_complete,
            "is_cancelled": self.is_cancelled,
            "current_generation": self.current_generation,
            "total_generations": self.generations,
            "progress": (self.current_generation / self.generations * 100) if self.generations > 0 else 0,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": (self.end_time - self.start_time) if self.end_time else (time.time() - self.start_time) if self.start_time else 0,
        }
        
        if self.evolution_stats:
            # Include the latest stats
            latest_stats = self.evolution_stats[-1]
            status.update({
                "best_fitness": latest_stats.get("best_fitness", 0),
                "max_fitness": latest_stats.get("max_fitness", 0),
                "avg_fitness": latest_stats.get("avg_fitness", 0),
            })
        
        if self.performance_comparison:
            status.update({
                "performance_improvement": self.performance_comparison.get("speed_improvement_percentage", 0)
            })
        
        return status
    
    def get_result(self) -> Dict[str, Any]:
        """
        Get the final result of the evolution process.
        
        Returns:
            Dictionary with result information
        """
        logger.debug(f"get_result called for process {self.process_id}, is_complete: {self.is_complete}")
        
        if not self.is_complete:
            logger.debug("Evolution not complete, returning error")
            return {"error": "Evolution still in progress"}
        
        # Log key data points
        logger.debug(f"Original source length: {len(self.original_source) if self.original_source else 0}")
        logger.debug(f"Evolved source length: {len(self.evolved_source) if self.evolved_source else 0}")
        logger.debug(f"Performance comparison: {self.performance_comparison}")
        logger.debug(f"Evolution stats count: {len(self.evolution_stats) if self.evolution_stats else 0}")
        logger.debug(f"Total time: {(self.end_time - self.start_time) if self.end_time else 0}")
        logger.debug(f"Generations completed: {self.current_generation}")
        logger.debug(f"Test cases count: {len(self.test_cases) if self.test_cases else 0}")
        
        result = {
            "process_id": self.process_id,
            "function_name": self.function_name,
            "original_source": self.original_source,
            "evolved_source": self.evolved_source,
            "performance_comparison": self.performance_comparison,
            "evolution_stats": self.evolution_stats,
            "total_time": (self.end_time - self.start_time) if self.end_time else 0,
            "generations_completed": self.current_generation,
            "test_cases": self._format_test_cases(self.test_cases),
            "is_complete": self.is_complete  # Add this flag to help frontend
        }
        
        logger.debug(f"Returning result with keys: {list(result.keys())}")
        return result
    
    def cancel(self):
        """Cancel the evolution process."""
        self.is_cancelled = True
        self.cancel_event.set()
        
        # If not running, clean up immediately
        if not self.is_running:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}") 
            logger.error(f"Error cleaning up: {str(e)}") 