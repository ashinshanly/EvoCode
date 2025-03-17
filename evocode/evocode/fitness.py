"""
Fitness evaluation for the EvoCode library.
"""

from typing import Callable, List, Dict, Any, Union, Optional
import timeit
import time
import logging
import traceback
import signal
from contextlib import contextmanager
import math

logger = logging.getLogger("evocode")


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for limiting execution time of code."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


class FitnessEvaluator:
    """Evaluates the fitness of code variants."""
    
    def __init__(self, 
                 test_cases: List[Dict[str, Any]], 
                 optimization_metric: str = "speed",
                 timeout: float = 1.0):
        """
        Initialize the fitness evaluator.
        
        Args:
            test_cases: List of test cases to evaluate functions against
            optimization_metric: What to optimize for ("speed", "memory", "accuracy", etc.)
            timeout: Maximum execution time per test case in seconds
        """
        self.test_cases = test_cases
        self.optimization_metric = optimization_metric.lower()
        self.timeout = max(0.1, timeout)  # Ensure minimum timeout
        
        # Validate the optimization metric
        valid_metrics = ["speed", "memory", "accuracy", "combined"]
        if self.optimization_metric not in valid_metrics:
            logger.warning(f"Invalid optimization metric: {optimization_metric}")
            logger.warning(f"Defaulting to 'speed'. Valid options are: {', '.join(valid_metrics)}")
            self.optimization_metric = "speed"
    
    def evaluate(self, func: Callable) -> float:
        """Evaluate a function's fitness score."""
        if func is None:
            logger.warning("Cannot evaluate None function")
            return float('-inf')
            
        try:
            # Verify function is callable
            if not callable(func):
                logger.warning(f"Object is not callable: {type(func)}")
                return float('-inf')
                
            # Get function name for logging
            func_name = getattr(func, '__name__', 'unknown')
            logger.debug(f"Evaluating function '{func_name}'")
                
            # Evaluate based on metric
            if self.optimization_metric == "speed":
                return self._evaluate_speed(func)
            elif self.optimization_metric == "memory":
                return self._evaluate_memory(func)
            elif self.optimization_metric == "accuracy":
                return self._evaluate_accuracy(func)
            elif self.optimization_metric == "combined":
                # Combined metric that balances speed and accuracy
                speed_score = self._evaluate_speed(func)
                accuracy_score = self._evaluate_accuracy(func)
                
                # Normalize and combine (higher is better)
                return 0.4 * speed_score + 0.6 * accuracy_score
            else:
                logger.error(f"Unsupported optimization metric: {self.optimization_metric}")
                return float('-inf')
        except Exception as e:
            logger.warning(f"Error evaluating function: {str(e)}")
            return float('-inf')
    
    def _evaluate_speed(self, func: Callable) -> float:
        """Evaluate function speed across test cases."""
        total_time = 0
        valid_tests = 0
        
        for i, test_case in enumerate(self.test_cases):
            try:
                args = test_case.get("args", [])
                kwargs = test_case.get("kwargs", {})
                
                try:
                    start_time = time.time()
                    with time_limit(self.timeout):
                        func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    valid_tests += 1
                    
                    # Record time for this test case
                    total_time += execution_time
                    logger.debug(f"Test case {i} execution time: {execution_time:.6f}s")
                    
                except TimeoutException:
                    # Penalize functions that timeout
                    logger.debug(f"Function timed out on test case {i}")
                    total_time += self.timeout * 5
                except Exception as e:
                    # Other errors during execution
                    logger.debug(f"Error executing function on test case {i}: {str(e)}")
                    total_time += self.timeout * 2
            except Exception as e:
                # Errors in the test case processing itself
                logger.warning(f"Error processing test case {i}: {str(e)}")
                total_time += self.timeout * 2
                
        # Calculate average time (lower is better)
        avg_time = total_time / max(1, len(self.test_cases))
        logger.debug(f"Speed evaluation: avg_time={avg_time:.6f}, valid_tests={valid_tests}/{len(self.test_cases)}")
        
        # Convert to a positive fitness score between 0 and 10 (higher is better)
        # Use an exponential decay formula: 10 * e^(-avg_time/timeout)
        # This makes very fast functions have values close to 10
        # and very slow functions have values close to 0
        positive_fitness = 10 * math.exp(-avg_time / self.timeout)
        
        logger.debug(f"Converted negative time {-avg_time} to positive fitness {positive_fitness}")
        return positive_fitness
    
    def _evaluate_memory(self, func: Callable) -> float:
        """Evaluate function memory usage across test cases."""
        # Implementation needed
        logger.error("Memory evaluation not implemented")
        return float('-inf')
    
    def _evaluate_accuracy(self, func: Callable) -> float:
        """Evaluate function accuracy across test cases."""
        correct_results = 0
        
        # If no test cases, return a poor score
        if not self.test_cases:
            logger.warning("No test cases provided for accuracy evaluation")
            return 0.0
            
        for i, test_case in enumerate(self.test_cases):
            try:
                args = test_case.get("args", [])
                kwargs = test_case.get("kwargs", {})
                expected = test_case.get("expected")
                
                if "expected" not in test_case:
                    logger.warning(f"Test case {i} missing 'expected' value")
                    continue
                    
                try:
                    with time_limit(self.timeout):
                        result = func(*args, **kwargs)
                    
                    if self._results_equal(result, expected):
                        correct_results += 1
                except (TimeoutException, Exception) as e:
                    # Functions that error or timeout don't add to score
                    logger.debug(f"Error in accuracy evaluation for test case {i}: {str(e)}")
                    pass
            except Exception as e:
                # Errors in the test case processing itself
                logger.warning(f"Error processing test case {i} for accuracy: {str(e)}")
                    
        # Return accuracy percentage (higher is better)
        accuracy = correct_results / max(1, len(self.test_cases))
        logger.debug(f"Accuracy evaluation: {correct_results}/{len(self.test_cases)} = {accuracy:.2f}")
        return accuracy
        
    def _results_equal(self, result, expected) -> bool:
        """
        Compare function results with expected values, handling special cases.
        
        Args:
            result: Result from the function
            expected: Expected result from test case
            
        Returns:
            True if results match, False otherwise
        """
        # Handle None values
        if result is None and expected is None:
            return True
            
        # Handle different numeric types
        if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
            # Use approximate equality for floating point
            if isinstance(result, float) or isinstance(expected, float):
                return abs(result - expected) < 1e-6
            else:
                return result == expected
                
        # Handle lists and other sequences
        if isinstance(result, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(result) != len(expected):
                return False
            return all(self._results_equal(r, e) for r, e in zip(result, expected))
            
        # Default comparison
        return result == expected 