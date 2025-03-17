"""
Utility functions for the EvoCode library.
"""

import ast
import inspect
import time
import logging
from typing import Callable, List, Dict, Any, Union, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("evocode")


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Human-readable time string
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1_000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes} min {seconds:.2f} s"


def function_complexity(func: Callable) -> int:
    """
    Calculate the complexity of a function using AST analysis.
    
    Args:
        func: Function to analyze
        
    Returns:
        Complexity score (higher means more complex)
    """
    try:
        # Get the source code
        source = inspect.getsource(func)
        
        # Parse into AST
        tree = ast.parse(source)
        
        # Count the number of nodes as a simple complexity metric
        node_count = sum(1 for _ in ast.walk(tree))
        
        return node_count
    except Exception as e:
        logger.warning(f"Error calculating function complexity: {str(e)}")
        return 0


def benchmark(func: Callable, args: List, kwargs: Dict = None, repeats: int = 10) -> Tuple[float, float]:
    """
    Benchmark a function's execution time.
    
    Args:
        func: Function to benchmark
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        repeats: Number of times to repeat the measurement
        
    Returns:
        Tuple of (average time, standard deviation)
    """
    if kwargs is None:
        kwargs = {}
        
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return avg_time, std_dev


def plot_evolution_progress(stats: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot the progress of the evolution process.
    
    Args:
        stats: Evolution statistics from EvolutionEngine.get_stats()
        save_path: Optional path to save the plot image
    """
    try:
        generations = [s["generation"] for s in stats]
        max_fitness = [s["max_fitness"] for s in stats]
        avg_fitness = [s["avg_fitness"] for s in stats]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, max_fitness, 'b-', label='Max Fitness')
        plt.plot(generations, avg_fitness, 'r--', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting evolution progress: {str(e)}")


def print_function_diff(original_func: Callable, evolved_func: Callable):
    """
    Print a side-by-side comparison of original and evolved functions.
    
    Args:
        original_func: The original function
        evolved_func: The evolved function
    """
    try:
        original_source = inspect.getsource(original_func)
        evolved_source = inspect.getsource(evolved_func)
        
        original_lines = original_source.splitlines()
        evolved_lines = evolved_source.splitlines()
        
        # Find the maximum line length for formatting
        max_line_length = max(len(line) for line in original_lines)
        
        print("\n" + "=" * 80)
        print("FUNCTION COMPARISON")
        print("=" * 80)
        print(f"{'ORIGINAL FUNCTION':<{max_line_length + 5}}| EVOLVED FUNCTION")
        print("-" * (max_line_length + 5) + "+" + "-" * 50)
        
        # Print the functions side by side
        for i in range(max(len(original_lines), len(evolved_lines))):
            original_line = original_lines[i] if i < len(original_lines) else ""
            evolved_line = evolved_lines[i] if i < len(evolved_lines) else ""
            
            print(f"{original_line:<{max_line_length + 5}}| {evolved_line}")
            
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Error printing function diff: {str(e)}")


def create_test_suite(func: Callable, num_cases: int = 10) -> List[Dict[str, Any]]:
    """
    Automatically generate test cases for a function based on its signature.
    
    Args:
        func: Function to generate test cases for
        num_cases: Number of test cases to generate
        
    Returns:
        List of test cases
    """
    test_cases = []
    sig = inspect.signature(func)
    
    # Get the parameter types if annotations are available
    param_types = {}
    for name, param in sig.parameters.items():
        if param.annotation != inspect.Parameter.empty:
            param_types[name] = param.annotation
    
    # Generate random test cases
    for _ in range(num_cases):
        args = []
        kwargs = {}
        
        for name, param in sig.parameters.items():
            # Generate a random value based on type, if known
            if name in param_types:
                param_type = param_types[name]
                if param_type == int:
                    value = np.random.randint(-100, 100)
                elif param_type == float:
                    value = np.random.uniform(-100, 100)
                elif param_type == bool:
                    value = np.random.choice([True, False])
                elif param_type == str:
                    value = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=5))
                else:
                    value = None
            else:
                # Default to random int if type not known
                value = np.random.randint(-100, 100)
            
            # Add to args or kwargs depending on parameter kind
            if param.kind == inspect.Parameter.POSITIONAL_ONLY or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(value)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[name] = value
        
        # Run the function to get the expected result
        try:
            expected = func(*args, **kwargs)
            test_cases.append({
                "args": args,
                "kwargs": kwargs,
                "expected": expected
            })
        except Exception as e:
            # Skip test cases that cause errors
            logger.debug(f"Skipping invalid test case: {str(e)}")
    
    return test_cases 