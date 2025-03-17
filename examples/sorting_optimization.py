#!/usr/bin/env python3
"""
Example: Evolving a sorting function for better performance.

This example demonstrates how EvoCode can be used to optimize
a simple sorting algorithm for better performance.
"""

import sys
import os
import random
import time
from typing import List

# Add the parent directory to the path so we can import evocode
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evocode import EvoCode
from evocode.utils import print_function_diff, benchmark
from evocode.visualization import FunctionVisualizer


def bubble_sort(arr: List[int]) -> List[int]:
    """
    A simple bubble sort implementation to be optimized.
    
    Args:
        arr: List of integers to sort
        
    Returns:
        Sorted list of integers
    """
    # Make a copy to avoid modifying the original
    result = arr.copy()
    n = len(result)
    
    # Bubble sort algorithm
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    
    return result


def generate_test_cases(num_cases=10, max_size=1000, max_value=1000):
    """Generate random test cases for sorting."""
    test_cases = []
    
    # Generate arrays of different sizes
    for _ in range(num_cases):
        # Random array size
        size = random.randint(10, max_size)
        
        # Generate random array
        arr = [random.randint(-max_value, max_value) for _ in range(size)]
        
        # Expected result is the sorted array
        expected = sorted(arr)
        
        # Add to test cases
        test_cases.append({
            "args": [arr],
            "expected": expected
        })
    
    return test_cases


def main():
    print("=" * 80)
    print("EvoCode Example: Sorting Optimization")
    print("=" * 80)
    print("\nGenerating test cases...")
    
    # Generate test cases
    test_cases = generate_test_cases(
        num_cases=15,
        max_size=500,
        max_value=1000
    )
    
    print(f"Generated {len(test_cases)} test cases for sorting")
    print(f"Example test case sizes: {[len(case['args'][0]) for case in test_cases[:3]]}")
    
    # Benchmark the original function
    print("\nBenchmarking original bubble_sort function...")
    original_times = []
    for case in test_cases:
        args = case["args"]
        start_time = time.perf_counter()
        result = bubble_sort(*args)
        elapsed = time.perf_counter() - start_time
        original_times.append(elapsed)
    
    avg_original_time = sum(original_times) / len(original_times)
    print(f"Average time: {avg_original_time:.6f} seconds")
    
    # Evolve the function
    print("\nStarting evolution process...")
    evolved_func = EvoCode.evolve_function(
        func=bubble_sort,
        test_cases=test_cases,
        optimization_metric="speed",
        generations=20,
        population_size=30,
        mutation_rate=0.3,
        timeout=5.0
    )
    
    # Print comparison
    print("\nComparison of original and evolved functions:")
    print_function_diff(bubble_sort, evolved_func)
    
    # Benchmark the evolved function
    print("\nBenchmarking evolved function...")
    evolved_times = []
    for case in test_cases:
        args = case["args"]
        start_time = time.perf_counter()
        result = evolved_func(*args)
        elapsed = time.perf_counter() - start_time
        evolved_times.append(elapsed)
    
    avg_evolved_time = sum(evolved_times) / len(evolved_times)
    print(f"Average time: {avg_evolved_time:.6f} seconds")
    
    # Calculate improvement
    improvement = (avg_original_time - avg_evolved_time) / avg_original_time * 100
    print(f"\nPerformance improvement: {improvement:.2f}%")
    
    # Visualize the comparison
    print("\nGenerating performance comparison visualization...")
    FunctionVisualizer.compare_performance(
        bubble_sort, 
        evolved_func, 
        test_cases,
        save_path="sorting_performance.png"
    )
    print("Visualization saved to 'sorting_performance.png'")
    
    print("\nDone!")


if __name__ == "__main__":
    main() 