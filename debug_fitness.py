"""
Script to debug fitness evaluation in EvoCode.
"""

import sys
import os
import logging
import math
import time

# Add the correct path to import evocode
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evocode'))

from fitness import FitnessEvaluator
from core import EvoCode

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evocode")

# Sample function to test
def bubble_sort(arr):
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

# Test cases
test_cases = [
    {"args": [[5, 3, 8, 6, 7, 2]], "expected": [2, 3, 5, 6, 7, 8]},
    {"args": [[1, 2, 3, 4, 5]], "expected": [1, 2, 3, 4, 5]},
    {"args": [[9, 8, 7, 6, 5]], "expected": [5, 6, 7, 8, 9]}
]

def test_fitness():
    """Directly test the fitness evaluation."""
    print("Testing fitness evaluation directly...")
    
    # Create a fitness evaluator
    evaluator = FitnessEvaluator(
        test_cases=test_cases,
        optimization_metric="speed",
        timeout=1.0
    )
    
    # Time the function execution directly
    start_time = time.time()
    bubble_sort([5, 3, 8, 6, 7, 2])
    execution_time = time.time() - start_time
    print(f"Direct execution time: {execution_time:.6f}s")
    
    # Evaluate the function using the fitness evaluator
    fitness = evaluator._evaluate_speed(bubble_sort)
    print(f"Direct fitness score: {fitness}")
    
    # Create another evaluator for accuracy
    accuracy_evaluator = FitnessEvaluator(
        test_cases=test_cases,
        optimization_metric="accuracy"
    )
    
    # Evaluate accuracy
    accuracy = accuracy_evaluator._evaluate_accuracy(bubble_sort)
    print(f"Accuracy score: {accuracy}")

def test_compare():
    """Test the comparison function."""
    print("\nTesting comparison function...")
    
    # Compare the function with itself (should be 0% improvement)
    comparison = EvoCode.compare_functions(
        original_func=bubble_sort,
        evolved_func=bubble_sort,
        test_cases=test_cases
    )
    
    print(f"Comparison result: {comparison}")
    print(f"Original speed: {comparison['original_speed']}")
    print(f"Evolved speed: {comparison['evolved_speed']}")
    print(f"Speed improvement: {comparison['speed_improvement_percentage']}%")

if __name__ == "__main__":
    test_fitness()
    test_compare() 