"""
Core functionality for the EvoCode library.
"""

from typing import Callable, List, Dict, Any, Optional, Union
import inspect
import logging
import traceback

from .engine import EvolutionEngine
from .fitness import FitnessEvaluator
from .mutation import CodeMutation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evocode")


class EvoCode:
    """Primary interface for the EvoCode library."""
    
    @staticmethod
    def evolve_function(
        func: Callable,
        test_cases: List[Dict[str, Any]],
        optimization_metric: str = "speed",
        generations: int = 50,
        population_size: int = 20,
        mutation_rate: float = 0.2,
        max_workers: int = None
    ) -> Callable:
        """
        Evolve a function to optimize for the given metric.
        
        Args:
            func: The initial function to evolve
            test_cases: Test cases to evaluate function variants
            optimization_metric: What to optimize for ("speed" or "accuracy")
            generations: Number of generations to evolve
            population_size: Size of each generation's population
            mutation_rate: Probability of mutation operations
            max_workers: Maximum number of worker processes for parallel execution
            
        Returns:
            The evolved function optimized for the given metric
        """
        try:
            logger.info(f"Starting evolution of function '{func.__name__}'")
            logger.info(f"Optimization metric: {optimization_metric}")
            logger.info(f"Generations: {generations}, Population size: {population_size}")
            
            # Validate test cases
            if not test_cases:
                logger.warning("No test cases provided. Creating minimal test case.")
                # Create a minimal test case to avoid errors
                test_cases = [{
                    "args": [],
                    "kwargs": {},
                    "expected": None  # Will be filled with actual result
                }]
                
                # Try to get expected result
                try:
                    test_cases[0]["expected"] = func()
                except Exception as e:
                    logger.warning(f"Could not execute function without arguments: {str(e)}")
            
            # Create a fitness evaluator
            evaluator = FitnessEvaluator(
                test_cases=test_cases,
                optimization_metric=optimization_metric
            )
            
            # Create and run the evolution engine
            engine = EvolutionEngine(
                initial_function=func,
                fitness_evaluator=evaluator,
                generations=generations,
                population_size=population_size,
                mutation_rate=mutation_rate,
                max_workers=max_workers
            )
            
            # Run the evolution process
            logger.info("Starting evolution process")
            evolved_func = engine.evolve()
            
            # Log results
            logger.info(f"Evolution complete. Best fitness: {engine.best_fitness:.6f}")
            
            return evolved_func
            
        except Exception as e:
            logger.error(f"Error during evolution: {str(e)}")
            logger.error(traceback.format_exc())
            # Return the original function as fallback
            return func
    
    @staticmethod
    def compare_functions(
        original_func: Callable,
        evolved_func: Callable,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare the performance of two functions.
        
        Args:
            original_func: The original function
            evolved_func: The evolved function
            test_cases: List of test cases to use for evaluation
            
        Returns:
            Dictionary with performance comparison metrics
        """
        logger.debug(f"Comparing functions: {original_func.__name__} vs evolved")
        
        # Check if functions are valid
        if not callable(original_func):
            logger.error(f"Original function is not callable: {original_func}")
            return {"error": "Original function is not callable"}
        
        if not callable(evolved_func):
            logger.error(f"Evolved function is not callable: {evolved_func}")
            return {"error": "Evolved function is not callable"}
        
        # Create a fitness evaluator
        evaluator = FitnessEvaluator(test_cases)
        
        try:
            # Evaluate speed
            original_speed = evaluator._evaluate_speed(original_func)
            evolved_speed = evaluator._evaluate_speed(evolved_func)
            
            # Log raw speed values
            logger.debug(f"Raw speed values - Original: {original_speed}, Evolved: {evolved_speed}")
            
            # Calculate speed improvement
            if original_speed > 0:
                speed_improvement = ((evolved_speed - original_speed) / original_speed) * 100
            else:
                speed_improvement = 0
                
            logger.debug(f"Speed improvement: {speed_improvement}%")
            
            # Evaluate accuracy
            original_accuracy = evaluator._evaluate_accuracy(original_func)
            evolved_accuracy = evaluator._evaluate_accuracy(evolved_func)
            
            # Log accuracy values
            logger.debug(f"Accuracy values - Original: {original_accuracy}, Evolved: {evolved_accuracy}")
            
            # Calculate accuracy change
            accuracy_improvement = evolved_accuracy - original_accuracy
            logger.debug(f"Accuracy improvement: {accuracy_improvement}")
            
            # Create result dictionary
            result = {
                "original_speed": original_speed,
                "evolved_speed": evolved_speed,
                "speed_improvement_percentage": speed_improvement,
                "original_accuracy": original_accuracy,
                "evolved_accuracy": evolved_accuracy,
                "accuracy_improvement": accuracy_improvement
            }
            
            logger.debug(f"Comparison result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error comparing functions: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": f"Error comparing functions: {str(e)}",
                "original_speed": 0,
                "evolved_speed": 0,
                "speed_improvement_percentage": 0,
                "original_accuracy": 0,
                "evolved_accuracy": 0,
                "accuracy_improvement": 0
            } 