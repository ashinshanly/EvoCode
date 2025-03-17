"""
Tests for the core EvoCode functionality.
"""

import pytest
from evocode import EvoCode
from evocode.fitness import FitnessEvaluator
from evocode.engine import EvolutionEngine
from evocode.mutation import CodeMutation


def add_function(a, b):
    """Simple function to test evolution."""
    return a + b


def test_evocode_initialization():
    """Test that the EvoCode class can be imported and initialized."""
    assert EvoCode is not None


def test_fitness_evaluator():
    """Test that the FitnessEvaluator works correctly."""
    # Create test cases
    test_cases = [
        {"args": [1, 2], "expected": 3},
        {"args": [3, 4], "expected": 7},
    ]
    
    # Create evaluator
    evaluator = FitnessEvaluator(test_cases, optimization_metric="accuracy")
    
    # Test accuracy evaluation
    accuracy = evaluator._evaluate_accuracy(add_function)
    assert accuracy == 1.0  # All test cases should pass
    
    # Create incorrect function
    def wrong_add(a, b):
        return a - b
    
    # Test accuracy evaluation with incorrect function
    accuracy = evaluator._evaluate_accuracy(wrong_add)
    assert accuracy == 0.0  # No test cases should pass


def test_code_mutation():
    """Test that code mutation produces valid functions."""
    # Create mutations
    mutations = CodeMutation.mutate_function(add_function, mutation_rate=0.5)
    
    # Check that we got at least one mutation
    assert len(mutations) > 0
    
    # Check that all mutations are callable
    for mutation in mutations:
        assert callable(mutation)
        
    # Check that mutations work correctly on basic cases
    for mutation in mutations:
        assert mutation(1, 2) == 3  # Basic test case should still work


def test_evolution_engine():
    """Test that the evolution engine works correctly."""
    # Create test cases
    test_cases = [
        {"args": [1, 2], "expected": 3},
        {"args": [3, 4], "expected": 7},
    ]
    
    # Create evaluator
    evaluator = FitnessEvaluator(test_cases, optimization_metric="accuracy")
    
    # Create engine
    engine = EvolutionEngine(
        initial_function=add_function,
        fitness_evaluator=evaluator,
        population_size=5,
        generations=2,
        mutation_rate=0.2,
        elite_size=1
    )
    
    # Run evolution
    result = engine.evolve()
    
    # Check that we got a callable result
    assert callable(result)
    
    # Check that the result still works correctly
    assert result(1, 2) == 3
    assert result(3, 4) == 7
    
    # Check that we got some statistics
    stats = engine.get_stats()
    assert len(stats) > 0


def test_evolve_function():
    """Test the main evolve_function API."""
    # Create test cases
    test_cases = [
        {"args": [1, 2], "expected": 3},
        {"args": [3, 4], "expected": 7},
    ]
    
    # Evolve the function
    evolved_func = EvoCode.evolve_function(
        func=add_function,
        test_cases=test_cases,
        optimization_metric="accuracy",
        generations=2,
        population_size=5
    )
    
    # Check that we got a callable result
    assert callable(evolved_func)
    
    # Check that the result still works correctly
    assert evolved_func(1, 2) == 3
    assert evolved_func(3, 4) == 7


def test_compare_functions():
    """Test the compare_functions API."""
    # Create test cases
    test_cases = [
        {"args": [1, 2], "expected": 3},
        {"args": [3, 4], "expected": 7},
    ]
    
    # Compare functions
    comparison = EvoCode.compare_functions(
        original_func=add_function,
        evolved_func=add_function,  # Same function for now
        test_cases=test_cases
    )
    
    # Check that we got a comparison result
    assert isinstance(comparison, dict)
    assert "original_speed" in comparison
    assert "evolved_speed" in comparison
    assert "original_accuracy" in comparison
    assert "evolved_accuracy" in comparison


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 