"""
Evolution engine for the EvoCode library.
"""

from typing import Callable, List, Dict, Any, Optional, Union
import random
import logging
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import copy
import inspect
import math

from .mutation import CodeMutation
from .fitness import FitnessEvaluator

logger = logging.getLogger("evocode")


class EvolutionEngine:
    """Core engine that manages the evolutionary process."""
    
    def __init__(self, 
                 initial_function: Callable,
                 fitness_evaluator: FitnessEvaluator,
                 population_size: int = 20,
                 generations: int = 50,
                 mutation_rate: float = 0.2,
                 elite_size: int = 2,
                 max_workers: int = None):
        """
        Initialize the evolution engine.
        
        Args:
            initial_function: The starting function to evolve
            fitness_evaluator: Evaluator to determine fitness
            population_size: Number of variants in each generation
            generations: How many generations to evolve
            mutation_rate: Probability of mutation operations
            elite_size: Number of top performers to keep unchanged
            max_workers: Maximum number of workers for parallel processing
        """
        self.initial_function = initial_function
        
        # Store the original source code
        try:
            self.initial_source = inspect.getsource(initial_function)
        except Exception as e:
            logger.error(f"Error getting source for initial function: {str(e)}")
            self.initial_source = "def error_function():\n    return None"
        
        self.fitness_evaluator = fitness_evaluator
        self.population_size = max(5, population_size)  # Ensure minimum population size
        self.generations = max(1, generations)  # Ensure at least one generation
        self.mutation_rate = max(0.01, min(0.5, mutation_rate))  # Limit between 0.01 and 0.5
        self.elite_size = min(elite_size, self.population_size // 4)  # Limit elite size
        self.max_workers = max_workers
        
        # Initialize with source code instead of function objects
        self.current_population = [self.initial_source]
        self.best_function = initial_function
        self.best_source = self.initial_source
        self.best_fitness = float('-inf')
        self.generation_stats = []
        
        # Store function globals for later use when compiling source to functions
        self.function_globals = initial_function.__globals__
        
    def evolve(self) -> Callable:
        """Run the evolutionary process and return the best function."""
        # Initialize population
        logger.info("Initializing population...")
        self._initialize_population()
        
        # Evolve for the specified number of generations
        for generation in range(self.generations):
            logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Evaluate fitness for each variant
            fitness_scores = self._evaluate_population()
            
            # Calculate statistics for this generation
            max_fitness = max(fitness_scores) if fitness_scores else float('-inf')
            avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else float('-inf')
            
            self.generation_stats.append({
                "generation": generation + 1,
                "max_fitness": max_fitness,
                "avg_fitness": avg_fitness
            })
            
            logger.info(f"  Max fitness: {max_fitness:.6f}")
            logger.info(f"  Avg fitness: {avg_fitness:.6f}")
            
            # Update best function if we found a better one
            if fitness_scores:
                best_idx = fitness_scores.index(max(fitness_scores))
                if fitness_scores[best_idx] > self.best_fitness:
                    self.best_fitness = fitness_scores[best_idx]
                    self.best_source = self.current_population[best_idx]
                    # Convert source to function for return value
                    best_func = CodeMutation.source_to_function(self.best_source, self.function_globals)
                    if best_func:
                        self.best_function = best_func
                        logger.info(f"  New best function found! Fitness: {self.best_fitness:.6f}")
                
            # Generate the next population
            self._create_next_generation(fitness_scores)
            
        logger.info(f"Evolution complete. Best fitness: {self.best_fitness:.6f}")
        return self.best_function
    
    def _initialize_population(self):
        """Create the initial population of function variants."""
        logger.info(f"Initializing population with size {self.population_size}")
        
        # Start with the initial function source
        self.current_population = [self.initial_source]
        
        # Track valid mutations to avoid repeating failures
        valid_mutations = []
        attempts = 0
        max_attempts = self.population_size * 5  # Limit attempts to avoid infinite loops
        
        # Generate the rest through mutations
        while len(self.current_population) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            # Select a random function from the current population
            source = random.choice(self.current_population)
            
            # Generate mutations
            try:
                mutations = CodeMutation.mutate_function(source, self.mutation_rate)
                
                # Validate each mutation before adding it
                for mutation in mutations:
                    # Skip if we already have enough population members
                    if len(self.current_population) >= self.population_size:
                        break
                        
                    # Skip duplicates
                    if mutation in self.current_population or mutation in valid_mutations:
                        continue
                        
                    # Verify the mutation is valid code
                    if not CodeMutation.validate_python_syntax(mutation):
                        fixed_mutation = CodeMutation.fix_incomplete_blocks(mutation)
                        if not CodeMutation.validate_python_syntax(fixed_mutation):
                            logger.debug(f"Skipping invalid mutation: {mutation[:50]}...")
                            continue
                        mutation = fixed_mutation
                    
                    # Verify we can create a function from it
                    func = CodeMutation.source_to_function(mutation, self.function_globals)
                    if func is None:
                        logger.debug("Skipping mutation that couldn't be converted to a function")
                        continue
                    
                    # Add valid mutation to our tracking and population
                    valid_mutations.append(mutation)
                    self.current_population.append(mutation)
                    
            except Exception as e:
                logger.warning(f"Error generating mutations: {str(e)}")
        
        # If we couldn't generate enough valid mutations, clone existing ones
        while len(self.current_population) < self.population_size:
            # Clone a random valid member
            self.current_population.append(random.choice(self.current_population))
        
        logger.info(f"Population initialized with {len(self.current_population)} individuals")
    
    def _evaluate_population(self) -> List[float]:
        """Evaluate the fitness of all functions in the current population."""
        fitness_scores = []
        
        # Convert source code to functions for evaluation
        functions = []
        valid_sources = []
        
        # First pass: try to convert all sources to functions
        for i, source in enumerate(self.current_population):
            try:
                # Make sure the source is a string
                if not isinstance(source, str):
                    logger.warning(f"Invalid source type: {type(source)}")
                    functions.append(None)
                    continue
                    
                # Try to fix syntax errors in the source
                if not CodeMutation.validate_python_syntax(source):
                    fixed_source = CodeMutation.fix_incomplete_blocks(source)
                    if CodeMutation.validate_python_syntax(fixed_source):
                        source = fixed_source
                        # Update the population with the fixed source
                        self.current_population[i] = fixed_source
                    else:
                        logger.warning(f"Source {i} contains unfixable syntax errors")
                        functions.append(None)
                        continue
                
                # Convert to function
                func = CodeMutation.source_to_function(source, self.function_globals)
                
                if func is not None:
                    functions.append(func)
                    valid_sources.append(source)
                else:
                    logger.warning(f"Could not create function from source {i}")
                    functions.append(None)
            except Exception as e:
                logger.warning(f"Error converting source {i} to function: {str(e)}")
                functions.append(None)
        
        # Use sequential evaluation instead of parallel to avoid pickling issues
        for i, func in enumerate(functions):
            try:
                if func is None:
                    # Use the lowest possible fitness for invalid functions
                    fitness_scores.append(float('-inf'))
                    continue
                    
                # Evaluate the function
                fitness = self.fitness_evaluator.evaluate(func)
                
                # Ensure the fitness is a valid number
                if fitness is None or not isinstance(fitness, (int, float)) or (isinstance(fitness, float) and not math.isfinite(fitness)):
                    logger.warning(f"Invalid fitness value for function {i}: {fitness}")
                    fitness = float('-inf')
                
                fitness_scores.append(fitness)
            except Exception as e:
                logger.warning(f"Error evaluating function {i}: {str(e)}")
                fitness_scores.append(float('-inf'))
        
        # Update the population to only include valid sources if we have too many invalid ones
        valid_count = sum(1 for f in fitness_scores if f > float('-inf'))
        if valid_count < len(self.current_population) // 2:
            logger.warning(f"Too many invalid functions: {len(self.current_population) - valid_count}. Keeping only valid ones.")
            
            # Keep only valid sources and their fitness scores
            valid_population = []
            valid_fitness = []
            
            for source, func, fitness in zip(self.current_population, functions, fitness_scores):
                if func is not None and fitness > float('-inf'):
                    valid_population.append(source)
                    valid_fitness.append(fitness)
            
            # Make sure we still have some valid functions
            if valid_population:
                self.current_population = valid_population
                fitness_scores = valid_fitness
                
                # Fill back up to population size if needed
                while len(self.current_population) < self.population_size:
                    idx = random.randrange(len(valid_population))
                    self.current_population.append(valid_population[idx])
                    fitness_scores.append(valid_fitness[idx])
                
                logger.info(f"Reduced population to {len(self.current_population)} valid individuals")
            
        return fitness_scores
    
    def _create_next_generation(self, fitness_scores: List[float]):
        """Create the next generation of function variants."""
        # Check if we have valid fitness scores
        if not fitness_scores or all(score == float('-inf') for score in fitness_scores):
            logger.warning("No valid fitness scores. Keeping current population with small mutations.")
            # Try to create variations of the original function instead
            try:
                original_mutations = CodeMutation.mutate_function(self.initial_source, self.mutation_rate * 0.5)
                valid_mutations = []
                for mutation in original_mutations:
                    if CodeMutation.validate_python_syntax(mutation):
                        # Verify that we can create a function from it
                        func = CodeMutation.source_to_function(mutation, self.function_globals)
                        if func is not None:
                            valid_mutations.append(mutation)
                
                if valid_mutations:
                    # Replace population with these new mutations
                    self.current_population = valid_mutations[:self.population_size]
                    if len(self.current_population) < self.population_size:
                        # Fill the rest with the original source
                        while len(self.current_population) < self.population_size:
                            self.current_population.append(self.initial_source)
                    return
            except Exception as e:
                logger.warning(f"Error creating fallback mutations: {str(e)}")
            
            # If everything else fails, keep the initial source
            self.current_population = [self.initial_source] * self.population_size
            return
            
        # Sort the current population by fitness
        # Remove -inf fitness values
        valid_pairs = [(score, source) for score, source in zip(fitness_scores, self.current_population) 
                       if score > float('-inf')]
        
        # If we have no valid pairs, resort to using original function
        if not valid_pairs:
            logger.warning("No valid fitness scores after filtering. Reverting to original function.")
            self.current_population = [self.initial_source] * self.population_size
            return
        
        # Sort by fitness (descending)
        valid_pairs.sort(reverse=True, key=lambda pair: pair[0])
        
        # Get the best source and its fitness
        best_fitness, best_source = valid_pairs[0]
        
        # Update best function if needed
        if (not hasattr(self, 'best_fitness') or 
            not hasattr(self, 'best_function') or 
            best_fitness > self.best_fitness):
            self.best_fitness = best_fitness
            self.best_source = best_source
            self.best_function = CodeMutation.source_to_function(best_source, self.function_globals)
            logger.info(f"New best fitness: {best_fitness}")
        
        # Extract sorted source code
        sorted_population = [pair[1] for pair in valid_pairs]
        
        # Keep the elite performers
        elite_size = min(self.elite_size, len(sorted_population))
        next_generation = sorted_population[:elite_size]
        
        # Create the rest of the population through selection, mutation and crossover
        remaining_to_fill = self.population_size - len(next_generation)
        
        # Split between mutation and crossover based on population size
        mutation_count = int(remaining_to_fill * 0.7)  # 70% mutation
        crossover_count = remaining_to_fill - mutation_count  # 30% crossover
        
        # Create new variants through mutation
        mutation_candidates = []
        max_attempts = mutation_count * 3  # Limit attempts
        attempts = 0
        
        while len(mutation_candidates) < mutation_count and attempts < max_attempts:
            attempts += 1
            try:
                # Select a parent using tournament selection
                parent = self._select_parent(sorted_population)
                
                # Generate mutations of this parent
                mutations = CodeMutation.mutate_function(parent, self.mutation_rate)
                
                # Filter and validate mutations
                for mutation in mutations:
                    # Skip if we have enough
                    if len(mutation_candidates) >= mutation_count:
                        break
                        
                    # Check that it's valid code
                    if not CodeMutation.validate_python_syntax(mutation):
                        fixed_mutation = CodeMutation.fix_incomplete_blocks(mutation)
                        if not CodeMutation.validate_python_syntax(fixed_mutation):
                            continue
                        mutation = fixed_mutation
                    
                    # Check that we can create a function from it
                    func = CodeMutation.source_to_function(mutation, self.function_globals)
                    if func is None:
                        continue
                        
                    mutation_candidates.append(mutation)
            except Exception as e:
                logger.warning(f"Error creating mutation: {str(e)}")
        
        # If we couldn't create enough mutations, fill with copies of good individuals
        while len(mutation_candidates) < mutation_count:
            if sorted_population:
                # Pick a random good individual
                mutation_candidates.append(random.choice(sorted_population[:max(elite_size*2, len(sorted_population))]))
            else:
                # Last resort - use original source
                mutation_candidates.append(self.initial_source)
        
        # Add mutations to next generation
        next_generation.extend(mutation_candidates)
        
        # For crossover, we'll use the best individuals
        # We need at least 2 parents for crossover
        if len(sorted_population) >= 2 and crossover_count > 0:
            crossover_candidates = []
            attempts = 0
            max_attempts = crossover_count * 3
            
            while len(crossover_candidates) < crossover_count and attempts < max_attempts:
                attempts += 1
                try:
                    # Select two different parents
                    parent1_idx = random.randint(0, min(10, len(sorted_population) - 1))
                    parent2_idx = random.randint(0, min(10, len(sorted_population) - 1))
                    
                    # Ensure they're different
                    while parent2_idx == parent1_idx and len(sorted_population) > 1:
                        parent2_idx = random.randint(0, min(10, len(sorted_population) - 1))
                        
                    parent1 = sorted_population[parent1_idx]
                    parent2 = sorted_population[parent2_idx]
                    
                    # Convert sources to functions for crossover
                    func1 = CodeMutation.source_to_function(parent1, self.function_globals)
                    func2 = CodeMutation.source_to_function(parent2, self.function_globals)
                    
                    if func1 and func2:
                        # Perform crossover
                        children_funcs = CodeMutation.crossover(func1, func2, 0.7)
                        
                        # Convert back to sources
                        for child_func in children_funcs:
                            try:
                                child_source = inspect.getsource(child_func)
                                
                                # Validate the child
                                if CodeMutation.validate_python_syntax(child_source):
                                    # Add to candidates
                                    crossover_candidates.append(child_source)
                                    
                                    # Check if we have enough
                                    if len(crossover_candidates) >= crossover_count:
                                        break
                            except Exception as e:
                                logger.warning(f"Error getting source for crossover child: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error performing crossover: {str(e)}")
            
            # If we couldn't create enough through crossover, fill with mutations or copies
            while len(crossover_candidates) < crossover_count:
                if sorted_population:
                    crossover_candidates.append(random.choice(sorted_population[:max(elite_size*2, len(sorted_population))]))
                else:
                    crossover_candidates.append(self.initial_source)
            
            # Add crossover results to next generation
            next_generation.extend(crossover_candidates)
        else:
            # If we can't do crossover, fill with more mutations
            while len(next_generation) < self.population_size:
                if sorted_population:
                    next_generation.append(random.choice(sorted_population))
                else:
                    next_generation.append(self.initial_source)
        
        # Update current population
        self.current_population = next_generation[:self.population_size]
        
        # Ensure we have exactly population_size individuals
        while len(self.current_population) < self.population_size:
            self.current_population.append(self.initial_source)
        
        logger.info(f"Created next generation with {len(self.current_population)} individuals")
    
    def _select_parent(self, sorted_population: List[str]) -> str:
        """
        Select a parent from the population using tournament selection.
        
        Args:
            sorted_population: Population sorted by fitness (best first)
            
        Returns:
            Selected parent source code
        """
        tournament_size = min(3, len(sorted_population))
        
        # Bias selection toward better individuals
        if random.random() < 0.7 and len(sorted_population) > tournament_size:
            # Select from top performers (top 30%)
            top_count = max(tournament_size, int(len(sorted_population) * 0.3))
            return random.choice(sorted_population[:top_count])
        else:
            # Tournament selection
            tournament = random.sample(sorted_population, tournament_size)
            return tournament[0]  # Return the first (best) from the tournament
    
    def get_stats(self) -> List[Dict[str, Any]]:
        """Get statistics about the evolution process."""
        return self.generation_stats 