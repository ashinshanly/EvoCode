"""
Visualization tools for the EvoCode library.
"""

import matplotlib.pyplot as plt
import networkx as nx
import ast
import inspect
from typing import Callable, List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger("evocode")


class FunctionVisualizer:
    """Tools for visualizing functions and their evolution."""
    
    @staticmethod
    def plot_function_ast(func: Callable, save_path: Optional[str] = None):
        """
        Visualize a function's AST as a network graph.
        
        Args:
            func: Function to visualize
            save_path: Optional path to save the plot image
        """
        try:
            # Get function source
            source = inspect.getsource(func)
            
            # Parse into AST
            tree = ast.parse(source)
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Node counter for unique IDs
            counter = [0]
            
            def add_nodes(node, parent_id=None):
                """Recursively add nodes and edges to the graph."""
                node_id = counter[0]
                counter[0] += 1
                
                # Add node
                node_name = node.__class__.__name__
                if hasattr(node, 'name'):
                    node_name += f"\n{node.name}"
                elif hasattr(node, 'id'):
                    node_name += f"\n{node.id}"
                elif hasattr(node, 'value') and isinstance(node.value, (int, float, str, bool)):
                    node_name += f"\n{node.value}"
                
                G.add_node(node_id, label=node_name)
                
                # Add edge from parent
                if parent_id is not None:
                    G.add_edge(parent_id, node_id)
                
                # Recursively process children
                for field, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, ast.AST):
                                add_nodes(item, node_id)
                    elif isinstance(value, ast.AST):
                        add_nodes(value, node_id)
                
                return node_id
            
            # Build the graph
            root_id = add_nodes(tree)
            
            # Draw the graph
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=False, node_size=300, node_color='skyblue', 
                    font_size=10, arrows=True)
            
            # Add node labels
            node_labels = {n: G.nodes[n]['label'] for n in G.nodes}
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
            
            plt.title(f"AST for {func.__name__}")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing function AST: {str(e)}")
    
    @staticmethod
    def plot_evolution_fitness(stats: List[Dict[str, Any]], save_path: Optional[str] = None):
        """
        Plot the fitness progression during evolution.
        
        Args:
            stats: Evolution statistics from EvolutionEngine.get_stats()
            save_path: Optional path to save the plot image
        """
        try:
            generations = [s["generation"] for s in stats]
            max_fitness = [s["max_fitness"] for s in stats]
            avg_fitness = [s["avg_fitness"] for s in stats]
            
            plt.figure(figsize=(10, 6))
            
            # Plot fitness curves
            plt.subplot(1, 1, 1)
            plt.plot(generations, max_fitness, 'b-', linewidth=2, label='Best Fitness')
            plt.plot(generations, avg_fitness, 'r--', linewidth=1.5, label='Average Fitness')
            
            # Add trend line for max fitness
            z = np.polyfit(generations, max_fitness, 1)
            p = np.poly1d(z)
            plt.plot(generations, p(generations), "g-.", 
                     linewidth=1, label=f'Trend: {z[0]:.6f}x + {z[1]:.6f}')
            
            plt.xlabel('Generation')
            plt.ylabel('Fitness Score')
            plt.title('Evolution Fitness Progress')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting evolution fitness: {str(e)}")
    
    @staticmethod
    def plot_mutation_distribution(mutations: List[Callable], original_func: Callable, 
                                  save_path: Optional[str] = None):
        """
        Visualize the distribution of mutations from the original function.
        
        Args:
            mutations: List of mutated functions
            original_func: Original function
            save_path: Optional path to save the plot image
        """
        try:
            # Calculate a similarity score between original and each mutation
            # using the Levenshtein distance on source code
            import Levenshtein  # Make sure to include in requirements
            
            original_source = inspect.getsource(original_func)
            distances = []
            
            for func in mutations:
                try:
                    mutated_source = inspect.getsource(func)
                    distance = Levenshtein.distance(original_source, mutated_source)
                    normalized_distance = distance / len(original_source)
                    distances.append(normalized_distance)
                except Exception:
                    # Skip functions we can't get source for
                    continue
            
            if not distances:
                logger.warning("No valid mutations for distribution plot")
                return
                
            # Create the distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(distances, bins=10, color='skyblue', edgecolor='black')
            plt.axvline(x=np.mean(distances), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(distances):.4f}')
            
            plt.xlabel('Normalized Edit Distance from Original')
            plt.ylabel('Count')
            plt.title('Distribution of Mutation Distances')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            logger.error("python-Levenshtein package is required for mutation distribution plots")
        except Exception as e:
            logger.error(f"Error plotting mutation distribution: {str(e)}")
    
    @staticmethod
    def compare_performance(original_func: Callable, evolved_func: Callable, 
                           test_cases: List[Dict[str, Any]], save_path: Optional[str] = None):
        """
        Visualize performance comparison between original and evolved functions.
        
        Args:
            original_func: Original function
            evolved_func: Evolved function
            test_cases: Test cases to compare with
            save_path: Optional path to save the plot image
        """
        try:
            import time
            
            original_times = []
            evolved_times = []
            labels = []
            
            for i, test_case in enumerate(test_cases):
                args = test_case.get("args", [])
                kwargs = test_case.get("kwargs", {})
                
                # Time original function
                start = time.perf_counter()
                original_func(*args, **kwargs)
                original_time = time.perf_counter() - start
                
                # Time evolved function
                start = time.perf_counter()
                evolved_func(*args, **kwargs)
                evolved_time = time.perf_counter() - start
                
                original_times.append(original_time)
                evolved_times.append(evolved_time)
                labels.append(f"Case {i+1}")
            
            # Create the comparison plot
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax = plt.subplot(1, 2, 1)
            ax.bar(x - width/2, original_times, width, label='Original', color='skyblue')
            ax.bar(x + width/2, evolved_times, width, label='Evolved', color='salmon')
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_ylabel('Execution Time (s)')
            ax.set_title('Performance Comparison by Test Case')
            ax.legend()
            
            # Add improvement percentage summary
            improvements = [(o - e) / o * 100 for o, e in zip(original_times, evolved_times)]
            avg_improvement = np.mean(improvements)
            
            ax2 = plt.subplot(1, 2, 2)
            ax2.pie([len([i for i in improvements if i > 0]), 
                    len([i for i in improvements if i <= 0])],
                   labels=['Improved', 'Not Improved'],
                   autopct='%1.1f%%',
                   colors=['lightgreen', 'lightcoral'])
            ax2.set_title(f'Average Improvement: {avg_improvement:.2f}%')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error comparing performance: {str(e)}")


# Additional visualization functions can be added here as needed 