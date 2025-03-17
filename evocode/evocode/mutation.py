"""
Mutation strategies for the EvoCode library.
"""

import ast
import copy
import random
import inspect
import logging
import types
import re
import traceback
from typing import Callable, List, Dict, Any, Union, Optional, Tuple

logger = logging.getLogger("evocode")


class ASTMutationVisitor(ast.NodeTransformer):
    """AST visitor that applies random mutations to code."""
    
    def __init__(self, mutation_rate=0.2):
        self.mutation_rate = mutation_rate
        self.mutation_count = 0
        self.max_mutations = 10  # Limit mutations to avoid excessive changes
        
    def visit_BinOp(self, node):
        """Potentially mutate binary operations."""
        # Visit children first
        self.generic_visit(node)
        
        # Check if we've reached the mutation limit
        if self.mutation_count >= self.max_mutations:
            return node
        
        # Randomly decide whether to mutate
        if random.random() < self.mutation_rate:
            # Possible operation replacements
            op_mappings = {
                ast.Add: [ast.Sub, ast.Mult, ast.Div],
                ast.Sub: [ast.Add, ast.Mult, ast.Div],
                ast.Mult: [ast.Add, ast.Sub, ast.Div],
                ast.Div: [ast.Add, ast.Sub, ast.Mult],
                ast.FloorDiv: [ast.Div],
                ast.Mod: [ast.Mult, ast.Div],
                ast.Pow: [ast.Mult],
            }
            
            # Check if we can mutate this operation
            current_op_type = type(node.op)
            if current_op_type in op_mappings:
                # Choose a random replacement operation
                new_op_type = random.choice(op_mappings[current_op_type])
                node.op = new_op_type()
                self.mutation_count += 1
                logger.debug(f"Mutated binary operation: {current_op_type.__name__} -> {new_op_type.__name__}")
        
        return node
    
    def visit_Compare(self, node):
        """Potentially mutate comparison operations."""
        # Visit children first
        self.generic_visit(node)
        
        # Check if we've reached the mutation limit
        if self.mutation_count >= self.max_mutations:
            return node
        
        # Randomly decide whether to mutate
        if random.random() < self.mutation_rate:
            # Possible comparison replacements
            comp_mappings = {
                ast.Eq: [ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
                ast.NotEq: [ast.Eq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
                ast.Lt: [ast.LtE, ast.Gt, ast.GtE, ast.Eq],
                ast.LtE: [ast.Lt, ast.Gt, ast.GtE, ast.Eq],
                ast.Gt: [ast.GtE, ast.Lt, ast.LtE, ast.Eq],
                ast.GtE: [ast.Gt, ast.Lt, ast.LtE, ast.Eq],
            }
            
            # Mutate each comparator with some probability
            for i, cmp_op in enumerate(node.ops):
                current_op_type = type(cmp_op)
                if current_op_type in comp_mappings and random.random() < self.mutation_rate:
                    new_op_type = random.choice(comp_mappings[current_op_type])
                    node.ops[i] = new_op_type()
                    self.mutation_count += 1
                    logger.debug(f"Mutated comparison: {current_op_type.__name__} -> {new_op_type.__name__}")
        
        return node
    
    def visit_If(self, node):
        """Potentially mutate if statements."""
        # Visit children first
        self.generic_visit(node)
        
        # Check if we've reached the mutation limit
        if self.mutation_count >= self.max_mutations:
            return node
        
        # Randomly decide whether to mutate
        if random.random() < self.mutation_rate:
            # Possible mutations:
            # 1. Invert the condition (add "not")
            # 2. Swap the body and orelse (if/else) blocks
            
            mutation_type = random.choice(["invert", "swap"])
            
            if mutation_type == "invert":
                # Invert the condition
                node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
                
                # Swap the blocks to maintain logic
                node.body, node.orelse = node.orelse, node.body
                self.mutation_count += 1
                logger.debug("Inverted if condition")
            
            elif mutation_type == "swap" and node.orelse:
                # Simply swap if and else blocks
                node.body, node.orelse = node.orelse, node.body
                self.mutation_count += 1
                logger.debug("Swapped if/else blocks")
        
        return node
    
    def visit_List(self, node):
        """Potentially optimize list operations."""
        # Visit children first
        self.generic_visit(node)
        
        # Check if we've reached the mutation limit
        if self.mutation_count >= self.max_mutations:
            return node
        
        # Randomly decide whether to mutate
        if random.random() < self.mutation_rate:
            # Possibility: Convert to tuple for immutable data
            self.mutation_count += 1
            logger.debug("Converted list to tuple")
            return ast.Tuple(elts=node.elts, ctx=node.ctx)
        
        return node
    
    def visit_For(self, node):
        """Potentially optimize for loops."""
        # Visit children first
        self.generic_visit(node)
        
        # Check if we've reached the mutation limit
        if self.mutation_count >= self.max_mutations:
            return node
        
        # Randomly decide whether to mutate
        if random.random() < self.mutation_rate:
            # Look for opportunities to convert to list comprehension
            # This is a simplistic implementation - real one would be more complex
            
            # Only attempt for simple loops with a single append operation
            if (len(node.body) == 1 and 
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Call) and
                isinstance(node.body[0].value.func, ast.Attribute) and
                node.body[0].value.func.attr == 'append'):
                
                # Extract the target list and value being appended
                target_list = node.body[0].value.func.value
                append_value = node.body[0].value.args[0]
                
                # Create a list comprehension
                comp = ast.ListComp(
                    elt=append_value,
                    generators=[
                        ast.comprehension(
                            target=node.target,
                            iter=node.iter,
                            ifs=[],
                            is_async=0
                        )
                    ]
                )
                
                # Create an assignment to replace the for loop
                self.mutation_count += 1
                logger.debug("Converted for loop to list comprehension")
                return ast.Assign(
                    targets=[target_list],
                    value=comp
                )
        
        return node
    
    def visit_BoolOp(self, node):
        """Potentially mutate boolean operations."""
        # Visit children first
        self.generic_visit(node)
        
        # Check if we've reached the mutation limit
        if self.mutation_count >= self.max_mutations:
            return node
        
        # Randomly decide whether to mutate
        if random.random() < self.mutation_rate:
            # Possible operation replacements
            op_mappings = {
                ast.And: ast.Or,
                ast.Or: ast.And,
            }
            
            # Check if we can mutate this operation
            current_op_type = type(node.op)
            if current_op_type in op_mappings:
                # Replace the operation
                new_op_type = op_mappings[current_op_type]
                node.op = new_op_type()
                self.mutation_count += 1
                logger.debug(f"Mutated boolean operation: {current_op_type.__name__} -> {new_op_type.__name__}")
        
        return node


class CodeMutation:
    """Advanced implementation of code mutation strategies."""
    
    @staticmethod
    def parse_function(func_or_source: Union[Callable, str]) -> Tuple[ast.Module, Dict[str, Any]]:
        """
        Convert a Python function or source code to an AST representation.
        
        Args:
            func_or_source: Either a callable function or source code string
            
        Returns:
            Tuple of (AST module, globals dictionary)
        """
        try:
            if callable(func_or_source):
                # It's a callable function
                source = inspect.getsource(func_or_source)
                globals_dict = func_or_source.__globals__
            else:
                # It's source code
                source = func_or_source
                globals_dict = {}
                
            # Parse the source code into an AST
            tree = ast.parse(source)
            return tree, globals_dict
        except Exception as e:
            logger.error(f"Error parsing function: {str(e)}")
            # Return a minimal valid AST with empty function
            fallback_source = "def fallback_function():\n    return None"
            tree = ast.parse(fallback_source)
            return tree, {}
    
    @staticmethod
    def ast_to_source(node: ast.AST) -> str:
        """Convert AST back to source code."""
        try:
            # Make sure all nodes have line numbers before unparsing
            CodeMutation.ensure_node_lineno(node)
            return ast.unparse(node)
        except Exception as e:
            logger.error(f"Error converting AST to source: {str(e)}")
            # More descriptive error function
            return "def error_function():\n    # AST conversion error\n    # Original error: " + str(e) + "\n    return None"
    
    @staticmethod
    def validate_python_syntax(source: str) -> bool:
        """
        Validate that a string contains valid Python syntax.
        
        Args:
            source: Source code string to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(source)
            return True
        except SyntaxError as e:
            logger.warning(f"Invalid Python syntax: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"Error validating Python syntax: {str(e)}")
            return False
    
    @staticmethod
    def fix_incomplete_blocks(source: str) -> str:
        """
        Attempt to fix incomplete code blocks in Python source code.
        
        Args:
            source: Source code string to fix
            
        Returns:
            Fixed source code string
        """
        try:
            # Common pattern: if statements without indented blocks
            lines = source.split('\n')
            fixed_lines = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                fixed_lines.append(line)
                
                # Check for control structures that require indented blocks
                if re.search(r'^\s*(if|for|while|def|class|with|try|except|finally|elif|else)\s.*:\s*$', line) and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # Check if the next line has proper indentation
                    current_indent = len(line) - len(line.lstrip())
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    # If next line doesn't have increased indentation, add a pass statement
                    if next_indent <= current_indent:
                        # Only add pass if the next line isn't a valid block continuation
                        if not next_line.strip().startswith(('elif', 'else', 'except', 'finally')):
                            indent = ' ' * (current_indent + 4)
                            fixed_lines.append(f"{indent}pass")
                # Also check for the last line in the file
                elif re.search(r'^\s*(if|for|while|def|class|with|try|except|finally|elif|else)\s.*:\s*$', line) and i + 1 == len(lines):
                    # This is the last line and it's a control structure with no block
                    indent = ' ' * (len(line) - len(line.lstrip()) + 4)
                    fixed_lines.append(f"{indent}pass")
                
                i += 1
            
            return '\n'.join(fixed_lines)
        except Exception as e:
            logger.warning(f"Error fixing incomplete blocks: {str(e)}")
            return source  # Return original if fixing fails
    
    @staticmethod
    def source_to_function(source: str, globals_dict: Dict[str, Any] = None) -> Optional[Callable]:
        """
        Convert source code to a callable function.
        
        Args:
            source: Source code string
            globals_dict: Optional globals dictionary for the function
            
        Returns:
            Callable function or None if conversion fails
        """
        try:
            if globals_dict is None:
                globals_dict = {}
                
            # Add builtins to globals if not present
            if '__builtins__' not in globals_dict:
                globals_dict['__builtins__'] = __builtins__
            
            # Normalize source code - trim whitespace
            source = source.strip()
            
            # Make sure the source starts with "def" 
            if not source.lstrip().startswith("def "):
                logger.error("Source code does not define a function")
                return None
            
            # First, validate the syntax
            if not CodeMutation.validate_python_syntax(source):
                # Try to fix common syntax issues
                fixed_source = CodeMutation.fix_incomplete_blocks(source)
                
                # Validate the fixed source
                if not CodeMutation.validate_python_syntax(fixed_source):
                    logger.error("Could not fix syntax errors in source code")
                    return None
                
                source = fixed_source
                
            # Verify that the code actually defines a function
            try:
                tree = ast.parse(source)
                has_function_def = False
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        has_function_def = True
                        break
                
                if not has_function_def:
                    logger.error("AST parsing found no function definition")
                    return None
            except Exception as e:
                logger.error(f"AST validation error: {str(e)}")
                return None
                
            # Create a local namespace for the function
            local_namespace = {}
            
            # Execute the source code in the provided globals context
            try:
                exec(source, globals_dict, local_namespace)
            except Exception as e:
                logger.error(f"Error executing function source: {str(e)}")
                return None
            
            # Find the function in the local namespace
            for obj in local_namespace.values():
                if callable(obj) and isinstance(obj, types.FunctionType):
                    return obj
            
            # If we can't find it, try parsing to extract the name
            try:
                tree = ast.parse(source)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        if func_name in local_namespace:
                            return local_namespace[func_name]
            except Exception as e:
                logger.error(f"Error extracting function name: {str(e)}")
                
            # If we got here, we couldn't find the function
            logger.warning("Could not find function in compiled source")
            return None
            
        except Exception as e:
            logger.error(f"Error converting source to function: {str(e)}")
            return None
    
    @staticmethod
    def ensure_node_lineno(node):
        """
        Make sure all AST nodes have line numbers.
        This is a recursive function that sets line numbers for all nodes in the tree.
        
        Args:
            node: AST node to process
        """
        try:
            # Skip primitive types
            if not isinstance(node, ast.AST):
                return
            
            # Set line number and col_offset if missing
            if not hasattr(node, 'lineno') or node.lineno is None:
                node.lineno = 1
            if not hasattr(node, 'col_offset') or node.col_offset is None:
                node.col_offset = 0
            
            # Set end_lineno and end_col_offset if missing (for Python 3.8+)
            if not hasattr(node, 'end_lineno') or node.end_lineno is None:
                node.end_lineno = node.lineno
            if not hasattr(node, 'end_col_offset') or node.end_col_offset is None:
                node.end_col_offset = node.col_offset + 1
                
            # Process all child nodes
            for child in ast.iter_child_nodes(node):
                CodeMutation.ensure_node_lineno(child)
                
        except Exception as e:
            logger.error(f"Error ensuring line numbers: {str(e)}")
    
    @staticmethod
    def mutate_function(func_or_source: Union[Callable, str], mutation_rate: float = 0.2) -> List[str]:
        """
        Generate mutations of the given function using various strategies.
        
        Args:
            func_or_source: Either a callable function or source code string
            mutation_rate: Probability of mutation (between 0.0 and 1.0)
            
        Returns:
            List of mutated source code strings
        """
        # Validate the mutation rate
        mutation_rate = max(0.01, min(0.5, mutation_rate))
        
        try:
            # Parse the input into an AST
            tree, globals_dict = CodeMutation.parse_function(func_or_source)
            
            # Make sure all nodes have line numbers
            CodeMutation.ensure_node_lineno(tree)
            
            # Extract function name for logging
            func_name = "unknown"
            if callable(func_or_source):
                func_name = func_or_source.__name__
            else:
                try:
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef):
                            func_name = node.name
                            break
                except:
                    pass
                    
            logger.info(f"Mutating function: {func_name} with mutation rate {mutation_rate}")
            
            # Store source code mutations
            source_mutations = []
            
            # Extract original source for reference
            original_source = CodeMutation.ast_to_source(tree)
            
            # Generate multiple mutations using different strategies
            mutation_strategies = [
                # Core AST mutations are the most reliable
                (CodeMutation._apply_ast_mutations, 0.9),  # Higher weight
                (CodeMutation._apply_algorithmic_mutations, 0.6),  # Medium weight
                (CodeMutation._apply_type_conversion_mutations, 0.5),  # Medium weight
                (CodeMutation._apply_library_mutations, 0.3),  # Lower weight
            ]
            
            # Apply each strategy with its own probability
            for strategy, strategy_prob in mutation_strategies:
                # Apply strategy with probability
                if random.random() > strategy_prob:
                    continue
                    
                try:
                    # Apply the mutation strategy
                    mutated_asts = strategy(tree, mutation_rate)
                    
                    # Convert each mutated AST to source code
                    for mutated_ast in mutated_asts:
                        try:
                            # Ensure line numbers in the mutated AST
                            CodeMutation.ensure_node_lineno(mutated_ast)
                            
                            mutation_source = CodeMutation.ast_to_source(mutated_ast)
                            
                            # Skip if identical to original
                            if mutation_source == original_source:
                                continue
                                
                            # Only add valid mutations
                            if CodeMutation.validate_python_syntax(mutation_source):
                                source_mutations.append(mutation_source)
                                logger.debug(f"Generated valid mutation using {strategy.__name__}")
                            else:
                                # Try to fix it
                                fixed_source = CodeMutation.fix_incomplete_blocks(mutation_source)
                                if CodeMutation.validate_python_syntax(fixed_source) and fixed_source != original_source:
                                    source_mutations.append(fixed_source)
                                    logger.debug(f"Fixed and added mutation from {strategy.__name__}")
                        except Exception as e:
                            logger.warning(f"Error generating source for mutation: {str(e)}")
                            continue
                except Exception as e:
                    logger.warning(f"Mutation strategy {strategy.__name__} failed: {str(e)}")
                    continue
            
            # If we have no mutations, create some simple ones directly
            if not source_mutations:
                logger.warning(f"No mutations generated using strategies, creating fallback mutations")
                
                # Create some simple mutations
                for i in range(3):
                    # Extract the function definition (if any)
                    func_def = None
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef):
                            func_def = node
                            break
                    
                    if func_def:
                        # Make a copy of the function definition
                        new_func = copy.deepcopy(func_def)
                        
                        # Apply simple mutation based on index
                        if i == 0:
                            # Add a simple parameter check
                            param_check = ast.parse("if args is None or len(args) == 0:\n    return None").body[0]
                            new_func.body.insert(0, param_check)
                            logger.info("Created simple parameter check mutation")
                        elif i == 1:
                            # Add a simple debug print
                            debug_print = ast.parse(f"print('Debug: {func_name} called with', locals())").body[0]
                            new_func.body.insert(0, debug_print)
                            logger.info("Created debug print mutation")
                        elif i == 2:
                            # Add a simple result caching mechanism based on input args
                            cache_code = ast.parse(
                                "cache_key = str(args)\n"
                                "if hasattr(func, '_cache') and cache_key in func._cache:\n"
                                "    return func._cache[cache_key]\n"
                            ).body
                            for stmt in reversed(cache_code):
                                new_func.body.insert(0, stmt)
                            # Add store in cache at end of function
                            store_cache = ast.parse(
                                "if not hasattr(func, '_cache'):\n"
                                "    func._cache = {}\n"
                                "func._cache[cache_key] = result\n"
                            ).body
                            # Find return statements and insert cache before them
                            for i, node in enumerate(new_func.body):
                                if isinstance(node, ast.Return):
                                    # Add a result variable assignment before return
                                    result_assign = ast.parse("result = " + ast.unparse(node.value)).body[0]
                                    new_func.body[i:i+1] = [result_assign] + store_cache + [ast.Return(value=ast.Name(id='result', ctx=ast.Load()))]
                                    break
                            logger.info("Created result caching mutation")
                        
                        # Create a new module with just this function
                        new_module = ast.Module(body=[new_func], type_ignores=[])
                        
                        # Convert to source
                        try:
                            # Ensure line numbers
                            CodeMutation.ensure_node_lineno(new_module)
                            mutation_source = CodeMutation.ast_to_source(new_module)
                            
                            # Validate and add
                            if CodeMutation.validate_python_syntax(mutation_source) and mutation_source != original_source:
                                source_mutations.append(mutation_source)
                                logger.info(f"Added fallback mutation {i}")
                        except Exception as e:
                            logger.warning(f"Error creating fallback mutation {i}: {str(e)}")
            
            # Add the original as last resort if no mutations worked
            if not source_mutations:
                source_mutations.append(original_source)
                logger.warning("No valid mutations generated, using original source only")
            
            logger.info(f"Generated {len(source_mutations)} mutations for {func_name}")
            return source_mutations
            
        except Exception as e:
            logger.error(f"Error in mutation process: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return a basic function as fallback
            if callable(func_or_source):
                try:
                    original_source = inspect.getsource(func_or_source)
                    return [original_source]
                except Exception:
                    logger.error("Could not get source for callable function")
            
            # Ultimate fallback
            return ["def fallback_function():\n    # Mutation error occurred\n    return None"]
    
    @staticmethod
    def _apply_ast_mutations(tree: ast.Module, mutation_rate: float) -> List[ast.Module]:
        """Apply AST-based mutations using the visitor pattern."""
        results = []
        
        # Create several variants with different random mutations
        for _ in range(3):
            try:
                # Deep copy the AST to avoid modifying the original
                mutated_ast = copy.deepcopy(tree)
                
                # Create and apply a mutation visitor
                visitor = ASTMutationVisitor(mutation_rate=mutation_rate)
                mutated_ast = visitor.visit(mutated_ast)
                
                # Add to results if mutations were applied
                if visitor.mutation_count > 0:
                    results.append(mutated_ast)
            except Exception as e:
                logger.warning(f"AST mutation failed: {str(e)}")
                continue
        
        return results
    
    @staticmethod
    def _apply_type_conversion_mutations(tree: ast.Module, mutation_rate: float) -> List[ast.Module]:
        """Apply mutations that optimize for numeric types or precision."""
        if random.random() > mutation_rate:
            return []  # Skip this mutation type sometimes
            
        results = []
        
        # Deep copy the AST to avoid modifying the original
        mutated_ast = copy.deepcopy(tree)
        
        # Add imports for advanced numeric types
        decimal_import = ast.ImportFrom(
            module='decimal',
            names=[ast.alias(name='Decimal', asname=None)],
            level=0
        )
        
        # Wrap the function in a module with the import
        module = ast.Module(
            body=[decimal_import, mutated_ast],
            type_ignores=[]
        )
        
        # Add to results
        results.append(module)
        
        return results
    
    @staticmethod
    def _apply_algorithmic_mutations(tree: ast.Module, mutation_rate: float) -> List[ast.Module]:
        """Apply mutations that change algorithms or approaches."""
        if random.random() > mutation_rate:
            return []  # Skip this mutation type sometimes
            
        # This would normally contain sophisticated algorithm transformations
        # For now, we'll just return a copy of the original
        return [ast.Module(body=[copy.deepcopy(tree)], type_ignores=[])]
    
    @staticmethod
    def _apply_library_mutations(tree: ast.Module, mutation_rate: float) -> List[ast.Module]:
        """Apply mutations that introduce library functions."""
        if random.random() > mutation_rate:
            return []  # Skip this mutation type sometimes
            
        # This would normally introduce specialized library functions
        # For now, we'll add a basic optimization with a safer approach
        results = []
        
        # Deep copy the AST to avoid modifying the original
        mutated_ast = copy.deepcopy(tree)
        
        # Generate an improved version that uses manual caching instead of lru_cache
        # This avoids the pickling problems with functools.lru_cache
        try:
            # Find the first function definition
            func_node = None
            for node in mutated_ast.body:
                if isinstance(node, ast.FunctionDef):
                    func_node = node
                    break
                    
            if func_node:
                # Add a dictionary cache to the function
                cache_dict = ast.Assign(
                    targets=[ast.Name(id='_cache', ctx=ast.Store())],
                    value=ast.Dict(keys=[], values=[]),
                    lineno=func_node.lineno
                )
                
                # Create the body of the function with caching
                original_body = func_node.body.copy()
                
                # Create cache key generation
                args_tuple = ast.Tuple(
                    elts=[
                        ast.Name(id=arg.arg, ctx=ast.Load())
                        for arg in func_node.args.args
                    ],
                    ctx=ast.Load()
                )
                
                # Check if result is in cache
                cache_check = ast.If(
                    test=ast.Compare(
                        left=args_tuple,
                        ops=[ast.In()],
                        comparators=[ast.Name(id='_cache', ctx=ast.Load())]
                    ),
                    body=[
                        ast.Return(
                            value=ast.Subscript(
                                value=ast.Name(id='_cache', ctx=ast.Load()),
                                slice=ast.Index(value=args_tuple) if hasattr(ast, 'Index') else args_tuple,
                                ctx=ast.Load()
                            )
                        )
                    ],
                    orelse=[]
                )
                
                # Compute result
                result_var = ast.Assign(
                    targets=[ast.Name(id='result', ctx=ast.Store())],
                    value=ast.Constant(value=None)
                )
                
                # Assign result in cache
                cache_store = ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name(id='_cache', ctx=ast.Load()),
                            slice=ast.Index(value=args_tuple) if hasattr(ast, 'Index') else args_tuple,
                            ctx=ast.Store()
                        )
                    ],
                    value=ast.Name(id='result', ctx=ast.Load())
                )
                
                # Return result
                return_result = ast.Return(
                    value=ast.Name(id='result', ctx=ast.Load())
                )
                
                # Build the new body with try-except for safety
                try_body = [
                    cache_dict,
                    cache_check,
                    result_var
                ]
                
                # Add the original function body
                for stmt in original_body:
                    if isinstance(stmt, ast.Return):
                        # Replace return with result assignment
                        try_body.append(
                            ast.Assign(
                                targets=[ast.Name(id='result', ctx=ast.Store())],
                                value=stmt.value
                            )
                        )
                    else:
                        try_body.append(stmt)
                
                # Add cache storage and return
                try_body.extend([
                    cache_store,
                    return_result
                ])
                
                # Wrap in try-except
                func_node.body = [
                    ast.Try(
                        body=try_body,
                        handlers=[
                            ast.ExceptHandler(
                                type=ast.Name(id='Exception', ctx=ast.Load()),
                                name=None,
                                body=[
                                    # If anything fails, just run the original function
                                    *original_body
                                ]
                            )
                        ],
                        orelse=[],
                        finalbody=[]
                    )
                ]
                
                results.append(mutated_ast)
        except Exception as e:
            logger.warning(f"Error applying library mutation: {str(e)}")
        
        return results
    
    @staticmethod
    def crossover(func1: Callable, func2: Callable, crossover_rate: float = 0.7) -> List[Callable]:
        """Perform crossover between two functions to create offspring."""
        if random.random() > crossover_rate:
            return [func1, func2]  # Skip crossover sometimes
            
        try:
            # Parse both functions to AST
            ast1 = CodeMutation.parse_function(func1)
            ast2 = CodeMutation.parse_function(func2)
            
            # For now, implement a simple crossover: swap function bodies
            # In a real implementation, this would be more sophisticated
            
            # Create two offspring
            offspring1 = copy.deepcopy(ast1)
            offspring2 = copy.deepcopy(ast2)
            
            # Swap bodies (this is a simplistic approach)
            if len(ast1.body) > 0 and len(ast2.body) > 0:
                # Random crossover point
                crossover_point1 = random.randint(0, len(ast1.body) - 1)
                crossover_point2 = random.randint(0, len(ast2.body) - 1)
                
                # Swap portions of the function bodies
                offspring1.body[crossover_point1:], offspring2.body[crossover_point2:] = \
                    offspring2.body[crossover_point2:], offspring1.body[crossover_point1:]
            
            # Convert back to callable functions
            offspring_source1 = CodeMutation.ast_to_source(
                ast.Module(body=[offspring1], type_ignores=[]))
            offspring_source2 = CodeMutation.ast_to_source(
                ast.Module(body=[offspring2], type_ignores=[]))
            
            # Create new functions
            offspring_namespace1 = {}
            offspring_namespace2 = {}
            
            # Use the globals from the original functions
            exec(offspring_source1, func1.__globals__, offspring_namespace1)
            exec(offspring_source2, func2.__globals__, offspring_namespace2)
            
            result1 = list(offspring_namespace1.values())[0]
            result2 = list(offspring_namespace2.values())[0]
            
            return [result1, result2]
        except Exception as e:
            logger.warning(f"Crossover failed: {str(e)}")
            return [func1, func2]  # Return parents as fallback 