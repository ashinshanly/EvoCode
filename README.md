# EvoCode: Evolutionary Code Generation Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-red.svg)]()

EvoCode is a Python library that uses evolutionary algorithms to evolve and optimize code. By applying genetic programming techniques to Python functions, EvoCode can automatically improve code for better performance, accuracy, or other metrics.

## Features

- **AST-Based Mutations**: Intelligently mutate code at the Abstract Syntax Tree level
- **Multiple Optimization Metrics**: Optimize for speed, accuracy, or combined metrics
- **Parallel Evaluation**: Evaluate function variants in parallel using multiprocessing
- **Visualization Tools**: Visualize the evolution process and results
- **Robust Error Handling**: Gracefully handle exceptions and timeouts during evolution

## Installation

### From PyPI (To do)

```bash
pip install evocode
```

### From Source

```bash
git clone https://github.com/evocode-team/evocode.git
cd evocode
pip install -e .
```

## Quick Start

Here's a simple example of using EvoCode to optimize a function:

```python
from evocode import EvoCode
from evocode.utils import print_function_diff

# Define a function to optimize
def add_numbers(a, b):
    return a + b

# Define test cases
test_cases = [
    {"args": [1, 2], "expected": 3},
    {"args": [100, 200], "expected": 300},
    {"args": [0.1, 0.2], "expected": 0.3},  # Floating point case that might be improved
]

# Evolve the function for better accuracy
evolved_func = EvoCode.evolve_function(
    func=add_numbers,
    test_cases=test_cases,
    optimization_metric="accuracy",
    generations=10,
    population_size=15
)

# Compare the original and evolved functions
print_function_diff(add_numbers, evolved_func)
```

## Examples

More examples can be found in the `/examples` directory:

- **Sorting Optimization**: Evolve a sorting algorithm for better performance
- **Numerical Accuracy**: Improve numerical precision in calculations
- **Algorithm Transformation**: Transform algorithms to equivalent but more efficient versions

## Documentation

For detailed documentation, visit [https://evocode.readthedocs.io](https://evocode.readthedocs.io) (To do)

## How It Works

EvoCode uses genetic programming techniques to evolve code:

1. **Initialization**: Start with an initial function and create a population of variants
2. **Evaluation**: Test each variant against a set of test cases
3. **Selection**: Select the best performers based on fitness scores
4. **Mutation**: Apply code mutations to selected variants
5. **Crossover**: Combine code from different successful variants (optional)
6. **Repeat**: Iterate the process for multiple generations

Mutations are applied at the AST (Abstract Syntax Tree) level, allowing for semantically aware code transformations that maintain the basic functionality while improving performance or other metrics.

## Use Cases

- **Performance Optimization**: Optimize functions for speed
- **Numerical Accuracy**: Improve precision in numerical calculations
- **Code Refactoring**: Transform code to equivalent but cleaner or more efficient forms
- **Research**: Explore evolutionary computation and genetic programming

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project was inspired by research in genetic programming and evolutionary algorithms
- Special thanks to the contributors of Python's AST module 
