Coding Standards
================

This document outlines the coding standards and best practices for GeminiSDR development.

Python Code Style
-----------------

PEP 8 Compliance
~~~~~~~~~~~~~~~

All Python code must follow PEP 8 guidelines with the following specific requirements:

* **Line Length**: Maximum 88 characters (Black formatter default)
* **Indentation**: 4 spaces (no tabs)
* **Imports**: Organized using isort with the following order:
  
  1. Standard library imports
  2. Third-party imports
  3. Local application imports

* **Naming Conventions**:
  
  * Classes: ``PascalCase`` (e.g., ``ConfigManager``)
  * Functions and variables: ``snake_case`` (e.g., ``load_config``)
  * Constants: ``UPPER_SNAKE_CASE`` (e.g., ``DEFAULT_BATCH_SIZE``)
  * Private methods: ``_leading_underscore`` (e.g., ``_validate_config``)

Code Formatting
~~~~~~~~~~~~~~~

Use the following tools for consistent code formatting:

.. code-block:: bash

   # Install formatting tools
   pip install black isort flake8 mypy
   
   # Format code
   black geminisdr/ tests/
   isort geminisdr/ tests/
   
   # Check style
   flake8 geminisdr/ tests/
   mypy geminisdr/

Configuration for tools:

.. code-block:: ini

   # setup.cfg
   [flake8]
   max-line-length = 88
   extend-ignore = E203, W503
   exclude = .git,__pycache__,build,dist,.venv
   
   [isort]
   profile = black
   multi_line_output = 3
   line_length = 88
   
   [mypy]
   python_version = 3.9
   warn_return_any = True
   warn_unused_configs = True
   disallow_untyped_defs = True

Type Hints
----------

All public functions and methods must include type hints:

.. code-block:: python

   from typing import Dict, List, Optional, Union, Any
   from pathlib import Path
   
   def load_config(
       config_path: Path,
       environment: Optional[str] = None,
       overrides: Optional[Dict[str, Any]] = None
   ) -> SystemConfig:
       """Load system configuration with optional overrides.
       
       Args:
           config_path: Path to the main configuration file
           environment: Environment name for specific overrides
           overrides: Dictionary of configuration overrides
           
       Returns:
           Validated system configuration object
           
       Raises:
           ConfigurationError: If configuration is invalid
           FileNotFoundError: If configuration file is missing
       """
       pass

Documentation Standards
-----------------------

Docstring Format
~~~~~~~~~~~~~~~

Use Google-style docstrings for all public functions, classes, and methods:

.. code-block:: python

   class ModelManager:
       """Centralized model management with versioning and MLflow integration.
       
       The ModelManager provides comprehensive model lifecycle management including
       saving, loading, versioning, and metadata tracking. It integrates with MLflow
       for experiment tracking and provides automatic model validation.
       
       Attributes:
           config: System configuration object
           models_dir: Directory for model storage
           registry: Model registry for version tracking
           
       Example:
           >>> manager = ModelManager(config)
           >>> metadata = ModelMetadata(name="my_model", version="1.0.0")
           >>> model_path = manager.save_model(model, metadata)
           >>> loaded_model, metadata = manager.load_model("my_model", "1.0.0")
       """
       
       def save_model(
           self,
           model: Any,
           metadata: ModelMetadata,
           model_path: Optional[str] = None
       ) -> str:
           """Save model with comprehensive metadata.
           
           Saves the model to disk with complete metadata including performance
           metrics, hyperparameters, and reproducibility information. The model
           is registered with MLflow for experiment tracking.
           
           Args:
               model: The model object to save
               metadata: Complete model metadata
               model_path: Optional custom path for model storage
               
           Returns:
               Path to the saved model file
               
           Raises:
               ModelError: If model saving fails
               ValidationError: If metadata is invalid
               
           Example:
               >>> metadata = ModelMetadata(
               ...     name="classifier",
               ...     version="1.0.0",
               ...     performance_metrics={"accuracy": 0.95}
               ... )
               >>> path = manager.save_model(my_model, metadata)
           """
           pass

Class Documentation
~~~~~~~~~~~~~~~~~~

All classes must include:

* **Purpose**: Clear description of the class's responsibility
* **Attributes**: Documentation of public attributes
* **Usage Examples**: Code examples showing typical usage
* **Related Classes**: References to related components

Error Handling Standards
------------------------

Exception Hierarchy
~~~~~~~~~~~~~~~~~~

Use the established exception hierarchy:

.. code-block:: python

   from geminisdr.core.error_handling import (
       GeminiSDRError,
       HardwareError,
       ConfigurationError,
       ModelError,
       MemoryError
   )
   
   def load_model(model_path: str) -> Any:
       """Load model with proper error handling."""
       try:
           if not Path(model_path).exists():
               raise ModelError(
                   f"Model file not found: {model_path}",
                   severity=ErrorSeverity.HIGH,
                   context={"model_path": model_path, "operation": "load"}
               )
           
           # Load model logic here
           return model
           
       except Exception as e:
           if isinstance(e, GeminiSDRError):
               raise
           else:
               raise ModelError(
                   f"Unexpected error loading model: {str(e)}",
                   severity=ErrorSeverity.MEDIUM,
                   context={"model_path": model_path, "original_error": str(e)}
               ) from e

Error Context
~~~~~~~~~~~~

Always provide meaningful error context:

.. code-block:: python

   # Good: Detailed context
   raise ConfigurationError(
       "Invalid batch size configuration",
       severity=ErrorSeverity.MEDIUM,
       context={
           "provided_value": batch_size,
           "valid_range": "1-1024",
           "config_section": "ml.batch_size",
           "platform": platform_info
       }
   )
   
   # Bad: Minimal context
   raise ValueError("Invalid batch size")

Testing Standards
-----------------

Test Organization
~~~~~~~~~~~~~~~~

Organize tests following the source code structure:

.. code-block:: text

   tests/
   ├── unit/                    # Unit tests
   │   ├── config/
   │   │   ├── test_config_manager.py
   │   │   └── test_config_models.py
   │   └── core/
   │       ├── test_error_handling.py
   │       └── test_memory_manager.py
   ├── integration/             # Integration tests
   │   ├── test_model_lifecycle.py
   │   └── test_cross_platform.py
   └── performance/             # Performance tests
       ├── test_training_speed.py
       └── test_memory_usage.py

Test Naming
~~~~~~~~~~~

Use descriptive test names that explain the scenario:

.. code-block:: python

   class TestConfigManager:
       """Test cases for ConfigManager class."""
       
       def test_load_config_with_valid_file_returns_config_object(self):
           """Test that loading a valid config file returns SystemConfig."""
           pass
       
       def test_load_config_with_missing_file_raises_configuration_error(self):
           """Test that missing config file raises ConfigurationError."""
           pass
       
       def test_load_config_with_invalid_yaml_raises_configuration_error(self):
           """Test that invalid YAML syntax raises ConfigurationError."""
           pass

Test Structure
~~~~~~~~~~~~~

Follow the Arrange-Act-Assert pattern:

.. code-block:: python

   def test_memory_manager_optimizes_batch_size_when_memory_limited(self):
       """Test batch size optimization under memory pressure."""
       # Arrange
       config = create_test_config()
       memory_manager = MemoryManager(config)
       base_batch_size = 64
       available_memory_mb = 512  # Limited memory
       
       # Mock memory stats to simulate low memory
       mock_stats = MemoryStats(
           total_ram_mb=1024,
           available_ram_mb=available_memory_mb,
           used_ram_mb=512
       )
       memory_manager.get_memory_stats = Mock(return_value=mock_stats)
       
       # Act
       optimized_batch_size = memory_manager.optimize_batch_size(
           base_batch_size, model_size_mb=200
       )
       
       # Assert
       assert optimized_batch_size < base_batch_size
       assert optimized_batch_size > 0
       assert optimized_batch_size <= 32  # Expected maximum for limited memory

Performance Standards
---------------------

Memory Management
~~~~~~~~~~~~~~~~

Follow these guidelines for efficient memory usage:

.. code-block:: python

   # Use context managers for resource management
   with memory_manager.memory_efficient_context():
       # Memory-intensive operations
       results = process_large_dataset(data)
   
   # Explicit cleanup for large objects
   del large_tensor
   torch.cuda.empty_cache()  # For GPU memory
   gc.collect()  # Force garbage collection
   
   # Use generators for large datasets
   def process_batches(dataset):
       for batch in dataset:
           yield process_batch(batch)
           # Batch is automatically cleaned up

Async/Await Usage
~~~~~~~~~~~~~~~~

Use async/await for I/O-bound operations:

.. code-block:: python

   import asyncio
   from typing import List
   
   async def load_models_async(model_paths: List[str]) -> List[Any]:
       """Load multiple models asynchronously."""
       tasks = [load_model_async(path) for path in model_paths]
       return await asyncio.gather(*tasks)
   
   async def load_model_async(model_path: str) -> Any:
       """Load a single model asynchronously."""
       loop = asyncio.get_event_loop()
       return await loop.run_in_executor(None, load_model_sync, model_path)

Configuration Standards
-----------------------

Configuration Files
~~~~~~~~~~~~~~~~~~

Use YAML for configuration with clear structure:

.. code-block:: yaml

   # Good: Clear, hierarchical structure
   hardware:
     device_preference: "auto"  # auto, cpu, mps, cuda
     memory_limit_gb: null     # null for auto-detection
     sdr_mode: "auto"          # auto, hardware, simulation
   
   ml:
     batch_size: null          # null for auto-optimization
     learning_rate: 1e-4
     model_cache_size: 3
     precision: "float32"      # float32, float16, mixed
   
   logging:
     level: "INFO"             # DEBUG, INFO, WARNING, ERROR
     format: "structured"      # structured, simple
     rotation: "daily"         # daily, weekly, size-based

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~

Always validate configuration with clear error messages:

.. code-block:: python

   def validate_ml_config(config: MLConfig) -> List[str]:
       """Validate ML configuration parameters."""
       errors = []
       
       if config.batch_size is not None:
           if config.batch_size < 1 or config.batch_size > 1024:
               errors.append(
                   f"batch_size must be between 1 and 1024, got {config.batch_size}"
               )
       
       if config.learning_rate <= 0 or config.learning_rate > 1:
           errors.append(
               f"learning_rate must be between 0 and 1, got {config.learning_rate}"
           )
       
       if config.model_cache_size < 1:
           errors.append(
               f"model_cache_size must be at least 1, got {config.model_cache_size}"
           )
       
       return errors

Git Workflow
------------

Branch Naming
~~~~~~~~~~~~

Use descriptive branch names with prefixes:

* ``feature/add-model-versioning``
* ``bugfix/fix-memory-leak``
* ``refactor/improve-error-handling``
* ``docs/update-api-documentation``

Commit Messages
~~~~~~~~~~~~~~

Follow conventional commit format:

.. code-block:: text

   feat: add model versioning with MLflow integration
   
   - Implement ModelManager class with save/load functionality
   - Add comprehensive metadata tracking
   - Integrate with MLflow for experiment tracking
   - Include model compatibility validation
   
   Closes #123

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~

All pull requests must include:

* **Clear Description**: Explain what changes were made and why
* **Testing**: Evidence that changes have been tested
* **Documentation**: Updates to documentation if needed
* **Breaking Changes**: Clear indication of any breaking changes

Code Review Checklist
~~~~~~~~~~~~~~~~~~~~~

Reviewers should check:

* [ ] Code follows style guidelines
* [ ] All functions have type hints and docstrings
* [ ] Tests are included and pass
* [ ] Error handling is appropriate
* [ ] Performance implications are considered
* [ ] Documentation is updated
* [ ] No security vulnerabilities introduced