Error Handling
==============

.. automodule:: geminisdr.core.error_handling
   :members:
   :undoc-members:
   :show-inheritance:

Classes and Enums
-----------------

ErrorSeverity
~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.error_handling.ErrorSeverity
   :members:
   :undoc-members:
   :show-inheritance:

   Enumeration of error severity levels for categorizing system errors.

GeminiSDRError
~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.error_handling.GeminiSDRError
   :members:
   :undoc-members:
   :show-inheritance:

   Base exception class for all GeminiSDR system errors with severity and context tracking.

HardwareError
~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.error_handling.HardwareError
   :members:
   :undoc-members:
   :show-inheritance:

   Exception for hardware-related failures (SDR, GPU, device connectivity).

ConfigurationError
~~~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.error_handling.ConfigurationError
   :members:
   :undoc-members:
   :show-inheritance:

   Exception for configuration validation and loading errors.

ModelError
~~~~~~~~~~

.. autoclass:: geminisdr.core.error_handling.ModelError
   :members:
   :undoc-members:
   :show-inheritance:

   Exception for ML model loading, saving, and compatibility errors.

MemoryError
~~~~~~~~~~~

.. autoclass:: geminisdr.core.error_handling.MemoryError
   :members:
   :undoc-members:
   :show-inheritance:

   Exception for memory allocation and resource management errors.

ErrorHandler
~~~~~~~~~~~~

.. autoclass:: geminisdr.core.error_handling.ErrorHandler
   :members:
   :undoc-members:
   :show-inheritance:

   Centralized error handling system with automatic recovery strategies.

   Example usage::

       from geminisdr.core.error_handling import ErrorHandler
       
       error_handler = ErrorHandler(config)
       
       # Use error context for automatic handling
       with error_handler.error_context("model_loading"):
           model = load_model("my_model")

Functions and Decorators
-----------------------

retry_with_backoff
~~~~~~~~~~~~~~~~~~

.. autofunction:: geminisdr.core.error_handling.retry_with_backoff

   Decorator for automatic retry with exponential backoff on transient failures.

fallback_to_simulation
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geminisdr.core.error_handling.fallback_to_simulation

   Context manager that automatically falls back to simulation mode on hardware errors.