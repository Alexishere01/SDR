Recovery Strategies
==================

.. automodule:: geminisdr.core.recovery_strategies
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

RecoveryStrategy
~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.recovery_strategies.RecoveryStrategy
   :members:
   :undoc-members:
   :show-inheritance:

   Abstract base class for error recovery strategies.

HardwareRecoveryStrategy
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.recovery_strategies.HardwareRecoveryStrategy
   :members:
   :undoc-members:
   :show-inheritance:

   Recovery strategy for hardware-related failures.

   Implements:
   
   * Automatic fallback to simulation mode
   * Device switching (GPU to CPU, different SDR devices)
   * Hardware reconnection attempts
   * Resource reallocation

MemoryRecoveryStrategy
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.recovery_strategies.MemoryRecoveryStrategy
   :members:
   :undoc-members:
   :show-inheritance:

   Recovery strategy for memory-related failures.

   Implements:
   
   * Automatic batch size reduction
   * Memory cleanup and garbage collection
   * Model cache eviction
   * Resource optimization

ModelRecoveryStrategy
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.recovery_strategies.ModelRecoveryStrategy
   :members:
   :undoc-members:
   :show-inheritance:

   Recovery strategy for model loading and compatibility failures.

   Implements:
   
   * Alternative model version loading
   * Model format migration
   * Fallback to default models
   * Compatibility validation

ConfigRecoveryStrategy
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.recovery_strategies.ConfigRecoveryStrategy
   :members:
   :undoc-members:
   :show-inheritance:

   Recovery strategy for configuration-related failures.

   Implements:
   
   * Fallback to default configuration
   * Environment-specific overrides
   * Configuration validation and repair
   * Dynamic reconfiguration

Functions
---------

register_recovery_strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geminisdr.core.recovery_strategies.register_recovery_strategies

   Register default recovery strategies for common error types.

create_recovery_chain
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geminisdr.core.recovery_strategies.create_recovery_chain

   Create a chain of recovery strategies for complex error scenarios.