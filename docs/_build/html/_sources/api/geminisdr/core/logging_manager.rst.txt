Logging Manager
===============

.. automodule:: geminisdr.core.logging_manager
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

StructuredLogger
~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.logging_manager.StructuredLogger
   :members:
   :undoc-members:
   :show-inheritance:

   Advanced logging system with structured JSON output and context management.

   Features:
   
   * Structured JSON logging with consistent format
   * Persistent context that applies to all log messages
   * Performance metrics logging
   * Error logging with full context and stack traces
   * Log rotation and archival

   Example usage::

       from geminisdr.core.logging_manager import StructuredLogger
       
       logger = StructuredLogger("my_module", config.logging)
       logger.add_context(user_id="123", session="abc")
       
       # All subsequent logs will include the context
       logger.info("Processing started")
       logger.log_performance("training", duration=45.2, accuracy=0.95)

LogRotationHandler
~~~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.logging_manager.LogRotationHandler
   :members:
   :undoc-members:
   :show-inheritance:

   Custom log rotation handler with configurable retention policies.

Functions
---------

setup_logging
~~~~~~~~~~~~~

.. autofunction:: geminisdr.core.logging_manager.setup_logging

   Configure the global logging system with structured output and rotation.