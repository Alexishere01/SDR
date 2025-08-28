Model Metadata
==============

.. automodule:: geminisdr.core.model_metadata
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

ModelMetadata
~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.model_metadata.ModelMetadata
   :members:
   :undoc-members:
   :show-inheritance:

   Comprehensive metadata structure for model versioning and tracking.

   Contains information about:
   
   * Model identification (name, version, timestamp)
   * Training configuration (hyperparameters, data sources)
   * Performance metrics and validation results
   * System information (platform, device, code version)
   * Reproducibility data (random seeds, environment)

   Example usage::

       from geminisdr.core.model_metadata import ModelMetadata
       from datetime import datetime
       
       metadata = ModelMetadata(
           name="signal_classifier",
           version="2.1.0",
           timestamp=datetime.now(),
           hyperparameters={
               "learning_rate": 0.001,
               "batch_size": 64,
               "epochs": 100
           },
           performance_metrics={
               "accuracy": 0.94,
               "f1_score": 0.92,
               "inference_time_ms": 15.3
           },
           training_data_hash="abc123def456",
           code_version="v1.2.3",
           platform="darwin",
           device="mps"
       )

PerformanceMetrics
~~~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.model_metadata.PerformanceMetrics
   :members:
   :undoc-members:
   :show-inheritance:

   Data class for model performance metrics.

TrainingConfig
~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.model_metadata.TrainingConfig
   :members:
   :undoc-members:
   :show-inheritance:

   Data class for training configuration parameters.

SystemInfo
~~~~~~~~~~

.. autoclass:: geminisdr.core.model_metadata.SystemInfo
   :members:
   :undoc-members:
   :show-inheritance:

   Data class for system and environment information.

Functions
---------

generate_model_hash
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geminisdr.core.model_metadata.generate_model_hash

   Generate a unique hash for model fingerprinting.

validate_metadata
~~~~~~~~~~~~~~~~~

.. autofunction:: geminisdr.core.model_metadata.validate_metadata

   Validate model metadata completeness and consistency.