Model Manager
=============

.. automodule:: geminisdr.core.model_manager
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

ModelManager
~~~~~~~~~~~~

.. autoclass:: geminisdr.core.model_manager.ModelManager
   :members:
   :undoc-members:
   :show-inheritance:

   Centralized model management with versioning and MLflow integration.

   Features:
   
   * Model saving with comprehensive metadata
   * Version compatibility validation
   * Model comparison and migration utilities
   * Integration with MLflow for experiment tracking
   * Automatic model fingerprinting and validation

   Example usage::

       from geminisdr.core.model_manager import ModelManager
       from geminisdr.core.model_metadata import ModelMetadata
       
       manager = ModelManager(config)
       
       # Save model with metadata
       metadata = ModelMetadata(
           name="my_model",
           version="1.0.0",
           hyperparameters={"lr": 0.001, "batch_size": 32},
           performance_metrics={"accuracy": 0.95}
       )
       model_path = manager.save_model(my_model, metadata)
       
       # Load model with validation
       loaded_model, metadata = manager.load_model("my_model", "1.0.0")
       
       # Compare model versions
       comparison = manager.compare_models("my_model:1.0.0", "my_model:1.1.0")

ModelRegistry
~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.model_manager.ModelRegistry
   :members:
   :undoc-members:
   :show-inheritance:

   Registry for tracking and managing model versions.

ModelMigrator
~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.model_manager.ModelMigrator
   :members:
   :undoc-members:
   :show-inheritance:

   Utilities for migrating models between versions and formats.