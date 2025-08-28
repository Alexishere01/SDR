Configuration Manager
====================

.. automodule:: geminisdr.config.config_manager
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

ConfigManager
~~~~~~~~~~~~~

.. autoclass:: geminisdr.config.config_manager.ConfigManager
   :members:
   :undoc-members:
   :show-inheritance:

   The ConfigManager class provides centralized configuration management with support for:
   
   * Environment-specific configuration loading
   * Configuration validation and error reporting
   * Hot-reloading of configuration changes
   * Hierarchical configuration with overrides

   Example usage::

       from geminisdr.config.config_manager import ConfigManager
       
       config_manager = ConfigManager()
       config = config_manager.load_config()
       
       # Access configuration values
       device = config.hardware.device_preference
       batch_size = config.ml.batch_size