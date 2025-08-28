Architecture Overview
===================

GeminiSDR is designed as a modular, cross-platform software-defined radio system with advanced machine learning capabilities. The architecture emphasizes maintainability, scalability, and developer experience while supporting multiple hardware platforms.

System Architecture Diagram
---------------------------

.. mermaid::

   graph TB
       subgraph "Application Layer"
           UI[User Interface]
           API[REST API]
           CLI[Command Line Interface]
       end
       
       subgraph "Core Services Layer"
           CM[Configuration Manager]
           EH[Error Handler]
           LM[Logging Manager]
           MM[Memory Manager]
           MC[Metrics Collector]
           MOM[Model Manager]
       end
       
       subgraph "ML Processing Layer"
           IR[Intelligent Receiver]
           NA[Neural AMR]
           TA[Traditional AMR]
           AJ[Adversarial Jamming]
       end
       
       subgraph "Hardware Abstraction Layer"
           HA[Hardware Abstraction]
           SI[SDR Interface]
           SG[Signal Generator]
           VZ[Visualizer]
       end
       
       subgraph "Infrastructure Layer"
           FS[File System]
           DB[(MLflow Tracking)]
           LOG[(Log Storage)]
           CACHE[(Model Cache)]
       end
       
       UI --> CM
       API --> CM
       CLI --> CM
       
       CM --> EH
       CM --> LM
       CM --> MM
       CM --> MC
       CM --> MOM
       
       EH --> IR
       EH --> NA
       EH --> TA
       EH --> AJ
       
       IR --> HA
       NA --> HA
       TA --> HA
       AJ --> HA
       
       HA --> SI
       HA --> SG
       HA --> VZ
       
       MOM --> DB
       LM --> LOG
       MM --> CACHE
       MC --> LOG

Design Principles
----------------

Modularity
~~~~~~~~~~

The system is designed with clear separation of concerns:

* **Configuration Management**: Centralized configuration with environment-specific overrides
* **Error Handling**: Comprehensive error handling with automatic recovery strategies
* **Memory Management**: Intelligent memory optimization and resource management
* **Model Management**: Version-controlled model persistence with metadata tracking

Cross-Platform Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The architecture supports multiple platforms through:

* **Hardware Abstraction**: Unified interface for different SDR devices and compute platforms
* **Device Detection**: Automatic detection and optimization for M1 Mac, Linux VM, and CUDA environments
* **Configuration Profiles**: Platform-specific configuration templates
* **Testing Framework**: Cross-platform validation and performance benchmarking

Scalability
~~~~~~~~~~~

The system is designed to scale across different deployment scenarios:

* **Resource Optimization**: Dynamic batch size adjustment and memory management
* **Model Caching**: LRU cache for efficient model loading and memory usage
* **Monitoring**: Comprehensive metrics collection and anomaly detection
* **Recovery**: Automatic fallback and recovery strategies for system resilience

Developer Experience
~~~~~~~~~~~~~~~~~~~

The architecture prioritizes developer productivity:

* **Comprehensive Documentation**: Auto-generated API docs with examples
* **Testing Framework**: Cross-platform CI/CD with performance regression detection
* **Error Messages**: Clear, actionable error messages with context
* **Configuration**: Simple YAML-based configuration with validation

Component Interactions
----------------------

Configuration Flow
~~~~~~~~~~~~~~~~~

1. **Startup**: ConfigManager loads base configuration from YAML files
2. **Environment Detection**: System detects platform and applies environment-specific overrides
3. **Validation**: Configuration is validated against schema with clear error reporting
4. **Distribution**: Validated configuration is distributed to all system components

Error Handling Flow
~~~~~~~~~~~~~~~~~~

1. **Error Detection**: Components detect and classify errors by severity
2. **Context Collection**: Error context is collected including system state
3. **Recovery Strategy**: Appropriate recovery strategy is selected and executed
4. **Logging**: Error and recovery actions are logged with full context
5. **Notification**: Critical errors trigger alerts and notifications

Memory Management Flow
~~~~~~~~~~~~~~~~~~~~~

1. **Monitoring**: Continuous monitoring of RAM and GPU memory usage
2. **Optimization**: Dynamic batch size adjustment based on available resources
3. **Caching**: Intelligent model caching with LRU eviction
4. **Cleanup**: Automatic garbage collection and resource cleanup
5. **Alerting**: Memory pressure alerts and automatic mitigation

Model Lifecycle
~~~~~~~~~~~~~~~

1. **Training**: Models are trained with comprehensive metadata collection
2. **Validation**: Model performance is validated against minimum thresholds
3. **Saving**: Models are saved with metadata, fingerprints, and reproducibility info
4. **Registration**: Models are registered in MLflow with version tracking
5. **Loading**: Models are loaded with compatibility validation
6. **Caching**: Frequently used models are cached for performance