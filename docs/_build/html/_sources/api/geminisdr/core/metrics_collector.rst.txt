Metrics Collector
=================

.. automodule:: geminisdr.core.metrics_collector
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

MetricsCollector
~~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.metrics_collector.MetricsCollector
   :members:
   :undoc-members:
   :show-inheritance:

   System metrics collection and monitoring with anomaly detection.

   Features:
   
   * Real-time system performance metrics collection
   * ML training and inference metrics tracking
   * Anomaly detection with configurable thresholds
   * Time-series data storage and querying
   * Integration with monitoring dashboards

   Example usage::

       from geminisdr.core.metrics_collector import MetricsCollector
       
       collector = MetricsCollector(config)
       
       # Record custom metrics
       collector.record_metric("inference_latency", 0.045, tags={"model": "resnet"})
       
       # Record training metrics
       collector.record_training_metrics(epoch=10, loss=0.23, accuracy=0.94)
       
       # Get metrics summary
       summary = collector.get_metrics_summary("1h")

SystemMetrics
~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.metrics_collector.SystemMetrics
   :members:
   :undoc-members:
   :show-inheritance:

   Data class for system performance metrics.

TrainingMetrics
~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.metrics_collector.TrainingMetrics
   :members:
   :undoc-members:
   :show-inheritance:

   Data class for ML training metrics.

AnomalyDetector
~~~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.metrics_collector.AnomalyDetector
   :members:
   :undoc-members:
   :show-inheritance:

   Statistical anomaly detection for metrics monitoring.