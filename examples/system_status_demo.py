#!/usr/bin/env python3
"""
System Status and Health Monitoring Demo

This script demonstrates the system status monitoring and error logging widgets.
"""

import sys
import logging
import time
from datetime import datetime
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PySide6.QtCore import QTimer

# Add the project root to the path
sys.path.insert(0, '.')

from gui.widgets.system_status import SystemStatusWidget, ErrorLogWidget, LogEntry, LogLevel


class SystemStatusDemo(QMainWindow):
    """Demo application for system status monitoring."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeminiSDR System Status Demo")
        self.setGeometry(100, 100, 1200, 800)
        
        # Setup logging
        self._setup_logging()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create system status widget
        self.system_status = SystemStatusWidget()
        layout.addWidget(self.system_status)
        
        # Create demo controls
        controls = self._create_demo_controls()
        layout.addWidget(controls)
        
        # Setup demo timer for generating test data
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self._generate_demo_logs)
        
        # Connect signals
        self.system_status.alert_triggered.connect(self._handle_alert)
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create loggers for different components
        self.loggers = {
            'sdr_interface': logging.getLogger('geminisdr.sdr_interface'),
            'ml_model': logging.getLogger('geminisdr.ml_model'),
            'signal_processor': logging.getLogger('geminisdr.signal_processor'),
            'gui_controller': logging.getLogger('geminisdr.gui_controller')
        }
        
    def _create_demo_controls(self) -> QWidget:
        """Create demo control buttons."""
        controls = QWidget()
        layout = QHBoxLayout(controls)
        
        # Start/stop demo logging
        self.demo_logging_btn = QPushButton("Start Demo Logging")
        self.demo_logging_btn.clicked.connect(self._toggle_demo_logging)
        layout.addWidget(self.demo_logging_btn)
        
        # Generate test alerts
        test_alert_btn = QPushButton("Generate Test Alert")
        test_alert_btn.clicked.connect(self._generate_test_alert)
        layout.addWidget(test_alert_btn)
        
        # Generate error logs
        error_log_btn = QPushButton("Generate Error Log")
        error_log_btn.clicked.connect(self._generate_error_log)
        layout.addWidget(error_log_btn)
        
        # Generate warning logs
        warning_log_btn = QPushButton("Generate Warning Log")
        warning_log_btn.clicked.connect(self._generate_warning_log)
        layout.addWidget(warning_log_btn)
        
        # Clear logs
        clear_logs_btn = QPushButton("Clear All Logs")
        clear_logs_btn.clicked.connect(self._clear_all_logs)
        layout.addWidget(clear_logs_btn)
        
        return controls
        
    def _toggle_demo_logging(self):
        """Toggle demo logging on/off."""
        if self.demo_timer.isActive():
            self.demo_timer.stop()
            self.demo_logging_btn.setText("Start Demo Logging")
        else:
            self.demo_timer.start(2000)  # Generate logs every 2 seconds
            self.demo_logging_btn.setText("Stop Demo Logging")
            
    def _generate_demo_logs(self):
        """Generate demo log entries."""
        import random
        
        # Randomly select a logger and level
        logger_name = random.choice(list(self.loggers.keys()))
        logger = self.loggers[logger_name]
        
        # Generate different types of log messages
        messages = [
            "Processing signal data batch",
            "Model inference completed successfully",
            "SDR parameter update applied",
            "GUI widget refresh completed",
            "Data buffer allocation successful",
            "Configuration parameter validated",
            "Network connection established",
            "File operation completed"
        ]
        
        warning_messages = [
            "High CPU usage detected during processing",
            "Memory usage approaching threshold",
            "Signal quality degraded",
            "Model accuracy below expected threshold",
            "Network latency increased",
            "Disk space running low"
        ]
        
        error_messages = [
            "Failed to connect to SDR device",
            "Model loading failed",
            "Signal processing error occurred",
            "Configuration file corrupted",
            "Network connection lost",
            "Memory allocation failed"
        ]
        
        # Randomly choose message type
        rand = random.random()
        if rand < 0.7:  # 70% info messages
            message = random.choice(messages)
            logger.info(message)
        elif rand < 0.9:  # 20% warning messages
            message = random.choice(warning_messages)
            logger.warning(message)
        else:  # 10% error messages
            message = random.choice(error_messages)
            logger.error(message)
            
    def _generate_test_alert(self):
        """Generate a test system alert."""
        import random
        
        alerts = [
            ("warning", "cpu_percent", "Test warning: CPU usage at 85%"),
            ("critical", "memory_percent", "Test critical: Memory usage at 96%"),
            ("warning", "disk_usage_percent", "Test warning: Disk usage at 92%"),
            ("critical", "gpu_temperature", "Test critical: GPU temperature at 95°C")
        ]
        
        severity, metric, message = random.choice(alerts)
        self.system_status._handle_alert(severity, metric, message)
        
    def _generate_error_log(self):
        """Generate a test error log."""
        import random
        
        error_messages = [
            "Database connection timeout",
            "Invalid configuration parameter",
            "File not found: /path/to/config.yaml",
            "Permission denied accessing device",
            "Network socket error",
            "Memory allocation failed for buffer"
        ]
        
        logger = random.choice(list(self.loggers.values()))
        message = random.choice(error_messages)
        logger.error(message)
        
    def _generate_warning_log(self):
        """Generate a test warning log."""
        import random
        
        warning_messages = [
            "Performance degradation detected",
            "Resource usage above normal levels",
            "Deprecated API usage detected",
            "Configuration parameter will be removed",
            "Network connection unstable",
            "Cache size approaching limit"
        ]
        
        logger = random.choice(list(self.loggers.values()))
        message = random.choice(warning_messages)
        logger.warning(message)
        
    def _clear_all_logs(self):
        """Clear all log entries."""
        # Find the error log widget in the system status widget
        for i in range(self.system_status.tab_widget.count()):
            tab_widget = self.system_status.tab_widget.widget(i)
            if hasattr(tab_widget, 'findChild'):
                error_log = tab_widget.findChild(ErrorLogWidget)
                if error_log:
                    error_log._clear_logs()
                    break
                    
    def _handle_alert(self, severity: str, metric: str, message: str):
        """Handle system alerts."""
        print(f"Alert received: [{severity.upper()}] {metric}: {message}")
        
    def closeEvent(self, event):
        """Handle application close event."""
        # Stop demo timer
        if self.demo_timer.isActive():
            self.demo_timer.stop()
            
        # Stop system monitoring
        if hasattr(self.system_status, 'monitor_thread'):
            self.system_status.monitor_thread.stop()
            
        event.accept()


def main():
    """Main function to run the demo."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("GeminiSDR System Status Demo")
    app.setApplicationVersion("1.0.0")
    
    # Create and show the demo window
    demo = SystemStatusDemo()
    demo.show()
    
    # Generate some initial log entries
    logger = logging.getLogger('demo')
    logger.info("System Status Demo started")
    logger.info("Monitoring system resources...")
    logger.warning("This is a demo warning message")
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()