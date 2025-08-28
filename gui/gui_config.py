"""
GUI Configuration Module

This module handles GUI-specific configuration settings and preferences.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import os
from pathlib import Path


class VisualizationMode(Enum):
    """Visualization display modes."""
    SPECTRUM = "spectrum"
    WATERFALL = "waterfall"
    CONSTELLATION = "constellation"
    COMBINED = "combined"


@dataclass
class GUIConfig:
    """GUI configuration settings."""
    theme: str = "dark"
    window_geometry: Optional[Dict[str, int]] = None
    layout_state: Optional[bytes] = None
    update_rate_fps: int = 30
    max_history_points: int = 1000
    auto_save_settings: bool = True
    
    def __post_init__(self):
        if self.window_geometry is None:
            self.window_geometry = {
                "x": 100,
                "y": 100,
                "width": 1200,
                "height": 800
            }


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    mode: VisualizationMode = VisualizationMode.SPECTRUM
    fft_size: int = 2048
    window_function: str = "hann"
    averaging_factor: float = 0.8
    color_map: str = "viridis"
    show_grid: bool = True
    show_measurements: bool = True
    waterfall_history_size: int = 1000
    constellation_points: int = 10000


@dataclass
class ControlConfig:
    """SDR control settings."""
    auto_apply_changes: bool = True
    confirmation_required: bool = False
    preset_auto_save: bool = True
    fine_tune_step_hz: float = 1000.0
    gain_step_db: float = 1.0
    frequency_limits: Dict[str, float] = field(default_factory=lambda: {
        "min_mhz": 70.0,
        "max_mhz": 6000.0
    })


class GUIConfigManager:
    """Manages GUI configuration loading and saving."""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".geminisdr")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.config_file = self.config_dir / "gui_config.json"
        
        # Default configurations
        self.gui_config = GUIConfig()
        self.visualization_config = VisualizationConfig()
        self.control_config = ControlConfig()
        
        # Load existing configuration
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load GUI config
            if "gui" in config_data:
                gui_data = config_data["gui"]
                self.gui_config = GUIConfig(**gui_data)
            
            # Load visualization config
            if "visualization" in config_data:
                viz_data = config_data["visualization"]
                if "mode" in viz_data:
                    viz_data["mode"] = VisualizationMode(viz_data["mode"])
                self.visualization_config = VisualizationConfig(**viz_data)
            
            # Load control config
            if "control" in config_data:
                control_data = config_data["control"]
                self.control_config = ControlConfig(**control_data)
                
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Warning: Failed to load GUI config: {e}")
            # Use default configurations
    
    def save_config(self) -> None:
        """Save configuration to file."""
        config_data = {
            "gui": {
                "theme": self.gui_config.theme,
                "window_geometry": self.gui_config.window_geometry,
                "update_rate_fps": self.gui_config.update_rate_fps,
                "max_history_points": self.gui_config.max_history_points,
                "auto_save_settings": self.gui_config.auto_save_settings,
            },
            "visualization": {
                "mode": self.visualization_config.mode.value,
                "fft_size": self.visualization_config.fft_size,
                "window_function": self.visualization_config.window_function,
                "averaging_factor": self.visualization_config.averaging_factor,
                "color_map": self.visualization_config.color_map,
                "show_grid": self.visualization_config.show_grid,
                "show_measurements": self.visualization_config.show_measurements,
                "waterfall_history_size": self.visualization_config.waterfall_history_size,
                "constellation_points": self.visualization_config.constellation_points,
            },
            "control": {
                "auto_apply_changes": self.control_config.auto_apply_changes,
                "confirmation_required": self.control_config.confirmation_required,
                "preset_auto_save": self.control_config.preset_auto_save,
                "fine_tune_step_hz": self.control_config.fine_tune_step_hz,
                "gain_step_db": self.control_config.gain_step_db,
                "frequency_limits": self.control_config.frequency_limits,
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except (IOError, OSError) as e:
            print(f"Warning: Failed to save GUI config: {e}")
    
    def get_gui_config(self) -> GUIConfig:
        """Get GUI configuration."""
        return self.gui_config
    
    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration."""
        return self.visualization_config
    
    def get_control_config(self) -> ControlConfig:
        """Get control configuration."""
        return self.control_config
    
    def update_gui_config(self, **kwargs) -> None:
        """Update GUI configuration."""
        for key, value in kwargs.items():
            if hasattr(self.gui_config, key):
                setattr(self.gui_config, key, value)
        
        if self.gui_config.auto_save_settings:
            self.save_config()
    
    def update_visualization_config(self, **kwargs) -> None:
        """Update visualization configuration."""
        for key, value in kwargs.items():
            if hasattr(self.visualization_config, key):
                setattr(self.visualization_config, key, value)
        
        if self.gui_config.auto_save_settings:
            self.save_config()
    
    def update_control_config(self, **kwargs) -> None:
        """Update control configuration."""
        for key, value in kwargs.items():
            if hasattr(self.control_config, key):
                setattr(self.control_config, key, value)
        
        if self.gui_config.auto_save_settings:
            self.save_config()