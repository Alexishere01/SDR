"""
Waterfall Display Widget

This module provides high-performance waterfall visualization using OpenGL acceleration
for time-frequency analysis with configurable color mapping and navigation.
"""

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSlider, QComboBox, QCheckBox, QSpinBox,
                               QDoubleSpinBox, QPushButton, QGroupBox,
                               QFrame, QSizePolicy)
from PySide6.QtCore import QTimer, Signal, Slot, Qt, QThread, QObject, QSize
from PySide6.QtGui import QFont, QPen, QColor, QOpenGLContext
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLTexture
from PySide6.QtOpenGLWidgets import QOpenGLWidget
import OpenGL.GL as gl
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import colorsys
import time


class ColorMap(Enum):
    """Available color maps for waterfall display."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    JET = "jet"
    HOT = "hot"
    COOL = "cool"
    GRAY = "gray"
    RAINBOW = "rainbow"


@dataclass
class WaterfallConfig:
    """Configuration for waterfall display."""
    history_size: int = 1000
    color_map: ColorMap = ColorMap.VIRIDIS
    intensity_min: float = -80.0  # dBm
    intensity_max: float = -20.0  # dBm
    time_span_seconds: float = 10.0
    auto_scale: bool = True
    show_colorbar: bool = True
    show_grid: bool = False
    interpolation: bool = True
    update_rate_fps: int = 30


class ColorMapGenerator:
    """Generate color maps for waterfall display."""
    
    @staticmethod
    def generate_colormap(colormap: ColorMap, size: int = 256) -> np.ndarray:
        """Generate RGB color map array."""
        colors = np.zeros((size, 3), dtype=np.float32)
        
        if colormap == ColorMap.VIRIDIS:
            # Viridis colormap approximation
            for i in range(size):
                t = i / (size - 1)
                colors[i] = ColorMapGenerator._viridis_color(t)
                
        elif colormap == ColorMap.PLASMA:
            # Plasma colormap approximation
            for i in range(size):
                t = i / (size - 1)
                colors[i] = ColorMapGenerator._plasma_color(t)
                
        elif colormap == ColorMap.INFERNO:
            # Inferno colormap approximation
            for i in range(size):
                t = i / (size - 1)
                colors[i] = ColorMapGenerator._inferno_color(t)
                
        elif colormap == ColorMap.JET:
            # Jet colormap
            for i in range(size):
                t = i / (size - 1)
                colors[i] = ColorMapGenerator._jet_color(t)
                
        elif colormap == ColorMap.HOT:
            # Hot colormap
            for i in range(size):
                t = i / (size - 1)
                colors[i] = ColorMapGenerator._hot_color(t)
                
        elif colormap == ColorMap.COOL:
            # Cool colormap
            for i in range(size):
                t = i / (size - 1)
                colors[i] = ColorMapGenerator._cool_color(t)
                
        elif colormap == ColorMap.GRAY:
            # Grayscale
            for i in range(size):
                t = i / (size - 1)
                colors[i] = [t, t, t]
                
        elif colormap == ColorMap.RAINBOW:
            # Rainbow colormap
            for i in range(size):
                t = i / (size - 1)
                colors[i] = ColorMapGenerator._rainbow_color(t)
                
        else:
            # Default to viridis
            for i in range(size):
                t = i / (size - 1)
                colors[i] = ColorMapGenerator._viridis_color(t)
                
        return colors
    
    @staticmethod
    def _viridis_color(t: float) -> List[float]:
        """Viridis color approximation."""
        # Simplified viridis approximation
        r = 0.267 + 0.005 * t + 0.322 * t**2 - 0.176 * t**3
        g = 0.004 + 0.396 * t + 0.776 * t**2 - 0.710 * t**3
        b = 0.329 + 1.074 * t - 0.733 * t**2 + 0.003 * t**3
        return [max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))]
    
    @staticmethod
    def _plasma_color(t: float) -> List[float]:
        """Plasma color approximation."""
        r = 0.050 + 0.530 * t + 1.543 * t**2 - 1.898 * t**3 + 0.775 * t**4
        g = 0.013 + 0.067 * t + 2.707 * t**2 - 2.298 * t**3 + 0.511 * t**4
        b = 0.267 + 1.249 * t - 2.724 * t**2 + 2.604 * t**3 - 0.896 * t**4
        return [max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))]
    
    @staticmethod
    def _inferno_color(t: float) -> List[float]:
        """Inferno color approximation."""
        r = 0.001 + 0.666 * t + 1.797 * t**2 - 1.882 * t**3 + 0.418 * t**4
        g = 0.014 + 0.125 * t + 2.635 * t**2 - 2.99 * t**3 + 1.216 * t**4
        b = 0.015 + 0.615 * t - 0.203 * t**2 - 0.580 * t**3 + 0.153 * t**4
        return [max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))]
    
    @staticmethod
    def _jet_color(t: float) -> List[float]:
        """Jet colormap."""
        if t < 0.25:
            r, g, b = 0, 4 * t, 1
        elif t < 0.5:
            r, g, b = 0, 1, 1 - 4 * (t - 0.25)
        elif t < 0.75:
            r, g, b = 4 * (t - 0.5), 1, 0
        else:
            r, g, b = 1, 1 - 4 * (t - 0.75), 0
        return [max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))]
    
    @staticmethod
    def _hot_color(t: float) -> List[float]:
        """Hot colormap."""
        if t < 1/3:
            r, g, b = 3 * t, 0, 0
        elif t < 2/3:
            r, g, b = 1, 3 * (t - 1/3), 0
        else:
            r, g, b = 1, 1, 3 * (t - 2/3)
        return [max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))]
    
    @staticmethod
    def _cool_color(t: float) -> List[float]:
        """Cool colormap."""
        return [t, 1 - t, 1]
    
    @staticmethod
    def _rainbow_color(t: float) -> List[float]:
        """Rainbow colormap using HSV."""
        h = t * 0.8  # Hue from 0 to 0.8 (red to violet)
        s = 1.0      # Full saturation
        v = 1.0      # Full value
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return [r, g, b]


class WaterfallRenderer(QOpenGLWidget):
    """High-performance waterfall display using OpenGL."""
    
    # Signals
    frequency_selected = Signal(float)
    time_selected = Signal(float)
    measurement_made = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.config = WaterfallConfig()
        
        # Data storage
        self.spectrum_history = []  # List of spectrum arrays
        self.time_stamps = []       # Corresponding timestamps
        self.frequencies = np.array([])
        
        # OpenGL resources
        self.texture = None
        self.shader_program = None
        self.vertex_buffer = None
        self.color_map_texture = None
        
        # Display parameters
        self.sample_rate = 2e6
        self.center_freq = 100e6
        self.zoom_factor = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        
        # Mouse interaction
        self.last_mouse_pos = None
        self.mouse_pressed = False
        
        # Performance tracking
        self.last_update_time = time.time()
        self.frame_count = 0
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def initializeGL(self) -> None:
        """Initialize OpenGL resources."""
        # Enable blending for transparency
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Create shader program
        self._create_shaders()
        
        # Create vertex buffer for quad
        self._create_vertex_buffer()
        
        # Create color map texture
        self._create_colormap_texture()
        
        # Set clear color
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        
    def _create_shaders(self) -> None:
        """Create OpenGL shaders for waterfall rendering."""
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec2 position;
        layout (location = 1) in vec2 texCoord;
        
        uniform mat4 mvpMatrix;
        uniform vec2 panOffset;
        uniform float zoomFactor;
        
        out vec2 TexCoord;
        
        void main()
        {
            vec2 pos = (position + panOffset) * zoomFactor;
            gl_Position = mvpMatrix * vec4(pos, 0.0, 1.0);
            TexCoord = texCoord;
        }
        """
        
        fragment_shader_source = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        uniform sampler2D waterfallTexture;
        uniform sampler1D colorMapTexture;
        uniform float intensityMin;
        uniform float intensityMax;
        uniform bool autoScale;
        
        void main()
        {
            float intensity = texture(waterfallTexture, TexCoord).r;
            
            // Normalize intensity to [0, 1] range
            float normalized = (intensity - intensityMin) / (intensityMax - intensityMin);
            normalized = clamp(normalized, 0.0, 1.0);
            
            // Sample color from color map
            vec3 color = texture(colorMapTexture, normalized).rgb;
            
            FragColor = vec4(color, 1.0);
        }
        """
        
        self.shader_program = QOpenGLShaderProgram()
        self.shader_program.addShaderFromSourceCode(
            QOpenGLShaderProgram.Vertex, vertex_shader_source
        )
        self.shader_program.addShaderFromSourceCode(
            QOpenGLShaderProgram.Fragment, fragment_shader_source
        )
        
        if not self.shader_program.link():
            print(f"Shader linking failed: {self.shader_program.log()}")
            
    def _create_vertex_buffer(self) -> None:
        """Create vertex buffer for rendering quad."""
        # Quad vertices (position + texture coordinates)
        vertices = np.array([
            # Position    # TexCoord
            -1.0, -1.0,   0.0, 0.0,  # Bottom-left
             1.0, -1.0,   1.0, 0.0,  # Bottom-right
             1.0,  1.0,   1.0, 1.0,  # Top-right
            -1.0,  1.0,   0.0, 1.0   # Top-left
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2,  # First triangle
            2, 3, 0   # Second triangle
        ], dtype=np.uint32)
        
        # Create and bind vertex array object
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        
        # Create vertex buffer
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        
        # Create element buffer
        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        
        # Set vertex attributes
        # Position attribute
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, None)
        gl.glEnableVertexAttribArray(0)
        
        # Texture coordinate attribute
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, gl.ctypes.c_void_p(2 * 4))
        gl.glEnableVertexAttribArray(1)
        
        gl.glBindVertexArray(0)
        
    def _create_colormap_texture(self) -> None:
        """Create 1D texture for color mapping."""
        # Generate color map
        colormap_data = ColorMapGenerator.generate_colormap(self.config.color_map, 256)
        
        # Create 1D texture
        self.color_map_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_1D, self.color_map_texture)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        # Upload color map data
        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, gl.GL_RGB, 256, 0, 
                       gl.GL_RGB, gl.GL_FLOAT, colormap_data)
        
        gl.glBindTexture(gl.GL_TEXTURE_1D, 0)
        
    def _update_waterfall_texture(self) -> None:
        """Update the main waterfall texture with spectrum history."""
        if not self.spectrum_history:
            return
            
        # Convert spectrum history to 2D array
        height = len(self.spectrum_history)
        width = len(self.spectrum_history[0]) if height > 0 else 0
        
        if width == 0:
            return
            
        # Create 2D array from spectrum history
        waterfall_data = np.array(self.spectrum_history, dtype=np.float32)
        
        # Normalize data if auto-scaling is enabled
        if self.config.auto_scale:
            data_min = np.min(waterfall_data)
            data_max = np.max(waterfall_data)
            if data_max > data_min:
                waterfall_data = (waterfall_data - data_min) / (data_max - data_min)
                waterfall_data = waterfall_data * (self.config.intensity_max - self.config.intensity_min) + self.config.intensity_min
        
        # Create or update texture
        if self.texture is None:
            self.texture = gl.glGenTextures(1)
            
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        if self.config.interpolation:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        else:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        
        # Upload texture data
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, width, height, 0,
                       gl.GL_RED, gl.GL_FLOAT, waterfall_data)
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        
    def paintGL(self) -> None:
        """Render the waterfall display."""
        # Clear the screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        if not self.spectrum_history or self.shader_program is None:
            return
            
        # Update waterfall texture
        self._update_waterfall_texture()
        
        # Use shader program
        self.shader_program.bind()
        
        # Set uniforms
        mvp_matrix = self._get_mvp_matrix()
        self.shader_program.setUniformValue("mvpMatrix", mvp_matrix)
        self.shader_program.setUniformValue("panOffset", self.pan_offset_x, self.pan_offset_y)
        self.shader_program.setUniformValue("zoomFactor", self.zoom_factor)
        self.shader_program.setUniformValue("intensityMin", self.config.intensity_min)
        self.shader_program.setUniformValue("intensityMax", self.config.intensity_max)
        self.shader_program.setUniformValue("autoScale", self.config.auto_scale)
        
        # Bind textures
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        self.shader_program.setUniformValue("waterfallTexture", 0)
        
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_1D, self.color_map_texture)
        self.shader_program.setUniformValue("colorMapTexture", 1)
        
        # Render quad
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        
        # Unbind shader
        self.shader_program.release()
        
        # Update performance metrics
        self._update_performance_metrics()
        
    def _get_mvp_matrix(self):
        """Get model-view-projection matrix."""
        # For now, return identity matrix
        # In a full implementation, this would handle proper 3D transformations
        from PySide6.QtGui import QMatrix4x4
        matrix = QMatrix4x4()
        matrix.setToIdentity()
        return matrix
        
    def _update_performance_metrics(self) -> None:
        """Update performance tracking."""
        current_time = time.time()
        self.frame_count += 1
        
        # Calculate FPS every second
        if current_time - self.last_update_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_update_time)
            self.last_update_time = current_time
            self.frame_count = 0
            
            # Emit performance signal if needed
            # self.performance_updated.emit({'fps': fps})
            
    def resizeGL(self, width: int, height: int) -> None:
        """Handle OpenGL resize events."""
        gl.glViewport(0, 0, width, height)
        
    def add_spectrum_data(self, spectrum: np.ndarray, timestamp: float = None) -> None:
        """Add new spectrum data to waterfall history."""
        if timestamp is None:
            timestamp = time.time()
            
        # Add to history
        self.spectrum_history.append(spectrum.copy())
        self.time_stamps.append(timestamp)
        
        # Limit history size
        while len(self.spectrum_history) > self.config.history_size:
            self.spectrum_history.pop(0)
            self.time_stamps.pop(0)
            
        # Trigger repaint
        self.update()
        
    def set_frequencies(self, frequencies: np.ndarray) -> None:
        """Set frequency array for the waterfall display."""
        self.frequencies = frequencies.copy()
        
    def set_sdr_parameters(self, sample_rate: float, center_freq: float) -> None:
        """Update SDR parameters."""
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        
    def set_config(self, config: WaterfallConfig) -> None:
        """Update waterfall configuration."""
        old_colormap = self.config.color_map
        self.config = config
        
        # Update color map texture if changed
        if config.color_map != old_colormap:
            self._create_colormap_texture()
            
        self.update()
        
    def get_config(self) -> WaterfallConfig:
        """Get current waterfall configuration."""
        return self.config
        
    def clear_history(self) -> None:
        """Clear waterfall history."""
        self.spectrum_history.clear()
        self.time_stamps.clear()
        self.update()
        
    def zoom_in(self, factor: float = 1.2) -> None:
        """Zoom in by the specified factor."""
        self.zoom_factor *= factor
        self.update()
        
    def zoom_out(self, factor: float = 1.2) -> None:
        """Zoom out by the specified factor."""
        self.zoom_factor /= factor
        self.update()
        
    def reset_zoom(self) -> None:
        """Reset zoom to default."""
        self.zoom_factor = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.update()
        
    def mousePressEvent(self, event) -> None:
        """Handle mouse press events."""
        self.last_mouse_pos = event.pos()
        self.mouse_pressed = True
        
    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move events for panning."""
        if self.mouse_pressed and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            
            # Convert to normalized coordinates
            self.pan_offset_x += delta.x() / self.width() * 2.0 / self.zoom_factor
            self.pan_offset_y -= delta.y() / self.height() * 2.0 / self.zoom_factor
            
            self.last_mouse_pos = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release events."""
        self.mouse_pressed = False
        
        # Emit frequency/time selection signal for single clicks
        if event.button() == Qt.LeftButton:
            # Convert mouse position to frequency/time
            freq, time_val = self._mouse_to_freq_time(event.pos())
            if freq is not None:
                self.frequency_selected.emit(freq)
                
    def wheelEvent(self, event) -> None:
        """Handle mouse wheel events for zooming."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in(1.1)
        else:
            self.zoom_out(1.1)
            
    def _mouse_to_freq_time(self, pos) -> Tuple[Optional[float], Optional[float]]:
        """Convert mouse position to frequency and time values."""
        if len(self.frequencies) == 0 or len(self.time_stamps) == 0:
            return None, None
            
        # Normalize mouse position
        x_norm = pos.x() / self.width()
        y_norm = 1.0 - pos.y() / self.height()  # Flip Y coordinate
        
        # Account for zoom and pan
        x_norm = (x_norm - 0.5) / self.zoom_factor - self.pan_offset_x / 2.0 + 0.5
        y_norm = (y_norm - 0.5) / self.zoom_factor - self.pan_offset_y / 2.0 + 0.5
        
        # Convert to frequency and time
        if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
            freq_idx = int(x_norm * len(self.frequencies))
            time_idx = int(y_norm * len(self.time_stamps))
            
            freq_idx = max(0, min(freq_idx, len(self.frequencies) - 1))
            time_idx = max(0, min(time_idx, len(self.time_stamps) - 1))
            
            frequency = self.frequencies[freq_idx]
            timestamp = self.time_stamps[time_idx]
            
            return frequency, timestamp
            
        return None, None


class WaterfallWidget(QWidget):
    """Complete waterfall widget with controls and OpenGL display."""
    
    # Signals
    frequency_selected = Signal(float)
    time_selected = Signal(float)
    config_changed = Signal(WaterfallConfig)
    measurement_made = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.config = WaterfallConfig()
        
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self) -> None:
        """Setup the waterfall widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Control panel
        controls = self._create_controls()
        layout.addWidget(controls)
        
        # Waterfall display
        self.waterfall_display = WaterfallRenderer()
        self.waterfall_display.setMinimumHeight(400)
        layout.addWidget(self.waterfall_display)
        
        # Status bar
        self.status_label = QLabel("Waterfall Display - Ready")
        self.status_label.setStyleSheet("QLabel { color: #888888; font-size: 10px; }")
        layout.addWidget(self.status_label)
        
    def _create_controls(self) -> QWidget:
        """Create waterfall control panel."""
        controls = QFrame()
        controls.setFrameStyle(QFrame.StyledPanel)
        controls.setMaximumHeight(120)
        
        layout = QHBoxLayout(controls)
        
        # Display Settings Group
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout(display_group)
        
        # Color map selection
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Color Map:"))
        self.colormap_combo = QComboBox()
        for colormap in ColorMap:
            self.colormap_combo.addItem(colormap.value.title(), colormap)
        self.colormap_combo.setCurrentText(self.config.color_map.value.title())
        self.colormap_combo.currentTextChanged.connect(self._update_colormap)
        colormap_layout.addWidget(self.colormap_combo)
        display_layout.addLayout(colormap_layout)
        
        # History size
        history_layout = QHBoxLayout()
        history_layout.addWidget(QLabel("History:"))
        self.history_spinbox = QSpinBox()
        self.history_spinbox.setRange(100, 5000)
        self.history_spinbox.setValue(self.config.history_size)
        self.history_spinbox.setSuffix(" lines")
        self.history_spinbox.valueChanged.connect(self._update_history_size)
        history_layout.addWidget(self.history_spinbox)
        display_layout.addLayout(history_layout)
        
        layout.addWidget(display_group)
        
        # Intensity Settings Group
        intensity_group = QGroupBox("Intensity Range")
        intensity_layout = QVBoxLayout(intensity_group)
        
        # Min intensity
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min:"))
        self.intensity_min_spinbox = QDoubleSpinBox()
        self.intensity_min_spinbox.setRange(-120, 0)
        self.intensity_min_spinbox.setValue(self.config.intensity_min)
        self.intensity_min_spinbox.setSuffix(" dBm")
        self.intensity_min_spinbox.valueChanged.connect(self._update_intensity_min)
        min_layout.addWidget(self.intensity_min_spinbox)
        intensity_layout.addLayout(min_layout)
        
        # Max intensity
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max:"))
        self.intensity_max_spinbox = QDoubleSpinBox()
        self.intensity_max_spinbox.setRange(-60, 40)
        self.intensity_max_spinbox.setValue(self.config.intensity_max)
        self.intensity_max_spinbox.setSuffix(" dBm")
        self.intensity_max_spinbox.valueChanged.connect(self._update_intensity_max)
        max_layout.addWidget(self.intensity_max_spinbox)
        intensity_layout.addLayout(max_layout)
        
        # Auto scale checkbox
        self.auto_scale_cb = QCheckBox("Auto Scale")
        self.auto_scale_cb.setChecked(self.config.auto_scale)
        self.auto_scale_cb.toggled.connect(self._toggle_auto_scale)
        intensity_layout.addWidget(self.auto_scale_cb)
        
        layout.addWidget(intensity_group)
        
        # Control Buttons Group
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.waterfall_display.zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.waterfall_display.zoom_out)
        zoom_layout.addWidget(self.zoom_out_btn)
        
        self.reset_zoom_btn = QPushButton("Reset")
        self.reset_zoom_btn.clicked.connect(self.waterfall_display.reset_zoom)
        zoom_layout.addWidget(self.reset_zoom_btn)
        
        control_layout.addLayout(zoom_layout)
        
        # Clear button
        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.clicked.connect(self.waterfall_display.clear_history)
        control_layout.addWidget(self.clear_btn)
        
        # Options
        self.interpolation_cb = QCheckBox("Interpolation")
        self.interpolation_cb.setChecked(self.config.interpolation)
        self.interpolation_cb.toggled.connect(self._toggle_interpolation)
        control_layout.addWidget(self.interpolation_cb)
        
        layout.addWidget(control_group)
        
        return controls
        
    def _setup_connections(self) -> None:
        """Setup signal-slot connections."""
        # Connect waterfall display signals
        self.waterfall_display.frequency_selected.connect(self.frequency_selected)
        self.waterfall_display.time_selected.connect(self.time_selected)
        
        # Update display configuration
        self.waterfall_display.set_config(self.config)
        
    @Slot(str)
    def _update_colormap(self, colormap_text: str) -> None:
        """Update color map."""
        for colormap in ColorMap:
            if colormap.value.title() == colormap_text:
                self.config.color_map = colormap
                self.waterfall_display.set_config(self.config)
                self.config_changed.emit(self.config)
                break
                
    @Slot(int)
    def _update_history_size(self, size: int) -> None:
        """Update history size."""
        self.config.history_size = size
        self.waterfall_display.set_config(self.config)
        self.config_changed.emit(self.config)
        
    @Slot(float)
    def _update_intensity_min(self, value: float) -> None:
        """Update minimum intensity."""
        self.config.intensity_min = value
        self.waterfall_display.set_config(self.config)
        self.config_changed.emit(self.config)
        
    @Slot(float)
    def _update_intensity_max(self, value: float) -> None:
        """Update maximum intensity."""
        self.config.intensity_max = value
        self.waterfall_display.set_config(self.config)
        self.config_changed.emit(self.config)
        
    @Slot(bool)
    def _toggle_auto_scale(self, enabled: bool) -> None:
        """Toggle auto scaling."""
        self.config.auto_scale = enabled
        self.waterfall_display.set_config(self.config)
        self.config_changed.emit(self.config)
        
    @Slot(bool)
    def _toggle_interpolation(self, enabled: bool) -> None:
        """Toggle interpolation."""
        self.config.interpolation = enabled
        self.waterfall_display.set_config(self.config)
        self.config_changed.emit(self.config)
        
    def add_spectrum_data(self, spectrum: np.ndarray, timestamp: float = None) -> None:
        """Add new spectrum data to waterfall."""
        self.waterfall_display.add_spectrum_data(spectrum, timestamp)
        
        # Update status
        if len(spectrum) > 0:
            max_power = np.max(spectrum)
            min_power = np.min(spectrum)
            history_count = len(self.waterfall_display.spectrum_history)
            self.status_label.setText(
                f"History: {history_count} lines | Range: {min_power:.1f} to {max_power:.1f} dBm"
            )
            
    def set_frequencies(self, frequencies: np.ndarray) -> None:
        """Set frequency array."""
        self.waterfall_display.set_frequencies(frequencies)
        
    def set_sdr_parameters(self, sample_rate: float, center_freq: float) -> None:
        """Update SDR parameters."""
        self.waterfall_display.set_sdr_parameters(sample_rate, center_freq)
        
    def get_config(self) -> WaterfallConfig:
        """Get current configuration."""
        return self.config
        
    def set_config(self, config: WaterfallConfig) -> None:
        """Set configuration."""
        self.config = config
        self.waterfall_display.set_config(config)
        
        # Update UI controls
        self.colormap_combo.setCurrentText(config.color_map.value.title())
        self.history_spinbox.setValue(config.history_size)
        self.intensity_min_spinbox.setValue(config.intensity_min)
        self.intensity_max_spinbox.setValue(config.intensity_max)
        self.auto_scale_cb.setChecked(config.auto_scale)
        self.interpolation_cb.setChecked(config.interpolation)