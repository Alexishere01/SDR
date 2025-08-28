Getting Started
===============

This guide will help you get GeminiSDR up and running quickly.

Prerequisites
-------------

* Python 3.8 or higher
* Git for cloning the repository
* Platform-specific dependencies (see installation guides)

Quick Installation
------------------

1. Clone the repository:

   .. code-block:: bash
   
      git clone https://github.com/your-org/geminisdr.git
      cd geminisdr

2. Install dependencies:

   .. code-block:: bash
   
      pip install -r requirements.txt

3. Run a basic test:

   .. code-block:: python
   
      import geminisdr
      from geminisdr.core.signal_generator import SignalGenerator
      
      # Create a simple signal generator
      generator = SignalGenerator()
      signal = generator.generate_sine_wave(frequency=1000, duration=1.0)
      print(f"Generated signal with {len(signal)} samples")

First Steps
-----------

Now that you have GeminiSDR installed, try these examples:

* :doc:`../examples/basic_usage` - Basic signal processing
* :doc:`../examples/ml_training` - Train your first ML model

Next Steps
----------

* Read the :doc:`../guides/installation/index` for detailed platform setup
* Explore the :doc:`../api/index` for complete API reference
* Check out :doc:`../examples/index` for more examples