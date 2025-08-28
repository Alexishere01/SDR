Contributing to GeminiSDR
========================

We welcome contributions to GeminiSDR! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash
   
      git clone https://github.com/your-username/geminisdr.git
      cd geminisdr

3. Create a virtual environment:

   .. code-block:: bash
   
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install development dependencies:

   .. code-block:: bash
   
      pip install -r requirements.txt
      pip install -r docs/requirements.txt
      pip install -e .

Code Style
----------

We use the following tools to maintain code quality:

* **Black** for code formatting
* **isort** for import sorting
* **flake8** for linting
* **mypy** for type checking

Run these before submitting:

.. code-block:: bash

   black .
   isort .
   flake8 .
   mypy geminisdr/

Testing
-------

Run the test suite:

.. code-block:: bash

   pytest tests/

For cross-platform testing:

.. code-block:: bash

   pytest tests/ --platform=m1_native
   pytest tests/ --platform=vm_ubuntu
   pytest tests/ --platform=cuda

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.

Submitting Changes
------------------

1. Create a feature branch:

   .. code-block:: bash
   
      git checkout -b feature/your-feature-name

2. Make your changes and add tests
3. Run the test suite and ensure all tests pass
4. Update documentation if needed
5. Commit your changes with a clear message
6. Push to your fork and submit a pull request

Pull Request Guidelines
-----------------------

* Include a clear description of the changes
* Reference any related issues
* Ensure all tests pass
* Update documentation for new features
* Follow the existing code style

Getting Help
------------

* Open an issue for bugs or feature requests
* Join our discussions for questions
* Check existing issues before creating new ones