"""
Tests for documentation infrastructure.
"""

import os
import pytest
from pathlib import Path


class TestDocumentationStructure:
    """Test documentation directory structure and files."""
    
    def test_docs_directory_exists(self):
        """Test that docs directory exists."""
        docs_dir = Path("docs")
        assert docs_dir.exists(), "docs directory should exist"
        assert docs_dir.is_dir(), "docs should be a directory"
    
    def test_sphinx_config_exists(self):
        """Test that Sphinx configuration exists."""
        conf_file = Path("docs/conf.py")
        assert conf_file.exists(), "docs/conf.py should exist"
    
    def test_main_index_exists(self):
        """Test that main index file exists."""
        index_file = Path("docs/index.rst")
        assert index_file.exists(), "docs/index.rst should exist"
    
    def test_makefile_exists(self):
        """Test that Makefile exists."""
        makefile = Path("docs/Makefile")
        assert makefile.exists(), "docs/Makefile should exist"
    
    def test_requirements_exists(self):
        """Test that documentation requirements exist."""
        req_file = Path("docs/requirements.txt")
        assert req_file.exists(), "docs/requirements.txt should exist"
    
    def test_directory_structure(self):
        """Test that all required directories exist."""
        required_dirs = [
            "docs/guides",
            "docs/api", 
            "docs/architecture",
            "docs/development",
            "docs/examples",
            "docs/_templates"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            assert path.exists(), f"{dir_path} should exist"
            assert path.is_dir(), f"{dir_path} should be a directory"
    
    def test_templates_exist(self):
        """Test that documentation templates exist."""
        templates = [
            "docs/_templates/module.rst",
            "docs/_templates/class.rst", 
            "docs/_templates/function.rst",
            "docs/_templates/guide_template.rst",
            "docs/_templates/tutorial_template.rst"
        ]
        
        for template in templates:
            path = Path(template)
            assert path.exists(), f"{template} should exist"
    
    def test_github_workflow_exists(self):
        """Test that GitHub Actions workflow exists."""
        workflow = Path(".github/workflows/docs.yml")
        assert workflow.exists(), "Documentation workflow should exist"
    
    def test_build_script_exists(self):
        """Test that build script exists and is executable."""
        script = Path("scripts/build_docs.py")
        assert script.exists(), "Build script should exist"
        
        # Check if executable (on Unix systems)
        if os.name != 'nt':  # Not Windows
            assert os.access(script, os.X_OK), "Build script should be executable"


class TestDocumentationContent:
    """Test documentation content and structure."""
    
    def test_sphinx_config_content(self):
        """Test that Sphinx config has required settings."""
        conf_file = Path("docs/conf.py")
        content = conf_file.read_text()
        
        # Check for required extensions
        required_extensions = [
            'sphinx.ext.autodoc',
            'sphinx.ext.autosummary', 
            'sphinx.ext.napoleon',
            'myst_parser'
        ]
        
        for ext in required_extensions:
            assert ext in content, f"Extension {ext} should be in conf.py"
    
    def test_index_has_toctree(self):
        """Test that main index has proper toctree."""
        index_file = Path("docs/index.rst")
        content = index_file.read_text()
        
        assert ".. toctree::" in content, "Index should have toctree"
        assert "guides/index" in content, "Index should include guides"
        assert "api/index" in content, "Index should include API docs"
    
    def test_requirements_has_sphinx(self):
        """Test that requirements includes Sphinx."""
        req_file = Path("docs/requirements.txt")
        content = req_file.read_text()
        
        assert "sphinx" in content.lower(), "Requirements should include Sphinx"
        assert "sphinx-rtd-theme" in content, "Requirements should include RTD theme"


@pytest.mark.skipif(not Path("docs").exists(), reason="Documentation not set up")
class TestDocumentationBuild:
    """Test documentation building (requires docs setup)."""
    
    def test_can_import_sphinx(self):
        """Test that Sphinx can be imported."""
        try:
            import sphinx
            assert sphinx.__version__, "Sphinx should have version"
        except ImportError:
            pytest.skip("Sphinx not installed")
    
    def test_build_script_syntax(self):
        """Test that build script has valid Python syntax."""
        script = Path("scripts/build_docs.py")
        if script.exists():
            # Try to compile the script
            with open(script) as f:
                code = f.read()
            compile(code, str(script), 'exec')