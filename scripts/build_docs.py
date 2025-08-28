#!/usr/bin/env python3
"""
Documentation build script for GeminiSDR.

This script automates the documentation build process, including:
- API documentation generation
- HTML build
- Link checking
- Documentation validation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    
    return result

def generate_api_docs(docs_dir):
    """Generate comprehensive API documentation using sphinx-apidoc."""
    print("Generating comprehensive API documentation...")
    
    # Generate API docs for geminisdr package
    geminisdr_api_dir = docs_dir / "api" / "geminisdr"
    geminisdr_api_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "sphinx-apidoc",
        "-o", str(geminisdr_api_dir),
        "../geminisdr",
        "--force",
        "--separate",
        "--module-first",
        "--maxdepth", "4"
    ]
    
    run_command(cmd, cwd=docs_dir)
    
    # Generate API docs for legacy modules
    legacy_modules = ["core", "ml", "config", "environments"]
    
    for module in legacy_modules:
        module_path = docs_dir.parent / module
        if module_path.exists():
            module_api_dir = docs_dir / "api" / module
            module_api_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "sphinx-apidoc",
                "-o", str(module_api_dir),
                str(module_path),
                "--force",
                "--separate",
                "--module-first"
            ]
            
            run_command(cmd, cwd=docs_dir)
    
    print("API documentation generation completed!")

def build_html(docs_dir):
    """Build HTML documentation."""
    print("Building HTML documentation...")
    
    cmd = ["make", "html"]
    run_command(cmd, cwd=docs_dir)

def check_links(docs_dir):
    """Check documentation links."""
    print("Checking documentation links...")
    
    cmd = ["make", "linkcheck"]
    result = run_command(cmd, cwd=docs_dir, check=False)
    
    if result.returncode != 0:
        print("Warning: Some links may be broken. Check the output above.")

def clean_build(docs_dir):
    """Clean build directory."""
    print("Cleaning build directory...")
    
    cmd = ["make", "clean"]
    run_command(cmd, cwd=docs_dir)

def main():
    parser = argparse.ArgumentParser(description="Build GeminiSDR documentation")
    parser.add_argument("--clean", action="store_true", help="Clean build directory first")
    parser.add_argument("--api", action="store_true", help="Generate API docs only")
    parser.add_argument("--html", action="store_true", help="Build HTML only")
    parser.add_argument("--check-links", action="store_true", help="Check links only")
    parser.add_argument("--all", action="store_true", help="Run all steps (default)")
    
    args = parser.parse_args()
    
    # Default to all if no specific step is requested
    if not any([args.api, args.html, args.check_links]):
        args.all = True
    
    # Get docs directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs"
    
    if not docs_dir.exists():
        print(f"Error: Documentation directory not found at {docs_dir}")
        sys.exit(1)
    
    # Change to project root for imports to work
    os.chdir(project_root)
    
    try:
        if args.clean or args.all:
            clean_build(docs_dir)
        
        if args.api or args.all:
            generate_api_docs(docs_dir)
        
        if args.html or args.all:
            build_html(docs_dir)
        
        if args.check_links or args.all:
            check_links(docs_dir)
        
        print("\nDocumentation build completed!")
        print(f"HTML documentation is available at: {docs_dir / '_build' / 'html' / 'index.html'}")
        
    except KeyboardInterrupt:
        print("\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during build: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()