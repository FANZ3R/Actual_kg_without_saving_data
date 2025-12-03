"""
Setup script for Fast Knowledge Graph Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="fast-kg-pipeline",
    version="1.0.0",
    description="Fast Knowledge Graph Extraction Pipeline with sub-second queries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Data Science Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "PyPDF2>=3.0.0",
        "python-docx>=1.1.0",
        "openpyxl>=3.1.0",
        "neo4j>=5.0.0",
        "cachetools>=5.3.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'run-kg-pipeline=scripts.run_pipeline:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)