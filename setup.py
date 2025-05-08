from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="weight-of-thought",
    version="0.1.0",
    author="Saif Punjwani",
    author_email="spunjwani3@gatech.edu",
    description="Weight-of-Thought Reasoning: Exploring Neural Network Weights for Enhanced LLM Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SaifPunjwani/weight-of-thought",
    project_urls={
        "Bug Tracker": "https://github.com/SaifPunjwani/weight-of-thought/issues",
        "Documentation": "https://github.com/SaifPunjwani/weight-of-thought#readme",
        "Paper": "https://arxiv.org/abs/2504.10646",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
    ],
    packages=find_packages(include=["wot", "wot.*"]),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.12.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "networkx>=2.5.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.8.0",
            "flake8>=3.9.0",
            "mypy>=0.812",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
        ],
        "visualization": [
            "ipywidgets>=7.6.0",
            "plotly>=5.0.0",
            "tqdm>=4.61.0",
        ],
    },
)