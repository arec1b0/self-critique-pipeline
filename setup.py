from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="self-critique-pipeline",
    version="1.0.0",
    author="MLOps Engineer",
    author_email="mlops@example.com",
    description="Production-ready Self-Critique Chain Pipeline for research paper summarization using Claude AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/self-critique-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.3",
            "pytest-cov>=5.0.0",
            "black>=24.8.0",
            "flake8>=7.1.1",
            "mypy>=1.11.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "self-critique=api.main:main",
        ],
    },
)