"""
Setup file for CS5416 Final Project - ML Inference Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cs5416-ml-pipeline",
    version="1.0.0",
    author="CS5416 Final Project",
    description="Distributed ML Inference Pipeline for Customer Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8,<3.12",
    install_requires=requirements,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

