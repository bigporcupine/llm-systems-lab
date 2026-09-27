"""Compatibility shim for editable installs with older pip versions."""

from setuptools import find_packages, setup


setup(
    name="llm-systems-lab",
    version="0.1.0",
    description="A reproducible, experiment-driven lab for understanding and optimizing LLM systems.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    entry_points={"console_scripts": ["llms-lab=llm_systems_lab.cli:main"]},
)
