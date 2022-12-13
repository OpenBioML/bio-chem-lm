#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    "transformers[sentencepiece]",
    "tokenizers",
    "tqdm",
    "torch",
    "datasets",
    "matplotlib",
    "mup",
    "rotary-embedding-torch",
    "einops",
    "isort",
    "black",
    "wandb",
    "accelerate",
    "torchmetrics",
    "deepspeed"
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Zach Nussbaum",
    author_email="zanussbaum@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="LLM for bio",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords="bio_lm",
    name="bio_lm",
    packages=find_packages(include=["bio_lm", "bio_lm.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/OpenBioML/bio-chem-lm",
    version="0.1.0",
    zip_safe=False,
)
