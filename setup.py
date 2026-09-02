"""
setup.py — make the Hybrid RAG package installable.

Install in development mode:
    pip install -e .

Then import from any project:
    from v24 import build_large_corpus_engine
"""

from setuptools import setup, find_packages

setup(
    name="hybrid-rag",
    version="2.3.0",
    description="Hybrid RAG with TF-IDF + Entity Graph + RRF + Cross-Encoder reranking",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    py_modules=["v24"],  # v23.py is a single-module package
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pyarrow>=12.0",
        "scikit-learn>=1.3",
    ],
    extras_require={
        "ner": ["spacy>=3.7"],
        "reranker": ["sentence-transformers>=2.2"],
        "config": ["PyYAML>=6.0"],
        "openai": ["openai>=1.0"],
        "all": [
            "spacy>=3.7",
            "sentence-transformers>=2.2",
            "PyYAML>=6.0",
            "openai>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybrid-rag-eval=batch_compare_v2:main",
        ],
    },
)
