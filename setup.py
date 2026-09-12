from setuptools import setup, find_packages

setup(
    name="qos_gnn_recommendation",
    version="1.0.0",
    author="Academic Research Group",
    description="QoS-Driven Service Recommendation Benchmarking: Sparse Matrix Factorization vs Graph Neural Networks with Cost-Performance Tradeoff",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "torch>=2.0.0",
    ],
)
