from setuptools import setup, find_packages

setup(
    name="signalrag",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.1",
        "transformers>=4.46.3",
        "accelerate>=1.1.1",
        "bitsandbytes>=0.45.0",
        "sentence-transformers>=3.3.1",
        "faiss-cpu>=1.9.0",
        "datasets>=3.1.0",
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "scikit-learn>=1.5.2",
        "xgboost>=2.1.3",
        "rouge-score>=0.1.2",
        "groq>=0.13.1",
        "tqdm>=4.67.0",
        "python-dotenv>=1.0.1",
    ],
)
