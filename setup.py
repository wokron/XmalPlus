from setuptools import setup, find_packages

setup(
    name="xmal-plus",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "torch",
        "androguard==3.3.5",
    ],
)
