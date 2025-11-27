from setuptools import setup, find_packages

setup(
    name="dinov23-cpp-py",
    version="0.1.0",
    packages=find_packages(),  # Automatically find all packages (e.g., my_package)
    install_requires='ggml-python scikit-learn Pillow'.split(),
    author="ZisIsNotZis",
    author_email="ZisIsNotZis@Gmail.com",
    description="A python ggml implementation of dinov2 and dinov3",
    url="https://github.com/ZisIsNotZis/dinov23.cpp.py",
)
