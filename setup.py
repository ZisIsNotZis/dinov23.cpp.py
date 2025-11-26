from setuptools import setup, find_packages
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="dinov23-cpp-py",
    version="0.1.0",
    packages=find_packages(),# Automatically find all packages (e.g., my_package)
    install_requires=[i for i in requirements if 'git' not in i],
    dependency_links=[i for i in requirements if 'git' in i],
    author="ZisIsNotZis",
    author_email="ZisIsNotZis@Gmail.com",
    description="A python ggml implementation of dinov2 and dinov3",
    url="https://github.com/ZisIsNotZis/dinov23.cpp.py",
)