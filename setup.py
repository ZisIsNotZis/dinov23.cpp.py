from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="dinov23-cpp-py",  # Must be unique on PyPI (use hyphens)
    version="0.1.0",         # Semantic versioning (MAJOR.MINOR.PATCH)
    packages=find_packages(),# Automatically find all packages (e.g., my_package)
    install_requires=requirements,  # Use dependencies from requirements.txt
    author="ZisIsNotZis",
    author_email="ZisIsNotZis@Gmail.com",
    description="A python ggml implementation of dinov2 and dinov3",
    url="https://github.com/ZisIsNotZis/dinov23.cpp.py",  # Optional (link to repo)
)