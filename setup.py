from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="yolo-tiling",
    version="0.1",
    author="Jordan Pierce",
    author_email="jordan.pierce@example.com",
    description="A package for tiling YOLO datasets for small object detection and instance segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jordan-Pierce/yolo-tiling",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "Pillow",
        "Shapely",
    ],
)
