from setuptools import setup
from pathlib import Path

setup(
    name="colored-shape-action-dataset",
    version="1.0.0",
    description="",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
    author="Žygimantas Liutkus",
    author_email="zygis009@gmail.com",
    url="https://github.com/zygis009/colored-shape-action-dataset",
    scripts=["generator.py"],
    requires=[
        "pillow==10.3.0",
        "opencv-python==4.9.0.80",
        "matplotlib==3.10.0",
        "numpy==1.26.4",
    ],
    license="Apache 2.0 license"
)