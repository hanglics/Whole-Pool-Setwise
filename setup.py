from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='whole-pool-setwise',
    version='0.1.0',
    packages=find_packages(),
    url='',
    license='Apache 2.0',
    author='',
    author_email='',
    description='Whole-pool Setwise re-ranking experiments with local LLMs.',
    python_requires='>=3.8',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "transformers>=4.31.0",
    ]
)
