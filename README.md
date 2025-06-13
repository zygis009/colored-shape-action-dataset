# Simple moving shape dataset generator

This is a simple project used to generate datasets of colorful moving shape videos, which can be used for toy experiments in computer vision tasks, such as action recognition, privacy preservation, etc. Example dataset with shapes used for generation are provided in the `data` directory.

## Setup

Install the requirements file (optionally set up a [virtual python environment](https://docs.python.org/3/library/venv.html) beforehand):

```bash
pip install -r requirements.txt
```

## Configuration

While the functions accept mostly accept custom arguments for configuring how the dataset is generated, currently the color possibilities and speed settings for shape movement are hardcoded at the top of `generator.py` file. In case customization is desired, add your custom values directly to the code.

## Running

To run the dataset generation, either clone the repository and run the bash script:

```bash
python generator.py --shapedir=<YOUR_SHAPES_DIR> --outdir=<YOUR_OUTPUT_DIR> ...
```

For other arguments, see the `parse_args()` function in `generator.py` or run `python generator.py --help` in your shell.

It is also possible to use the generator externaly in your project. First install the package with pip:

```bash
pip install git+https://github.com/zygis009/colored-shape-action-dataset.git
```

Then import the desired generator function:

```python
from ... import generate_dataset, get_frames
```

## Contributing

If you'd like to contribute:

1. Fork this GitHub repository.

2. Create a new branch (feature-branch or bugfix-branch).

3. Make your changes and ensure the code follows best practices.

4. Submit a pull request with a clear description of your changes.
