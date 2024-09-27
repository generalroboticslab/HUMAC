# Algorithms
Algorithms for Dojo

## Setup

This project uses [Poetry](https://python-poetry.org/) to manage dependencies. Follow the instructions on the Poetry page to properly install it.

Then run the command to install all of the dependencies:
```bash
poetry install
```

Then to activate the shell run the following command:
```bash
poetry shell
```

Since torchrl & tensodict are still in active development, it is best to install them from their repos:
```bash
pip install git+https://github.com/pytorch/rl.git@7ae614043b67bfccf849f661491587339e3262a2
pip install git+https://github.com/pytorch-labs/tensordict

conda install -c conda-forge pyaudio
```

Then you can run all of the commands in this repository.

If you want to contribute to this repository, you can install the pre-commit hooks for some nice features:
```bash
poetry run pre-commit install
```

export PATH="$/Users/michael/.local/bin.poetry/bin:$PATH"