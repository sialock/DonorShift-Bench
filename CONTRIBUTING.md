# Contributing

Python 3.10+; create a virtual environment, `pip install -e '.[dev]'`, and run `pytest`.
Keep all fitting of supervised or unsupervised trainable transformations inside each donor fold. Any model/hyperparameter selection must be group-nested to avoid leakage. Add test cases for sparse and dense counts. Never add private patient records or controlled access data to the repository. Keep model comparisons reproducible with explicit seeds, documented sample selection, package versions, and clear limits of biological interpretation.