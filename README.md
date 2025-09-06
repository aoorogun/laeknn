# laeknn

**Locally Adaptive Evidential K‑Nearest Neighbours (LAE‑KNN)** is a machine learning library
implementing a novel variant of the classical K‑Nearest Neighbours algorithm.  The
model combines local metric learning, density‑tempered kernel weighting and
Dirichlet evidential aggregation to provide probability estimates and
uncertainty alongside predictions.

## Key features

* Learns a local Mahalanobis metric from nearby coreset points to adapt to
  anisotropic data distributions.
* Applies a density‑tempered kernel to reduce the influence of crowded regions.
* Aggregates neighbour information as evidence for each class and outputs a
  Dirichlet distribution, providing both class probabilities and uncertainty.
* Supports reduction of the training set via simple coreset construction to
  speed up predictions and reduce memory usage.

## Real‑world analogy

Imagine you arrive in a new town and want to decide which restaurant to try.
Rather than asking everyone, you pick a handful of locals from nearby streets
(*the coreset*), consider how people are clustered (*density*), adjust your
perception of distance based on the street layout (*local metric*), and weigh
their recommendations accordingly.  You also keep track of how confident you
are about their suggestions (*uncertainty mass*).  LAE‑KNN formalises this
intuition in a simple machine learning model.

## Basic usage

The library exposes a single class `LAEKNN`.

for terminal:
```python
pip install git+https://github.com/aoorogun/laeknn.git
```
for Jupyter notebook:
```python
!pip install git+https://github.com/aoorogun/laeknn.git
```

sample code using the iris dataset
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from laeknn import LAEKNN

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


clf = LAEKNN(per_class_centers=3, m=6, k_density=6, beta=0.4, tau=0.8, lam=1e-2)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
```

## Credits

This package was developed by **Okunola Orogun** of **Endow Tech Limited**.

## License

This project is licensed under the MIT License – see the `LICENSE` file for
details.
