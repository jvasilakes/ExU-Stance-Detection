import numpy as np


def total_variation(a, b, axis=0, norm=False):
    """
    Total variation distance of probability measures
    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures

    a, b: vector of probabilities for a single example
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    # Ensure both vectors are probabilities
    assert (a.min(axis) >= 0.0).all(), f"{a.min(axis)}"
    assert (a.max(axis) <= 1.0).all(), f"{a.max(axis)}"
    assert (b.min(axis) >= 0.0).all(), f"{b.min(axis)}"
    assert (b.max(axis) <= 1.0).all(), f"{b.max(axis)}"
    tv = np.max(np.abs(a - b), axis)
    if norm is True:
        # Direction of greatest positive change
        idxs = np.argmax(b - a, axis, keepdims=True)
        # Maximum possible value in direction of change
        denom = 1. - np.take_along_axis(a, idxs, axis=axis).flatten()
        denom[denom == 0.0] = 1.0
        tv = tv / denom
    if tv.shape == (1, ):
        tv = tv.item()
    return tv


def stubbornness(init, trained, axis=0, norm=True):
    """
    a, b: vector of probabilities for a single example

    stubbornness in [0, 1]
    0: The model is perfectly unstubborn
       (i.e., it is capable of inverting its 0-shot predictions)
    1: The model is perfectly stubborn
       (i.e., it does not change its 0-shot predictions at all)
    """
    return 1.0 - total_variation(init, trained, axis=axis, norm=norm)


def teachability(init, trained, y, axis=0):
    """
    init: Initial vector of probabilities for a single example.
    trained: Fine-tuned/trained vector of probabilities for a single example.
    y: one-hot encoded gold-standard label

    teachability in [-1, 1]
    -1: The model is perfectly stubborn and therefore unable to be taught
    teachability < 0: The model changes but shifts probability
                      away from the training signal
    teachability = 0: The model changes but does not re-assign
                      any probability to the training signal
    teachability > 0: The model changes and shifts probability
                      towards the training signal
    1: The model is perfectly unstubborn and shifts all
       probability to the training signal
    """
    tv_init = total_variation(init, y, axis=axis, norm=False)
    tv_trained = total_variation(trained, y, axis=axis, norm=False)
    return tv_init - tv_trained
