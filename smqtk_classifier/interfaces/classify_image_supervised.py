import abc
from typing import Any, Iterable, Mapping, Hashable
import numpy as np

from .classify_image import ClassifyImage
from smqtk_classifier.exceptions import ExistingModelError


class ClassifyImageSupervised(ClassifyImage):
    """
    Class of classifiers that are trainable via supervised training, i.e. are
    given specific Image examples for class labels.
    """

    @abc.abstractmethod
    def has_model(self) -> bool:
        """
        If this instance currently has a model loaded. If no model is
        present, classification of Images cannot happen (needs to be trained).
        """

    def train(
        self,
        class_examples: Mapping[Hashable, Iterable[np.ndarray]],
        **extra_params: Any
    ) -> Iterable[np.ndarray]:
        """
        Train the supervised classifier model.

        If a model is already loaded, we will raise an exception in order to
        prevent accidental overwrite.

        If the same label is provided to both ``class_examples`` and ``kwds``,
        the examples given to the reference in ``kwds`` will prevail.
        """
        if self.has_model():
            raise ExistingModelError("Instance currently has a model. Halting "
                                     "training to prevent overwrite of "
                                     "existing trained model.")

        if not class_examples:
            raise ValueError("No class examples were provided.")
        elif len(class_examples) < 2:
            raise ValueError("Need 2 or more classes for training. Given %d."
                             % len(class_examples))

        return self._train(class_examples, **extra_params)

    @abc.abstractmethod
    def _train(
        self,
        class_examples: Mapping[Hashable, Iterable[np.ndarray]],
        **extra_params: Any
    ) -> Iterable[np.ndarray]:
        """
        Internal method that trains the classifier implementation.

        This method is called after checking that there is not already a model
        trained, thus it can be assumed that no model currently exists.

        The class labels will have already been checked before entering this
        method, so it can be assumed that the ``class_examples`` will container
        at least two classes.
        """
