import abc
from typing import Mapping, Hashable

from .classify_image import ClassifyImage, IMAGE_ITER_T
from smqtk_classifier.exceptions import ExistingModelError


class ClassifyImageSupervised(ClassifyImage):
    """
    Class of classifiers that are trainable via supervised training, i.e. are
    given specific Image examples for class labels.
    """

    @abc.abstractmethod
    def has_model(self) -> bool:
        """
        :return: If this instance currently has a model loaded. If no model is
            present, classification of images cannot happen (needs to be
            trained).
        """

    def train(
        self,
        class_examples: Mapping[Hashable, IMAGE_ITER_T]
    ) -> None:
        """
        Train the supervised classifier model.

        If a model is already loaded, we will raise an exception in order to
        prevent accidental overwrite.

        If the same label is provided to both ``class_examples`` and ``kwds``,
        the examples given to the reference in ``kwds`` will prevail.

        :param class_examples: Dictionary mapping class labels to iterables of
            Image training examples.
        :raises ValueError: There were no class examples provided.
        :raises ValueError: Less than 2 classes were given.
        :raises RuntimeError: A model already exists in this instance.
            Following through with training would overwrite this model.
            Throwing an exception for information protection.
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

        return self._train(class_examples)

    @abc.abstractmethod
    def _train(
        self,
        class_examples: Mapping[Hashable, IMAGE_ITER_T]
    ) -> None:
        """
        Internal method that trains the classifier implementation.

        This method is called after checking that there is not already a model
        trained, thus it can be assumed that no model currently exists.

        The class labels will have already been checked before entering this
        method, so it can be assumed that the ``class_examples`` will container
        at least two classes.

        :param class_examples: Dictionary mapping class labels to iterables of
            Image training examples.
        """
