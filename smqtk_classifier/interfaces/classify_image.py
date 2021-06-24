import abc
import logging
from typing import Hashable, Iterable, Iterator, Sequence, Union
import numpy as np

from smqtk_core import Plugfigurable

from smqtk_classifier.interfaces.classification_element import (
    CLASSIFICATION_DICT_T
)


IMAGE_ITER_T = Union[np.ndarray, Iterable[np.ndarray]]
LOG = logging.getLogger(__name__)


class ClassifyImage (Plugfigurable):
    """
    Interface for algorithms that classify input images into discrete
    labels and/or label confidences. Images are expected to be formatted
    in numpy.ndarray. Any assumptions about shapes must be handled on the
    implementation side and will not be tested by the interface.
    """

    @abc.abstractmethod
    def get_labels(self) -> Sequence[Hashable]:
        """
        Get the sequence of class labels that this classifier can classify
        images into. This includes the negative or background label if the
        classifier embodies such a concept.
        """

    @abc.abstractmethod
    def _classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
        """
        Overridable method for classifying an iterable of
        np.ndarray whose images should be classified.

        *Remember:* A single-pass `Iterator` is a valid `Iterable`. If an
        implementation needs to pass over the input multiple times, either
        ensure you are receiving an ndarray, or *not* and Iterator.

        Each classification mapping should contain confidence values for each
        label the configured model contains.
        Implementations may act in a discrete manner whereby only one label is
        marked with a ``1`` value (others being ``0``), or in a continuous
        manner whereby each label is given a confidence-like value in the
        [0, 1] range.
        """

    def classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
        """
        Classify an input iterable of np.ndarray into a parallel iterable of
        label-to-confidence mappings (dictionaries).

        Each classification mapping should contain confidence values for each
        label the configured model contains.
        Implementations may act in a discrete manner whereby only one label is
        marked with a ``1`` value (others being ``0``), or in a continuous
        manner whereby each label is given a confidence-like value in the
        [0, 1] range.
        """
        return self._classify_images(
            img_iter
        )
