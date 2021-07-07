import abc
from typing import Hashable, Iterable, Iterator, Sequence, Union
import numpy as np

from smqtk_core import Plugfigurable

from smqtk_classifier.interfaces.classification_element import (
    CLASSIFICATION_DICT_T
)


IMAGE_ITER_T = Union[np.ndarray, Iterable[np.ndarray]]


class ClassifyImage (Plugfigurable):
    """
    Interface for algorithms that classify input images into discrete
    labels and/or label confidences. Images are expected to be formatted
    in the format of `np.ndarray` matrices.
    """

    @abc.abstractmethod
    def get_labels(self) -> Sequence[Hashable]:
        """
        Get the sequence of class labels that this classifier can classify
        images into. This includes the negative or background label if the
        classifier embodies such a concept.

        :return: Sequence of possible classifier labels.

        :raises RuntimeError: No model loaded.
        """

    @abc.abstractmethod
    def classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
        """
        Classify an input iterable of images, in the form of `np.ndarray`
        matricies into a parallel iterable of label-to-confidence mappings
        (dictionaries).

        We expect input image matrices to come in either the `[H, W]` or
        `[H, W, C]` dimension formats.

        Each classification mapping should contain confidence values for each
        label the configured model contains.
        Implementations may act in a discrete manner whereby only one label is
        marked with a ``1`` value (others being ``0``), or in a continuous
        manner whereby each label is given a confidence-like value in the
        [0, 1] range.

        :param  array_iter: Iterable of images, as numpy arrays, to be
            classified.

        :raises ValueError: Input arrays were not all of consistent
            dimensionality.

        :return: Iterator of dictionaries, parallel in association to the input
            images. Each dictionary should map labels to associated
            confidence values.
        """
