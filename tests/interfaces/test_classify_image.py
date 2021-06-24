from typing import Any, Dict, Hashable, Iterator, Sequence
import unittest
import unittest.mock as mock
import numpy as np

from smqtk_classifier import ClassifyImage

from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_classifier.interfaces.classify_image import IMAGE_ITER_T


class DummyClassifier (ClassifyImage):

    EXPECTED_LABELS = ['constant']

    def __init__(self) -> None:
        super().__init__()
        # Mock "method" for testing functionality is called post-final-yield.
        self.post_iterator_check = mock.Mock()

    @classmethod
    def is_usable(cls) -> bool:
        return True

    def get_config(self) -> Dict[str, Any]:
        return {}

    def get_labels(self) -> Sequence[Hashable]:
        return self.EXPECTED_LABELS

    def _classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
        """
        Some deterministic dummy impl
        Simply returns a classification with one label "test" whose value is
        the first image
        """
        for v in img_iter:
            yield {'test': v}
        self.post_iterator_check()

    def _classify_too_few(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
        """ Swap-in for _classify_images that under-generates."""
        # Yield all but one
        image_list = list(img_iter)
        for i, v in enumerate(image_list[:-1]):
            yield {'test': i}
        self.post_iterator_check()

    def _classify_too_many(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
        """ Swap-in for _classify_images that over-generates."""
        i = 0
        for i, v in enumerate(img_iter):
            yield {'test': i}
        # Yield some extra stuff
        yield {'test': i+1}
        yield {'test': i+2}
        self.post_iterator_check()


class TestClassifierAbstractClass (unittest.TestCase):

    def setUp(self) -> None:
        # Common dummy instance setup per test case.
        self.inst = DummyClassifier()

    def test_classify_images_empty_iter(self) -> None:
        """ Test that passing an empty iterator correctly yields another empty
        iterator."""
        images: IMAGE_ITER_T = []
        assert list(self.inst.classify_images(images)) == []
        self.inst.post_iterator_check.assert_called_once()  # type: ignore

    def test_classify_images(self) -> None:
        """ Test "successful" function of classify images. """
        images = [np.zeros([100, 100, 3], dtype=np.uint8)]
        list(self.inst.classify_images(images))
        # noinspection PyUnresolvedReferences
        self.inst.post_iterator_check.assert_called_once()
