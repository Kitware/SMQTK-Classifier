import os
import unittest

from smqtk_core.configuration import configuration_test_helper
import numpy
import pytest

from smqtk_classifier import ClassifyDescriptor
from smqtk_classifier.impls.classify_descriptor.classify_index_label_descriptor import ClassifyIndexLabelDescriptor

from tests import TEST_DATA_DIR


class TestClassifyIndexLabelDescriptor(unittest.TestCase):

    EXPECTED_LABEL_VEC = [
        b'label_1',
        b'label_2',
        b'negative',
        b'label_3',
        b'Kitware',
        b'label_4',
    ]

    FILEPATH_TEST_LABELS = os.path.join(TEST_DATA_DIR, 'test_labels.txt')

    def test_is_usable(self) -> None:
        # Should always be available
        self.assertTrue(ClassifyIndexLabelDescriptor.is_usable())

    def test_impl_findable(self) -> None:
        self.assertIn(ClassifyIndexLabelDescriptor,
                      ClassifyDescriptor.get_impls())

    def test_configurable(self) -> None:
        c = ClassifyIndexLabelDescriptor(self.FILEPATH_TEST_LABELS)
        for inst in configuration_test_helper(c):
            assert inst.index_to_label_uri == self.FILEPATH_TEST_LABELS

    def test_new(self) -> None:
        c = ClassifyIndexLabelDescriptor(self.FILEPATH_TEST_LABELS)
        self.assertEqual(c.label_vector, self.EXPECTED_LABEL_VEC)

    def test_get_labels(self) -> None:
        c = ClassifyIndexLabelDescriptor(self.FILEPATH_TEST_LABELS)
        self.assertEqual(c.get_labels(), self.EXPECTED_LABEL_VEC)

    def test_configuration(self) -> None:
        cfg = ClassifyIndexLabelDescriptor.get_default_config()
        self.assertEqual(cfg, {'index_to_label_uri': None})

        cfg['index_to_label_uri'] = self.FILEPATH_TEST_LABELS
        c = ClassifyIndexLabelDescriptor.from_config(cfg)
        self.assertEqual(c.get_config(), cfg)

    def test_classify_arrays(self) -> None:
        c = ClassifyIndexLabelDescriptor(self.FILEPATH_TEST_LABELS)
        c_expected = {
            b'label_1': 1,
            b'label_2': 2,
            b'negative': 3,
            b'label_3': 4,
            b'Kitware': 5,
            b'label_4': 6,
        }

        a = numpy.array([1, 2, 3, 4, 5, 6])
        c_result = list(c._classify_arrays([a]))[0]
        self.assertEqual(c_result, c_expected)

    def test_classify_arrays_invalid_descriptor_dimensions(self) -> None:
        c = ClassifyIndexLabelDescriptor(self.FILEPATH_TEST_LABELS)

        # One less
        a = numpy.array([1, 2, 3, 4, 5])
        with pytest.raises(RuntimeError):
            list(c._classify_arrays([a]))

        # One more
        a = numpy.array([1, 2, 3, 4, 5, 6, 7])
        with pytest.raises(RuntimeError):
            list(c._classify_arrays([a]))
