import unittest
import unittest.mock as mock

import numpy as np

from smqtk_classifier import ClassifyDescriptorCollection
from smqtk_classifier.exceptions import MissingLabelError
from smqtk_classifier.impls.classification_element.memory import MemoryClassificationElement

from smqtk_descriptors.impls.descriptor_element.memory import DescriptorMemoryElement

from tests.interfaces.test_classify_descriptor import DummyClassifier


class TestClassifyDescriptorCollection (unittest.TestCase):

    ##########################################################################
    # Constructor Tests

    def test_new_empty(self) -> None:
        ccol = ClassifyDescriptorCollection()
        self.assertEqual(ccol._label_to_classifier, {})

    def test_new_not_classifier_positional(self) -> None:
        # First invalid key should be in error message.
        self.assertRaisesRegex(
            ValueError,
            "for key 'some label'",
            ClassifyDescriptorCollection,
            classifiers={'some label': 0}
        )

    def test_new_not_classifier_kwarg(self) -> None:
        # First invalid key should be in error message.
        self.assertRaisesRegex(
            ValueError,
            "for key 'some_label'",
            ClassifyDescriptorCollection,
            some_label=0
        )

    def test_new_positional(self) -> None:
        c = DummyClassifier()
        ccol = ClassifyDescriptorCollection(classifiers={'a label': c})
        self.assertEqual(ccol._label_to_classifier, {'a label': c})

    def test_new_kwargs(self) -> None:
        c = DummyClassifier()
        ccol = ClassifyDescriptorCollection(a_label=c)
        self.assertEqual(ccol._label_to_classifier, {'a_label': c})

    def test_new_both_pos_and_kwds(self) -> None:
        c1 = DummyClassifier()
        c2 = DummyClassifier()
        ccol = ClassifyDescriptorCollection({'a': c1}, b=c2)
        self.assertEqual(ccol._label_to_classifier,
                         {'a': c1, 'b': c2})

    def test_new_duplicate_label(self) -> None:
        c1 = DummyClassifier()
        c2 = DummyClassifier()
        self.assertRaisesRegex(
            ValueError,
            "Duplicate classifier label 'c'",
            ClassifyDescriptorCollection,
            {'c': c1},
            c=c2
        )

    ##########################################################################
    # Configuration Tests

    def test_get_default_config(self) -> None:
        # Returns a non-empty dictionary with just the example key. Contains
        # a sub-dictionary that would container the implementation
        # specifications.
        c = ClassifyDescriptorCollection .get_default_config()

        # Should just contain the default example
        self.assertEqual(len(c), 1)
        self.assertIn('__example_label__', c.keys())
        # Should be a plugin config after this.
        self.assertIn('type', c['__example_label__'])

    def test_get_config_empty(self) -> None:
        # The config coming out of an empty collection should be an empty
        # dictionary.
        ccol = ClassifyDescriptorCollection()
        self.assertEqual(ccol.get_config(), {})

    def test_get_config_with_stuff(self) -> None:
        c1 = DummyClassifier()
        c2 = DummyClassifier()
        ccol = ClassifyDescriptorCollection({'a': c1}, b=c2)
        # dummy returns {} config.
        self.assertEqual(
            ccol.get_config(),
            {
                'a': {
                    'type': 'tests.interfaces.test_classify_descriptor.DummyClassifier',
                    'tests.interfaces.test_classify_descriptor.DummyClassifier': {},
                },
                'b': {
                    'type': 'tests.interfaces.test_classify_descriptor.DummyClassifier',
                    'tests.interfaces.test_classify_descriptor.DummyClassifier': {},
                }
            }
        )

    def test_from_config_empty(self) -> None:
        ccol = ClassifyDescriptorCollection .from_config({})
        self.assertEqual(ccol._label_to_classifier, {})

    def test_from_config_skip_example_key(self) -> None:
        # If the default example is left in the config, it should be skipped.
        # The string chosen for the example key should be unlikely to be used
        # in reality.
        ccol = ClassifyDescriptorCollection .from_config({
            '__example_label__':
                'this should be skipped regardless of content'
        })
        self.assertEqual(ccol._label_to_classifier, {})

    @mock.patch('smqtk_classifier.interfaces.classify_descriptor.ClassifyDescriptor.get_impls')
    def test_from_config_with_content(self, m_get_impls: mock.MagicMock) -> None:
        # Mocking implementation getter to only return the dummy
        # implementation.
        m_get_impls.side_effect = lambda: {DummyClassifier}
        ccol = ClassifyDescriptorCollection .from_config({
            'a': {
                'type': 'tests.interfaces.test_classify_descriptor.DummyClassifier',
                'tests.interfaces.test_classify_descriptor.DummyClassifier': {},
            },
            'b': {
                'type': 'tests.interfaces.test_classify_descriptor.DummyClassifier',
                'tests.interfaces.test_classify_descriptor.DummyClassifier': {},
            },
        })
        self.assertEqual(
            # Using sort because return from ``keys()`` has no guarantee on
            # order.
            sorted(ccol._label_to_classifier.keys()), ['a', 'b']
        )
        self.assertIsInstance(ccol._label_to_classifier['a'], DummyClassifier)
        self.assertIsInstance(ccol._label_to_classifier['b'], DummyClassifier)

    ##########################################################################
    # Accessor Method Tests

    def test_size_len(self) -> None:
        ccol = ClassifyDescriptorCollection()
        self.assertEqual(ccol.size(), 0)
        self.assertEqual(len(ccol), 0)

        ccol = ClassifyDescriptorCollection(
            a=DummyClassifier(),
            b=DummyClassifier(),
        )
        self.assertEqual(ccol.size(), 2)
        self.assertEqual(len(ccol), 2)

    def test_labels_empty(self) -> None:
        ccol = ClassifyDescriptorCollection()
        self.assertEqual(ccol.labels(), set())

    def test_labels(self) -> None:
        ccol = ClassifyDescriptorCollection(
            classifiers={
                'b': DummyClassifier(),
            },
            a=DummyClassifier(),
            label2=DummyClassifier(),
        )
        self.assertEqual(ccol.labels(), {'a', 'b', 'label2'})

    def test_add_classifier_not_classifier(self) -> None:
        # Attempt adding a non-classifier instance
        ccol = ClassifyDescriptorCollection()
        # The string 'b' is not a classifier instance.
        self.assertRaisesRegex(
            ValueError,
            "Not given a Classifier instance",
            ccol.add_classifier,
            'a', 'b'
        )

    def test_add_classifier_duplicate_label(self) -> None:
        ccol = ClassifyDescriptorCollection(a=DummyClassifier())
        self.assertRaisesRegex(
            ValueError,
            "Duplicate label provided: 'a'",
            ccol.add_classifier,
            'a', DummyClassifier()
        )

    def test_add_classifier(self) -> None:
        ccol = ClassifyDescriptorCollection()
        self.assertEqual(ccol.size(), 0)

        c = DummyClassifier()
        ccol.add_classifier('label', c)
        self.assertEqual(ccol.size(), 1)
        self.assertEqual(ccol._label_to_classifier['label'], c)

    def test_get_classifier_bad_label(self) -> None:
        c = DummyClassifier()
        ccol = ClassifyDescriptorCollection(a=c)
        self.assertRaises(
            KeyError,
            ccol.get_classifier,
            'b'
        )

    def test_get_classifier(self) -> None:
        c = DummyClassifier()
        ccol = ClassifyDescriptorCollection(a=c)
        self.assertEqual(ccol.get_classifier('a'), c)

    def test_remove_classifier_bad_label(self) -> None:
        c = DummyClassifier()
        ccol = ClassifyDescriptorCollection(a=c)
        self.assertRaises(
            KeyError,
            ccol.remove_classifier, 'b'
        )

    def test_remove_classifier(self) -> None:
        c = DummyClassifier()
        ccol = ClassifyDescriptorCollection(a=c)
        ccol.remove_classifier('a')
        self.assertEqual(ccol._label_to_classifier, {})

    ##########################################################################
    # Classification Method Tests

    def test_classify(self) -> None:
        """ Test invoking `classify` in a valid manner. """
        ccol = ClassifyDescriptorCollection({
            'subjectA': DummyClassifier(),
            'subjectB': DummyClassifier(),
        })

        d_v = np.array([0, 1, 2, 3, 4])
        d = DescriptorMemoryElement('0')
        d.set_vector(d_v)
        result = ccol.classify(d)

        # Should contain one entry for each configured classifier.
        self.assertEqual(len(result), 2)
        self.assertIn('subjectA', result)
        self.assertIn('subjectB', result)
        # Each key should map to a classification element (memory in this case
        # because we're using the default factory)
        self.assertIsInstance(result['subjectA'], MemoryClassificationElement)
        self.assertIsInstance(result['subjectB'], MemoryClassificationElement)
        # We know the dummy classifier outputs "classifications" in a
        # deterministic way: class label is "test" and classification
        # value is the index of the descriptor .
        self.assertDictEqual(result['subjectA'].get_classification(),
                             {'test': 0})
        self.assertDictEqual(result['subjectB'].get_classification(),
                             {'test': 0})

    def test_classify_arrays(self) -> None:
        """ Test invoking `classify_arrays` in a valid manner. """
        # Use some dummy classifiers that
        ccol = ClassifyDescriptorCollection({
            'subjectA': DummyClassifier(),
            'subjectB': DummyClassifier(),
        })
        dmat = np.asarray([[0, 1, 2, 3, 4],
                           [5, 6, 7, 8, 9]])
        result = ccol.classify_arrays(dmat)
        # Should contain one entry for each configured classifier.
        assert len(result) == 2
        assert 'subjectA' in result
        assert 'subjectB' in result
        # Each key should map to a list of dictionaries mapping classifier
        # labels to confidence values. The DummyClassify maps input descriptor
        # first values as the "confidence" for simplicity. Should be in order
        # of input vectors.
        assert result['subjectA'] == [{'test': 0}, {'test': 5}]
        assert result['subjectB'] == [{'test': 0}, {'test': 5}]

    def test_classify_subset(self) -> None:
        ccol = ClassifyDescriptorCollection({
            'subjectA': DummyClassifier(),
            'subjectB': DummyClassifier(),
        })

        classifierB = ccol._label_to_classifier['subjectB']
        classifierB.classify_one_element = mock.Mock()  # type: ignore

        d_v = [0, 1, 2, 3, 4]
        d = DescriptorMemoryElement('0')
        d.set_vector(d_v)
        result = ccol.classify(d, labels=['subjectA'])

        # Should contain one entry for each requested classifier.
        self.assertEqual(len(result), 1)
        self.assertIn('subjectA', result)
        self.assertNotIn('subjectB', result)
        classifierB.classify_one_element.assert_not_called()
        # Each key should map to a classification element (memory in this case
        # because we're using the default factory)
        self.assertIsInstance(result['subjectA'], MemoryClassificationElement)
        # We know the dummy classifier outputs "classifications" in a
        # deterministic way: class label is descriptor UUID and classification
        # value is its vector as a list.
        self.assertDictEqual(result['subjectA'].get_classification(),
                             {'test': 0})

    def test_classify_empty_subset(self) -> None:
        ccol = ClassifyDescriptorCollection({
            'subjectA': DummyClassifier(),
            'subjectB': DummyClassifier(),
        })

        classifierA = ccol._label_to_classifier['subjectA']
        classifierA.classify_one_element = mock.Mock()  # type: ignore
        classifierB = ccol._label_to_classifier['subjectB']
        classifierB.classify_one_element = mock.Mock()  # type: ignore

        d_v = [0, 1, 2, 3, 4]
        d = DescriptorMemoryElement('0')
        d.set_vector(d_v)
        result = ccol.classify(d, labels=[])

        # Should contain no entries.
        self.assertEqual(len(result), 0)
        self.assertNotIn('subjectA', result)
        classifierA.classify_one_element.assert_not_called()
        self.assertNotIn('subjectB', result)
        classifierB.classify_one_element.assert_not_called()

    def test_classify_missing_label(self) -> None:
        ccol = ClassifyDescriptorCollection({
            'subjectA': DummyClassifier(),
            'subjectB': DummyClassifier(),
        })

        d_v = [0, 1, 2, 3, 4]
        d = DescriptorMemoryElement('0')
        d.set_vector(d_v)

        # Should throw a MissingLabelError
        with self.assertRaises(MissingLabelError) as cm:
            ccol.classify(d, labels=['subjectC'])
        self.assertSetEqual(cm.exception.labels, {'subjectC'})

        # Should throw a MissingLabelError
        with self.assertRaises(MissingLabelError) as cm:
            ccol.classify(d, labels=['subjectA', 'subjectC'])
        self.assertSetEqual(cm.exception.labels, {'subjectC'})

        # Should throw a MissingLabelError
        with self.assertRaises(MissingLabelError) as cm:
            ccol.classify(d, labels=['subjectC', 'subjectD'])
        self.assertSetEqual(cm.exception.labels, {'subjectC', 'subjectD'})

        # Should throw a MissingLabelError
        with self.assertRaises(MissingLabelError) as cm:
            ccol.classify(d, labels=['subjectA', 'subjectC', 'subjectD'])
        self.assertSetEqual(cm.exception.labels, {'subjectC', 'subjectD'})
