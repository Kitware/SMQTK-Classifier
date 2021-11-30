import pickle
import multiprocessing
import multiprocessing.pool
from typing import Tuple
import unittest
import unittest.mock as mock

from smqtk_core.configuration import configuration_test_helper
from smqtk_descriptors import DescriptorElement, DescriptorElementFactory
from smqtk_descriptors.impls.descriptor_element.memory import DescriptorMemoryElement
import numpy
import pytest

from smqtk_classifier import ClassifyDescriptor
from smqtk_classifier.impls.classify_descriptor_supervised.sklearn_svm import SkLearnSvmClassifier


@pytest.mark.skipif(not SkLearnSvmClassifier.is_usable(),
                    reason="SkLearnSvmClassifier does not report as usable.")
class TestSkLearnSvmClassifier (unittest.TestCase):

    def test_impl_findable(self) -> None:
        self.assertIn(SkLearnSvmClassifier, ClassifyDescriptor.get_impls())

    @mock.patch('smqtk_classifier.impls.classify_descriptor_supervised.sklearn_svm.SkLearnSvmClassifier._reload_model')
    def test_configuration(self, m_inst_load_model: mock.Mock) -> None:
        """ Test configuration handling for this implementation. """
        ex_model_uri = 'some model uri'
        ex_c = 4.0
        ex_kernel = 'linear'
        ex_probability = True
        ex_normalize = 2

        c = SkLearnSvmClassifier(ex_model_uri,
                                 C=ex_c,
                                 kernel=ex_kernel,
                                 probability=ex_probability,
                                 normalize=ex_normalize)
        for inst in configuration_test_helper(c):  # type: SkLearnSvmClassifier
            assert inst.svm_model_uri == ex_model_uri
            assert inst.C == ex_c
            assert inst.kernel == ex_kernel
            assert inst.probability == ex_probability
            assert inst.normalize == ex_normalize

    def test_save_model(self) -> None:
        classifier = SkLearnSvmClassifier(
            normalize=None,  # DO NOT normalize descriptors
        )
        self.assertTrue(classifier.svm_model is None)
        _ = pickle.loads(pickle.dumps(classifier))

        # train arbitrary model (same as ``test_simple_classification``)
        DIM = 2
        N = 1000
        POS_LABEL = 'positive'
        NEG_LABEL = 'negative'
        d_factory = DescriptorElementFactory(DescriptorMemoryElement, {})

        def make_element(iv: Tuple[int, numpy.ndarray]) -> DescriptorElement:
            i, v = iv
            d = d_factory.new_descriptor(i)
            d.set_vector(v)
            return d

        # Constructing artificial descriptors
        x = numpy.random.rand(N, DIM)
        x_pos = x[x[:, 1] <= 0.45]
        x_neg = x[x[:, 1] >= 0.55]
        p = multiprocessing.pool.ThreadPool()
        d_pos = p.map(make_element, enumerate(x_pos))
        d_neg = p.map(make_element, enumerate(x_neg, start=N//2))
        p.close()
        p.join()

        # Training
        classifier.train({POS_LABEL: d_pos, NEG_LABEL: d_neg})

        # Test original classifier
        # - Using classification method implemented by the subclass directly
        #   in order to test simplest scope possible.
        t_v = numpy.random.rand(DIM)
        c_expected = list(classifier._classify_arrays([t_v]))[0]

        # Restored classifier should classify the same test descriptor the
        # same.
        classifier2 = pickle.loads(pickle.dumps(classifier))
        c_post_pickle = list(classifier2._classify_arrays([t_v]))[0]
        # There may be floating point error, so extract actual confidence
        # values and check post round
        c_pp_positive = c_post_pickle[POS_LABEL]
        c_pp_negative = c_post_pickle[NEG_LABEL]
        c_e_positive = c_expected[POS_LABEL]
        c_e_negative = c_expected[NEG_LABEL]
        self.assertAlmostEqual(c_e_positive, c_pp_positive, 5)
        self.assertAlmostEqual(c_e_negative, c_pp_negative, 5)

    def test_simple_classification(self) -> None:
        """
        simple SkLearnSvmClassifier test - 2-class

        Test SkLearn classification functionality using random constructed
        data, training the y=0.5 split
        """
        DIM = 2
        N = 1000
        POS_LABEL = 'positive'
        NEG_LABEL = 'negative'
        p = multiprocessing.pool.ThreadPool()
        d_factory = DescriptorElementFactory(DescriptorMemoryElement, {})

        def make_element(iv: Tuple[int, numpy.ndarray]) -> DescriptorElement:
            _i, _v = iv
            elem = d_factory.new_descriptor(_i)
            elem.set_vector(_v)
            return elem

        # Constructing artificial descriptors
        x = numpy.random.rand(N, DIM)
        x_pos = x[x[:, 1] <= 0.45]
        x_neg = x[x[:, 1] >= 0.55]

        d_pos = p.map(make_element, enumerate(x_pos))
        d_neg = p.map(make_element, enumerate(x_neg, start=N//2))

        # Create/Train test classifier
        classifier = SkLearnSvmClassifier(
            normalize=None,  # DO NOT normalize descriptors
        )
        classifier.train({POS_LABEL: d_pos, NEG_LABEL: d_neg})

        # Test classifier
        x = numpy.random.rand(N, DIM)
        x_pos = x[x[:, 1] <= 0.45]
        x_neg = x[x[:, 1] >= 0.55]

        # Test that examples expected to classify to the positive class are,
        # and same for those expected to be in the negative class.
        c_map_pos = list(classifier._classify_arrays(x_pos))
        for v, c_map in zip(x_pos, c_map_pos):
            assert c_map[POS_LABEL] > c_map[NEG_LABEL], \
                "Found False positive: {} :: {}" \
                .format(v, c_map)

        c_map_neg = list(classifier._classify_arrays(x_neg))
        for v, c_map in zip(x_neg, c_map_neg):
            assert c_map[NEG_LABEL] > c_map[POS_LABEL], \
                "Found False negative: {} :: {}" \
                .format(v, c_map)

        # Closing resources
        p.close()
        p.join()

    def test_simple_multiclass_classification(self) -> None:
        """
        simple SkLearnSvmClassifier test - 3-class

        Test SkLearnSVM classification functionality using random constructed
        data, training the y=0.33 and y=.66 split
        """
        DIM = 2
        N = 1000
        P1_LABEL = 'p1'
        P2_LABEL = 'p2'
        P3_LABEL = 'p3'
        p = multiprocessing.pool.ThreadPool()
        d_factory = DescriptorElementFactory(DescriptorMemoryElement, {})
        di = 0

        def make_element(iv: Tuple[int, numpy.ndarray]) -> DescriptorElement:
            _i, _v = iv
            elem = d_factory.new_descriptor(_i)
            elem.set_vector(_v)
            return elem

        # Constructing artificial descriptors
        x = numpy.random.rand(N, DIM)
        x_p1 = x[x[:, 1] <= 0.30]
        x_p2 = x[(x[:, 1] >= 0.36) & (x[:, 1] <= 0.63)]
        x_p3 = x[x[:, 1] >= 0.69]

        d_p1 = p.map(make_element, enumerate(x_p1, di))
        di += len(d_p1)
        d_p2 = p.map(make_element, enumerate(x_p2, di))
        di += len(d_p2)
        d_p3 = p.map(make_element, enumerate(x_p3, di))
        di += len(d_p3)

        # Create/Train test classifier
        classifier = SkLearnSvmClassifier(
            normalize=None,  # DO NOT normalize descriptors
        )
        classifier.train({P1_LABEL: d_p1, P2_LABEL: d_p2, P3_LABEL: d_p3})

        # Test classifier
        x = numpy.random.rand(N, DIM)
        x_p1 = x[x[:, 1] <= 0.30]
        x_p2 = x[(x[:, 1] >= 0.36) & (x[:, 1] <= 0.63)]
        x_p3 = x[x[:, 1] >= 0.69]

        # Test that examples expected to classify to certain classes are.
        c_map_p1 = list(classifier._classify_arrays(x_p1))
        for v, c_map in zip(x_p1, c_map_p1):
            assert c_map[P1_LABEL] > max(c_map[P2_LABEL], c_map[P3_LABEL]), \
                "Incorrect {} label: {} :: {}".format(P1_LABEL, v, c_map)

        c_map_p2 = list(classifier._classify_arrays(x_p2))
        for v, c_map in zip(x_p2, c_map_p2):
            assert c_map[P2_LABEL] > max(c_map[P1_LABEL], c_map[P3_LABEL]), \
                "Incorrect {} label: {} :: {}".format(P2_LABEL, v, c_map)

        c_map_p3 = list(classifier._classify_arrays(x_p3))
        for v, c_map in zip(x_p3, c_map_p3):
            assert c_map[P3_LABEL] > max(c_map[P1_LABEL], c_map[P2_LABEL]), \
                "Incorrect {} label: {} :: {}".format(P3_LABEL, v, c_map)

        # Closing resources
        p.close()
        p.join()
