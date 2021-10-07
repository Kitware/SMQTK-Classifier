import unittest

from smqtk_core.configuration import configuration_test_helper
from smqtk_descriptors.impls.descriptor_element.memory import DescriptorMemoryElement
import numpy
import pytest

from smqtk_classifier.impls.classify_descriptor_supervised.sklearn_logistic_regression import SkLearnLogisticRegression


@pytest.mark.skipif(
    not SkLearnLogisticRegression.is_usable(),
    reason="SkLearnLogisticRegression does not report as usable."
)
class TestSklearnLogisticRegressionClassifier (unittest.TestCase):
    """
    Tests for the SkLearnLogisticRegression plugin implementation.
    """

    def test_configuration(self) -> None:
        """ Standard configuration test. """
        # If there is no scikit-learn installed, this object has no constructor
        # parameterization. We would not be in here at that point because the
        # impl would not report as usable, but mypy doesn't care about that.
        inst = SkLearnLogisticRegression(  # type: ignore
            penalty='l1', dual=True, tol=1e-6, C=2.0, fit_intercept=False,
            intercept_scaling=3, class_weight={0: 2.0, 1: 3.0},
            random_state=456, solver='liblinear', max_iter=99,
            multi_class='multinomial', verbose=1, warm_start=True, n_jobs=2,
        )
        for inst_i in configuration_test_helper(inst):  # type: SkLearnLogisticRegression
            assert inst.penalty == inst_i.penalty == 'l1'
            assert inst.dual is inst_i.dual is True
            assert inst.tol == inst_i.tol == 1e-6
            assert inst.C == inst_i.C == 2.0
            assert inst.fit_intercept is inst_i.fit_intercept is False
            assert inst.intercept_scaling == inst_i.intercept_scaling == 3
            assert inst.class_weight == inst_i.class_weight == {0: 2.0, 1: 3.0}
            assert inst.random_state == inst_i.random_state == 456
            assert inst.solver == inst_i.solver == 'liblinear'
            assert inst.multi_class == inst_i.multi_class == 'multinomial'
            assert inst.verbose == inst_i.verbose == 1
            assert inst.warm_start is inst_i.warm_start is True
            assert inst.n_jobs == inst_i.n_jobs == 2

    def test_simple_classification(self) -> None:
        """ Test simple train and classify setup. """
        # Fix random seed for deterministic testing.
        numpy.random.seed(0)

        N = 1000
        POS_LABEL = 'positive'
        NEG_LABEL = 'negative'

        # Set up artificial training data set.
        # - 1 dimensional for obvious separation, this is not a performance
        #   test.
        train1 = numpy.interp(numpy.random.rand(N), [0, 1], [0.0, .45])[:, numpy.newaxis]
        train2 = numpy.interp(numpy.random.rand(N), [0, 1], [.55, 1.0])[:, numpy.newaxis]
        train1_e = [DescriptorMemoryElement(i).set_vector(v)
                    for i, v in enumerate(train1)]
        train2_e = [DescriptorMemoryElement(i).set_vector(v)
                    for i, v in enumerate(train2, start=len(train1_e))]

        # Set up artificial test set.
        test1 = numpy.interp(numpy.random.rand(N), [0, 1], [0.0, .45])[:, numpy.newaxis]
        test2 = numpy.interp(numpy.random.rand(N), [0, 1], [.55, 1.0])[:, numpy.newaxis]

        # See L22
        classifier = SkLearnLogisticRegression(random_state=0)  # type: ignore
        classifier.train({
            POS_LABEL: train1_e,
            NEG_LABEL: train2_e,
        })
        c_maps_pos = list(classifier._classify_arrays(test1))
        c_maps_neg = list(classifier._classify_arrays(test2))

        for v, m in zip(test1, c_maps_pos):
            assert m[POS_LABEL] > m[NEG_LABEL], \
                "Found false negative: {} :: {}".format(m, v)
        for v, m in zip(test2, c_maps_neg):
            assert m[NEG_LABEL] > m[POS_LABEL], \
                "Found false positive: {} :: {}".format(m, v)

    def test_simple_multiclass_classification(self) -> None:
        """ Test simple train and classify setup with 3 classes. """
        # Fix random seed for deterministic testing.
        numpy.random.seed(0)

        N = 1000
        LABEL_1 = 'p1'
        LABEL_2 = 'p2'
        LABEL_3 = 'p3'

        # Setup training dataset
        # - 1 dimensional for obvious separation, this is not a performance
        #   test.
        train1 = numpy.interp(numpy.random.rand(N), [0, 1], [0.0, .30])[:, numpy.newaxis]
        train2 = numpy.interp(numpy.random.rand(N), [0, 1], [.40, .60])[:, numpy.newaxis]
        train3 = numpy.interp(numpy.random.rand(N), [0, 1], [.70, 1.0])[:, numpy.newaxis]

        train1_e = [DescriptorMemoryElement(i).set_vector(v)
                    for i, v in enumerate(train1)]
        train2_e = [DescriptorMemoryElement(i).set_vector(v)
                    for i, v in enumerate(train2, start=len(train1_e))]
        train3_e = [DescriptorMemoryElement(i).set_vector(v)
                    for i, v
                    in enumerate(train3,
                                 start=len(train1_e) + len(train2_e))]

        # Setup testing dataset
        test1 = numpy.interp(numpy.random.rand(N), [0, 1], [0.0, .30])[:, numpy.newaxis]
        test2 = numpy.interp(numpy.random.rand(N), [0, 1], [.40, .60])[:, numpy.newaxis]
        test3 = numpy.interp(numpy.random.rand(N), [0, 1], [.70, 1.0])[:, numpy.newaxis]

        # Train and test classifier instance
        # See L22
        classifier = SkLearnLogisticRegression(random_state=0)
        classifier.train({
            LABEL_1: train1_e,
            LABEL_2: train2_e,
            LABEL_3: train3_e,
        })
        c_maps_l1 = list(classifier._classify_arrays(test1))
        c_maps_l2 = list(classifier._classify_arrays(test2))
        c_maps_l3 = list(classifier._classify_arrays(test3))

        for v, m in zip(test1, c_maps_l1):
            assert m[LABEL_1] > max(m[LABEL_2], m[LABEL_3]), \
                "Incorrect {} label: c_map={} :: test_vector={}".format(
                    LABEL_1, m, v
                )
        for v, m in zip(test2, c_maps_l2):
            assert m[LABEL_2] > max(m[LABEL_1], m[LABEL_3]), \
                "Incorrect {} label: c_map={} :: test_vector={}".format(
                    LABEL_2, m, v
                )
        for v, m in zip(test3, c_maps_l3):
            assert m[LABEL_3] > max(m[LABEL_2], m[LABEL_1]), \
                "Incorrect {} label: c_map={} :: test_vector={}".format(
                    LABEL_3, m, v
                )
