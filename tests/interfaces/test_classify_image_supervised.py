from typing import Any, Dict, Hashable, Iterator, List, Mapping, Sequence
import unittest
import unittest.mock as mock
import numpy as np

from smqtk_classifier import ClassifyImageSupervised
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_classifier.interfaces.classify_image import IMAGE_ITER_T
from smqtk_classifier.exceptions import ExistingModelError


class DummySupervisedClassifier (ClassifyImageSupervised):

    EXPECTED_LABELS = ['constant']
    EXPECTED_HAS_MODEL = False

    def get_config(self) -> Dict[str, Any]: ...

    def get_labels(self) -> Sequence[Hashable]: ...

    def classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]: ...

    def has_model(self) -> bool:
        return self.EXPECTED_HAS_MODEL

    def _train(
        self,
        class_examples: Mapping[Hashable, IMAGE_ITER_T]
    ) -> None: ...


class TestSupervisedClassifierAbstractClass (unittest.TestCase):

    test_classifier: DummySupervisedClassifier

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_classifier = DummySupervisedClassifier()

    def test_train_hasModel(self) -> None:
        # Calling the train method should fail the class also reports that it
        # already has a model. Shouldn't matter what is passed to the method
        # (or lack of things passed to the method).
        self.test_classifier.EXPECTED_HAS_MODEL = True
        self.assertRaises(
            ExistingModelError,
            self.test_classifier.train, {}
        )

    #
    # Testing train abstract function functionality. Method currently does not
    # care what the value for labels are.
    #

    def test_train_noModel_noExamples(self) -> None:
        self.test_classifier.EXPECTED_HAS_MODEL = False
        self.assertRaises(
            ValueError,
            self.test_classifier.train, {}
        )

    def test_train_noModel_oneExample_classExamples(self) -> None:
        self.test_classifier.EXPECTED_HAS_MODEL = False
        input_class_examples = {
            'label_1': [0, 1, 2],
        }
        self.assertRaises(
            ValueError,
            self.test_classifier.train, input_class_examples
        )

    def test_train_noModel_classExamples_only(self) -> None:
        self.test_classifier.EXPECTED_HAS_MODEL = False
        input_class_examples: Dict[Hashable, List[np.ndarray]] = {
            'label_1': [mock.Mock(spec=np.ndarray)],
            'label_2': [mock.Mock(spec=np.ndarray)],
            'label_3': [mock.Mock(spec=np.ndarray)],
            'special symbolLabel +here': [mock.Mock(spec=np.ndarray)],
        }
        # Intentionally not passing np.ndarray's here.
        self.test_classifier._train = mock.MagicMock()  # type: ignore
        self.test_classifier.train(class_examples=input_class_examples)
        self.test_classifier._train.assert_called_once_with(
            input_class_examples
        )
