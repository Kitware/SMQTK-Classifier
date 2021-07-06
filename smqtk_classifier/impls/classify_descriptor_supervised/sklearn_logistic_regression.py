from typing import Any, Dict, Hashable, Iterable, Iterator, Mapping, Sequence, Union
import warnings

import numpy as np
from smqtk_descriptors import DescriptorElement

from smqtk_classifier.interfaces.classify_descriptor_supervised import ClassifyDescriptorSupervised


try:
    import sklearn  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
except ImportError:
    warnings.warn(
        "sklearn.linear_model.LogisticRegression was not importable: the "
        "SkLearnLogisticRegression supervised classifier will not be usable."
    )
    sklearn = None

    # Actually no, mypy, if we're here then it is not actually defined.
    class LogisticRegression:  # type: ignore
        """ Stub """


class SkLearnLogisticRegression (LogisticRegression, ClassifyDescriptorSupervised):
    """
    Classifier implementation using Scikit Learn's LogisticRegression
    classifier.

    See ``sklearn.linear_model.LogisticRegression`` documentation for more
    details.
    """

    @classmethod
    def is_usable(cls) -> bool:
        return sklearn is not None

    def get_config(self) -> Dict[str, Any]:
        return self.get_params()

    def has_model(self) -> bool:
        try:
            return self.coef_ is not None
        except AttributeError:
            return False

    def get_labels(self) -> Sequence[Hashable]:
        return self.classes_.tolist()

    def _train(
        self,
        class_examples: Mapping[Any, Iterable[DescriptorElement]],
        **extra_params: Any
    ) -> None:
        # convert descriptor elements into combines ndarray with associated
        # label vector.
        vec_list = []
        label_list = []
        for label, examples in class_examples.items():
            label_vectors = \
                DescriptorElement.get_many_vectors(examples)
            # ``is`` or ``count`` method messes up when elements are np arrays.
            none_count = len([e for e in label_vectors if e is None])
            assert none_count == 0, \
                "Some descriptor elements for label {} did not contain " \
                "vectors! (n={})".format(label, none_count)
            vec_list.extend(label_vectors)
            label_list.extend([label] * len(label_vectors))
        vec_list = np.vstack(vec_list)
        self.fit(vec_list, label_list)

    def _classify_arrays(self, array_iter: Union[np.ndarray, Iterable[np.ndarray]]) -> Iterator[Dict[Hashable, float]]:
        # Collect arrays for prediction.
        # - Collect into numpy.ndarray if not already one.
        if isinstance(array_iter, np.ndarray):
            mat = array_iter
        else:
            # Expand out input iterable of arrays into a large matrix for
            # prediction.
            mat = np.array(list(array_iter))
        proba_mat = self.predict_proba(mat)
        class_list = self.classes_.tolist()
        for proba in proba_mat:
            yield dict(zip(class_list, proba))
