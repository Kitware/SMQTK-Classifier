import collections
import logging
import pickle
from typing import Any, Dict, Hashable, Iterable, Iterator, Mapping, Optional, Sequence, Union
import warnings

import numpy as np

from smqtk_dataprovider import from_uri
from smqtk_descriptors import DescriptorElement

from smqtk_classifier.interfaces.classify_descriptor_supervised import ClassifyDescriptorSupervised

LOG = logging.getLogger(__name__)

try:
    # noinspection PyPackageRequirements
    import scipy.stats  # type: ignore
except ImportError:
    warnings.warn(
        "scipy.stats not importable: SkLearnSvmClassifier will not be usable."
    )
    scipy = None

try:
    from sklearn import svm
except ImportError:
    warnings.warn(
        "svm not importable: SkLearnSvmClassifier will not be usable."
    )
    svm = None


class SkLearnSvmClassifier (ClassifyDescriptorSupervised):
    """
    Classifier that wraps the SkLearn SVM (Support Vector Machine)
    SVC (C-Support Vector Classification) module.

    Model file paths are optional. If they are given and the file(s) exist,
    we will load them. If they do not, we treat the path(s) as the output
    path(s) for saving a model after calling ``train``. If this is None
    (default), no model is loaded nor output via training, thus any model
    trained will only exist in memory during the lifetime of this instance.

    :param svm_model_uri: Path to the model file.
    :param C: Regularization parameter passed to SkLearn SVM SVC model.
    :param kernel: Kernel type passed to SkLearn SVM SVC model.
    :param probability: Whether to enable probability estimates or not.
    :param calculate_class_weights: Whether to manually calculate the
        class weights to be passed to the SVM model or not.
        Defaults to true. If false, all classes will be given equal weight.
    :param normalize: Normalize input vectors to training and
        classification methods using ``numpy.linalg.norm``. This may either
        be  ``None``, disabling normalization, or any valid value that
        could be passed to the ``ord`` parameter in ``numpy.linalg.norm``
        for 1D arrays. This is ``None`` by default (no normalization).
    """

    # noinspection PyDefaultArgument
    def __init__(
        self,
        svm_model_uri: Optional[str] = None,
        C: float = 2.0,  # Regularization parameter
        kernel: str = 'linear',  # Kernel type
        probability: bool = True,  # Enable probabilty estimates
        calculate_class_weights: bool = True,  # Enable calculation of class weights
        normalize: Optional[Union[int, float, str]] = None,
    ):
        super(SkLearnSvmClassifier, self).__init__()

        self.svm_model_uri = svm_model_uri

        # Elements will be None if input URI is None
        #: :type: None | smqtk.representation.DataElement
        self.svm_model_elem = \
            svm_model_uri and from_uri(svm_model_uri)

        self.C = C
        self.kernel = kernel
        self.probability = probability
        self.calculate_class_weights = calculate_class_weights
        self.normalize = normalize

        # Validate normalization parameter by trying it on a random vector
        if normalize is not None:
            self._norm_vector(np.random.rand(8))

        # generated parameters
        self.svm_model: Optional[svm.SVC] = None

        self._reload_model()

    @classmethod
    def is_usable(cls) -> bool:
        return None not in {scipy, svm}

    def get_config(self) -> Dict[str, Any]:
        return {
            "svm_model_uri": self.svm_model_uri,
            "C": self.C,
            "kernel": self.kernel,
            "probability": self.probability,
            "calculate_class_weights": self.calculate_class_weights,
            "normalize": self.normalize,
        }

    def _reload_model(self) -> None:
        """
        Reload SVM model from configured file path.
        """
        if self.svm_model_elem and not self.svm_model_elem.is_empty():
            svm_model_tmp_fp = self.svm_model_elem.write_temp()
            with open(svm_model_tmp_fp, 'rb') as f:
                self.svm_model = pickle.load(f)
            self.svm_model_elem.clean_temp()

    def _norm_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Class standard array normalization. Normalized along max dimension (a=0
        for a 1D array, a=1 for a 2D array, etc.).

        :param v: Vector to normalize

        :return: Returns the normalized version of input array ``v``.
        """
        if self.normalize is not None:
            n = np.linalg.norm(v, self.normalize, v.ndim - 1,
                               keepdims=True)
            # replace 0's with 1's, preventing div-by-zero
            n[n == 0.] = 1.
            return v / n

        # Normalization off
        return v

    def has_model(self) -> bool:
        """
        :return: If this instance currently has a model loaded. If no model is
            present, classification of descriptors cannot happen.
        :rtype: bool
        """
        return self.svm_model is not None

    def _train(
        self,
        class_examples: Mapping[Hashable, Iterable[DescriptorElement]]
    ) -> None:
        train_labels = []
        train_vectors = []
        train_group_sizes: Dict = {}  # number of examples per class
        # Making SVM label assignment deterministic to lexicographical order
        # of the type repr.
        # -- Can't specifically guarantee that dict key types will all support
        #    less-than operator, however we can always get some kind of repr
        #    which is a string which does support less-than. In the common case
        #    keys will be strings and ints, but this "should" handle more
        #    exotic cases, at least for the purpose of ordering keys reasonably
        #    deterministically.
        for i, l in enumerate(sorted(class_examples, key=lambda e: str(e))):
            # requires a sequence, so making the iterable ``g`` a tuple
            g = class_examples[l]
            if not isinstance(g, collections.abc.Sequence):
                LOG.debug('   (expanding iterable into sequence)')
                g = tuple(g)

            train_group_sizes[l] = float(len(g))
            x = np.array(DescriptorElement.get_many_vectors(g))
            x = self._norm_vector(x)

            train_labels.extend([l] * x.shape[0])
            train_vectors.extend(x)
            del g, x

        assert len(train_labels) == len(train_vectors), \
            "Count mismatch between parallel labels and descriptor vectors" \
            "(%d != %d)" \
            % (len(train_labels), len(train_vectors))

        # Calculate class weights
        weights = None
        if self.calculate_class_weights:
            weights = {}
            # (john.moeller): The weighting should probably be the geometric
            # mean of the number of examples over the classes divided by the
            # number of examples for the current class.
            gmean = scipy.stats.gmean(list(train_group_sizes.values()))

            for i, g in enumerate(train_group_sizes):
                w = gmean / train_group_sizes[g]
                weights[g] = w

        self.svm_model = svm.SVC(C=self.C,
                                 kernel=self.kernel,
                                 probability=self.probability,
                                 class_weight=weights)

        LOG.debug("Training SVM model")
        self.svm_model.fit(train_vectors, train_labels)

        if self.svm_model_elem and self.svm_model_elem.writable():
            LOG.debug("Saving model to element (%s)", self.svm_model_elem)
            self.svm_model_elem.set_bytes(pickle.dumps(self.svm_model))

    def get_labels(self) -> Sequence[Hashable]:
        if self.svm_model is not None:
            return list(self.svm_model.classes_)
        else:
            raise RuntimeError("No model loaded")

    def _classify_arrays(self, array_iter: Union[np.ndarray, Iterable[np.ndarray]]) -> Iterator[Dict[Hashable, float]]:
        if self.svm_model is None:
            raise RuntimeError("No SVM model present for classification")

        # Dump descriptors into a matrix for normalization and use in
        # prediction.
        vec_mat = np.array(list(array_iter))
        vec_mat = self._norm_vector(vec_mat)

        svm_model_labels = self.get_labels()

        if self.svm_model.probability:
            proba_mat = self.svm_model.predict_proba(vec_mat)
            for proba in proba_mat:
                yield dict(zip(svm_model_labels, proba))
        else:
            c_base = {label: 0.0 for label in svm_model_labels}

            proba_mat = self.svm_model.predict(vec_mat)
            for p in proba_mat:
                c = dict(c_base)
                c[p] = 1.0
                yield c
