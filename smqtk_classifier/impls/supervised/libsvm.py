import collections.abc
from copy import deepcopy
import ctypes
import logging
import os
import pickle
import tempfile
from typing import Any, Dict, Hashable, Iterable, Iterator, Mapping, Optional, Sequence, Union
import warnings

import numpy
import numpy as np
import numpy.linalg

from smqtk_dataprovider import from_uri
from smqtk_descriptors import DescriptorElement
from smqtk_descriptors.utils import parallel_map

from smqtk_classifier.interfaces.classify_descriptor import CLASSIFICATION_DICT_T
from smqtk_classifier.interfaces.classify_descriptor_supervised import ClassifyDescriptorSupervised


LOG = logging.getLogger(__name__)
SVM_PARAM_MAPPING_T = Mapping[str, Union[int, float, str]]


try:
    # noinspection PyPackageRequirements
    import scipy.stats  # type: ignore
except ImportError:
    warnings.warn(
        "scipy.stats not importable: LibSvmClassifier will not be usable."
    )
    scipy = None

try:
    # noinspection PyPackageRequirements
    import svm  # type: ignore
    # noinspection PyPackageRequirements
    import svmutil  # type: ignore
except ImportError:
    warnings.warn(
        "svm/svmutil not importable: LibSvmClassifier will not be usable."
    )
    svm = None
    svmutil = None


class LibSvmClassifier (ClassifyDescriptorSupervised):
    """
    Classifier that uses libSVM for support-vector machine functionality.

    Model file paths are optional. If they are given and the file(s) exist,
    we will load them. If they do not, we treat the path(s) as the output
    path(s) for saving a model after calling ``train``. If this is None
    (default), no model is loaded nor output via training, thus any model
    trained will only exist in memory during the lifetime of this instance.

    **Note**: *If pickled without a having model file paths configured, this
    implementation will write out temporary files upon pickling and loading.
    This is required because the model instance is not transportable via
    serialization due to libSVM being an external C library.*

    :param svm_model_uri: Path to the libSVM model file.
    :param svm_label_map_uri: Path to the pickle file containing this
        model's output labels.
    :param train_params: SVM parameters used for training. See libSVM
        documentation for parameter flags and values. A parameter with no value
        on the SVM CLI should be given an empty string value,
        E.g. `{"-q": ""}`.
    :param normalize: Normalize input vectors to training and
        classification methods using ``numpy.linalg.norm``. This may either
        be  ``None``, disabling normalization, or any valid value that
        could be passed to the ``ord`` parameter in ``numpy.linalg.norm``
        for 1D arrays. This is ``None`` by default (no normalization).
    :param n_jobs:
        Number of processes to use to parallelize prediction. If None or a
        negative value, all cores are used.
    """

    # noinspection PyDefaultArgument
    def __init__(
        self,
        svm_model_uri: Optional[str] = None,
        svm_label_map_uri: Optional[str] = None,
        train_params: SVM_PARAM_MAPPING_T = {
            '-s': 0,  # C-SVC, assumed default if not provided
            '-t': 0,  # linear kernel
            '-b': 1,  # enable probability estimates
            '-c': 2,  # SVM parameter C
            # '-g': 0.0078125,  # initial gamma (1 / 128)
        },
        normalize: Optional[Union[int, float, str]] = None,
        n_jobs: Optional[int] = 4,
    ):
        super(LibSvmClassifier, self).__init__()

        self.svm_model_uri = svm_model_uri
        self.svm_label_map_uri = svm_label_map_uri

        # Elements will be None if input URI is None
        #: :type: None | smqtk.representation.DataElement
        self.svm_model_elem = \
            svm_model_uri and from_uri(svm_model_uri)
        #: :type: None | smqtk.representation.DataElement
        self.svm_label_map_elem = \
            svm_label_map_uri and from_uri(svm_label_map_uri)

        # Shallow copy to shield from modifying input.
        self.train_params = dict(train_params)
        self.normalize = normalize
        self.n_jobs = n_jobs
        # Validate normalization parameter by trying it on a random vector
        if normalize is not None:
            self._norm_vector(numpy.random.rand(8))

        # generated parameters
        self.svm_model: Optional[svm.svm_model] = None
        # dictionary mapping SVM integer labels to semantic labels
        self.svm_label_map: Dict[int, Hashable] = {}

        self._reload_model()

    @classmethod
    def is_usable(cls) -> bool:
        return None not in {scipy, svm, svmutil}

    def get_config(self) -> Dict[str, Any]:
        return {
            "svm_model_uri": self.svm_model_uri,
            "svm_label_map_uri": self.svm_label_map_uri,
            "train_params": self.train_params,
            "normalize": self.normalize,
            "n_jobs": self.n_jobs,
        }

    def __getstate__(self) -> Any:
        # If we don't have a model, or if we have one but its not being saved
        # to files.
        if not self.has_model() or (self.svm_model_uri is not None and
                                    self.svm_label_map_uri is not None):
            return self.get_config()
        else:
            LOG.debug("Saving model to temp file for pickling")
            fd, fp = tempfile.mkstemp()
            try:
                os.close(fd)

                state = self.get_config()
                state['__LOCAL__'] = True
                state['__LOCAL_LABELS__'] = self.svm_label_map

                fp_bytes = fp.encode('utf8')
                svmutil.svm_save_model(fp_bytes, self.svm_model)
                with open(fp, 'rb') as model_f:
                    state['__LOCAL_MODEL__'] = model_f.read()

                return state
            finally:
                os.remove(fp)

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state)

        self.svm_model_elem = \
            self.svm_model_uri and from_uri(self.svm_model_uri)
        self.svm_label_map_elem = \
            self.svm_label_map_uri and from_uri(self.svm_label_map_uri)

        # C libraries/pointers don't survive across processes.
        if '__LOCAL__' in state:
            # These would have gotten copied into dict during the updated.
            # The instance doesn't need to keep them around after this.
            del self.__dict__['__LOCAL__']
            del self.__dict__['__LOCAL_LABELS__']
            del self.__dict__['__LOCAL_MODEL__']

            fd, fp = tempfile.mkstemp()
            try:
                os.close(fd)

                self.svm_label_map = state['__LOCAL_LABELS__']

                # write model to file, then load via libSVM
                with open(fp, 'wb') as model_f:
                    model_f.write(state['__LOCAL_MODEL__'])

                fp_bytes = fp.encode('utf8')
                self.svm_model = svmutil.svm_load_model(fp_bytes)

            finally:
                os.remove(fp)
        else:
            self.svm_model = None
            self._reload_model()

    def _reload_model(self) -> None:
        """
        Reload SVM model from configured file path.
        """
        if self.svm_model_elem and not self.svm_model_elem.is_empty():
            svm_model_tmp_fp = self.svm_model_elem.write_temp()
            self.svm_model = svmutil.svm_load_model(svm_model_tmp_fp)
            self.svm_model_elem.clean_temp()

        if self.svm_label_map_elem and not self.svm_label_map_elem.is_empty():
            self.svm_label_map = \
                pickle.loads(self.svm_label_map_elem.get_bytes())

    @staticmethod
    def _gen_param_string(params: SVM_PARAM_MAPPING_T) -> str:
        """
        Make a single string out of a parameters dictionary
        """
        return ' '.join((str(k) + ' ' + str(v)
                         for k, v in params.items()))

    def _norm_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Class standard array normalization. Normalized along max dimension (a=0
        for a 1D array, a=1 for a 2D array, etc.).

        :param v: Vector to normalize

        :return: Returns the normalized version of input array ``v``.
        """
        if self.normalize is not None:
            n = numpy.linalg.norm(v, self.normalize, v.ndim - 1,
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
        return None not in (self.svm_model, self.svm_label_map)

    def _train(
        self,
        class_examples: Mapping[Any, Iterable[DescriptorElement]],
        **extra_params: Any
    ) -> None:
        # Offset from 0 for positive class labels to use
        # - not using label of 0 because we think libSVM wants positive labels
        CLASS_LABEL_OFFSET = 1

        # Stuff for debug reporting
        param_debug = {'-q': ''}
        if LOG.getEffectiveLevel() <= logging.DEBUG:
            param_debug = {}

        # Form libSVM problem input values
        LOG.debug("Formatting problem input")
        train_labels = []
        train_vectors = []
        train_group_sizes = []  # number of examples per class
        self.svm_label_map = {}
        # Making SVM label assignment deterministic to lexicographical order
        # of the type repr.
        # -- Can't specifically guarantee that dict key types will all support
        #    less-than operator., however we can always get some kind of repr
        #    which is a string which does support less-than. In the common case
        #    keys will be strings and ints, but this "should" handle more
        #    exotic cases, at least for the purpose of ordering keys reasonably
        #    deterministically.
        for i, l in enumerate(sorted(class_examples, key=lambda e: str(e)), CLASS_LABEL_OFFSET):
            # Map integer SVM label to semantic label
            self.svm_label_map[i] = l

            LOG.debug('-- class %d (%s)', i, l)
            # requires a sequence, so making the iterable ``g`` a tuple
            g = class_examples[l]
            if not isinstance(g, collections.abc.Sequence):
                LOG.debug('   (expanding iterable into sequence)')
                g = tuple(g)

            train_group_sizes.append(float(len(g)))
            x = numpy.array(DescriptorElement.get_many_vectors(g))
            x = self._norm_vector(x)
            train_labels.extend([i] * x.shape[0])
            train_vectors.extend(x.tolist())
            del g, x

        assert len(train_labels) == len(train_vectors), \
            "Count mismatch between parallel labels and descriptor vectors" \
            "being sent to libSVM (%d != %d)" \
            % (len(train_labels), len(train_vectors))

        LOG.debug("Forming train params")
        params = deepcopy(self.train_params)
        params.update(param_debug)
        # Calculating class weights if set to C-SVC type SVM
        if '-s' not in params or int(params['-s']) == 0:
            # (john.moeller): The weighting should probably be the geometric
            # mean of the number of examples over the classes divided by the
            # number of examples for the current class.
            gmean = scipy.stats.gmean(train_group_sizes)
            for i, n in enumerate(train_group_sizes, CLASS_LABEL_OFFSET):
                w = gmean / n
                params['-w' + str(i)] = w
                LOG.debug("-- class '%s' weight: %s", self.svm_label_map[i], w)

        LOG.debug("Making parameters obj")
        svm_params = svmutil.svm_parameter(self._gen_param_string(params))
        LOG.debug("Creating SVM problem")
        svm_problem = svm.svm_problem(train_labels, train_vectors)
        del train_vectors
        LOG.debug("Training SVM model")
        self.svm_model = svmutil.svm_train(svm_problem, svm_params)
        LOG.debug("Training SVM model -- Done")

        if self.svm_label_map_elem and self.svm_label_map_elem.writable():
            LOG.debug("saving labels to element (%s)", self.svm_label_map_elem)
            self.svm_label_map_elem.set_bytes(
                pickle.dumps(self.svm_label_map, -1)
            )
        if self.svm_model_elem and self.svm_model_elem.writable():
            LOG.debug("saving model to element (%s)", self.svm_model_elem)
            # LibSvm I/O only works with filepaths, thus the need for an
            # intermediate temporary file.
            fd, fp = tempfile.mkstemp()
            try:
                svmutil.svm_save_model(fp, self.svm_model)
                # Use the file descriptor to create the file object.
                # This avoids reopening the file and will automatically
                # close the file descriptor on exiting the with block.
                # fdopen() is required because in Python 2 open() does
                # not accept a file descriptor.
                with os.fdopen(fd, 'rb') as f:
                    self.svm_model_elem.set_bytes(f.read())
            finally:
                os.remove(fp)

    def get_labels(self) -> Sequence[Hashable]:
        if not self.has_model():
            raise RuntimeError("No model loaded")
        return list(self.svm_label_map.values())

    def _classify_arrays(self, array_iter: Union[np.ndarray, Iterable[np.ndarray]]) -> Iterator[Dict[Hashable, float]]:
        if not self.has_model():
            raise RuntimeError("No SVM model present for classification")
        assert self.svm_model is not None, (
            "Should have an SVM model at this point."
        )

        # Dump descriptors into a matrix for normalization and use in
        # prediction.
        vec_mat = numpy.array(list(array_iter))
        vec_mat = self._norm_vector(vec_mat)
        n_jobs = self.n_jobs
        if n_jobs is not None:
            n_jobs = min(len(vec_mat), n_jobs)
        # Else: `n_jobs` is `None`, which is OK as it's the default  value for
        # parallel_map.

        svm_label_map = self.svm_label_map
        c_base = dict((la, 0.) for la in svm_label_map.values())

        # Effectively reproducing the body of svmutil.svm_predict in order to
        # simplify and get around excessive prints
        svm_type = self.svm_model.get_svm_type()
        nr_class = self.svm_model.get_nr_class()
        # Model internal labels. Parallel to ``prob_estimates`` array.
        svm_model_labels = self.svm_model.get_labels()

        # TODO: Normalize input arrays in batch(es). TEST if current norm
        #       function can just take a matrix?

        if self.svm_model.is_probability_model():
            # noinspection PyUnresolvedReferences
            if svm_type in [svm.NU_SVR, svm.EPSILON_SVR]:
                nr_class = 0

            def single_pred(v: np.ndarray) -> CLASSIFICATION_DICT_T:
                prob_estimates = (ctypes.c_double * nr_class)()
                v, idx = svm.gen_svm_nodearray(v.tolist())
                svm.libsvm.svm_predict_probability(self.svm_model, v,
                                                   prob_estimates)
                c = dict(c_base)  # Shallow copy
                c.update({svm_label_map[label]: prob for label, prob
                          in zip(svm_model_labels, prob_estimates[:nr_class])})
                return c
            # If n_jobs == 1, just be serial
            if n_jobs == 1:
                return (single_pred(v) for v in vec_mat)
            else:
                return parallel_map(single_pred, vec_mat,
                                    cores=n_jobs,
                                    use_multiprocessing=True)

        else:
            # noinspection PyUnresolvedReferences
            if svm_type in (svm.ONE_CLASS, svm.EPSILON_SVR, svm.NU_SVC):
                nr_classifier = 1
            else:
                nr_classifier = nr_class * (nr_class - 1) // 2

            def single_label(v: np.ndarray) -> CLASSIFICATION_DICT_T:
                dec_values = (ctypes.c_double * nr_classifier)()
                v, idx = svm.gen_svm_nodearray(v.tolist())
                label = svm.libsvm.svm_predict_values(self.svm_model, v,
                                                      dec_values)
                c = dict(c_base)  # Shallow copy
                c[svm_label_map[label]] = 1.
                return c
            # If n_jobs == 1, just be serial
            if n_jobs == 1:
                return (single_label(v) for v in vec_mat)
            else:
                return parallel_map(single_label, vec_mat,
                                    cores=n_jobs,
                                    use_multiprocessing=True)
