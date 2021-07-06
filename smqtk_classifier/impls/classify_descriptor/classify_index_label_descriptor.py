from typing import Any, Dict, Hashable, Sequence, Iterable, Iterator, Union
from warnings import warn
import numpy as np

from smqtk_dataprovider import from_uri

from smqtk_classifier.interfaces.classify_descriptor import ClassifyDescriptor


class ClassifyIndexLabelDescriptor(ClassifyDescriptor):
    """
    Applies a listing of labels (new-line separated) to input "descriptor"
    values, which is actually a vector of class confidence values.

    We expect to be given a URI to a new-line separated text file where each
    line is a separate label in order and matching the dimensionality of an
    input descriptor.

    :param index_to_label_uri: URI to new-line separated sequence of labels.
    """

    def __init__(self, index_to_label_uri: str):
        super().__init__()

        # load label vector
        self.index_to_label_uri = index_to_label_uri
        self.label_vector = [line.strip() for line in
                             from_uri(index_to_label_uri).to_buffered_reader()]

    @classmethod
    def is_usable(cls) -> bool:
        return True

    def get_config(self) -> Dict[str, Any]:
        return {
            "index_to_label_uri": self.index_to_label_uri,
        }

    def get_labels(self) -> Sequence[Hashable]:
        # copying container
        return list(self.label_vector)

    def _classify_arrays(self, array_iter: Union[np.ndarray, Iterable[np.ndarray]]) -> Iterator[Dict[Hashable, float]]:
        check_dim = True
        for d_vector in array_iter:
            if check_dim:
                if len(self.label_vector) != len(d_vector):
                    raise RuntimeError(
                        "Failed to apply label vector to input descriptor of "
                        "incongruous dimensionality ({} labels != {} vector "
                        "shape)".format(len(self.label_vector), d_vector.shape)
                    )
                check_dim = False
            yield dict(zip(self.label_vector, d_vector))


class IndexLabelClassifier(ClassifyIndexLabelDescriptor):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warn("IndexLabelClassifier was renamed to "
             "ClassifyIndexLabelDescriptor", category=DeprecationWarning,
             stacklevel=2)
        super().__init__(*args, **kwargs)
