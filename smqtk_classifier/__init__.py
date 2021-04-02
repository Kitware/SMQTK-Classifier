import pkg_resources

from .interfaces.classification_element import ClassificationElement  # noqa: F401
from .interfaces.classifier import Classifier  # noqa: F401
from .interfaces.supervised import SupervisedClassifier  # noqa: F401

from .classification_element_factory import ClassificationElementFactory  # noqa: F401
from .classifier_collection import ClassifierCollection  # noqa: F401


# It is known that this will fail if this package is not "installed" in the
# current environment. Additional support is pending defined use-case-driven
# requirements.
__version__ = pkg_resources.get_distribution(__name__).version
