from smqtk_classifier.classification_element_factory import ClassificationElementFactory
from smqtk_classifier.impls.classification_element.memory import MemoryClassificationElement


# Default classifier element factory for interfaces.
DFLT_CLASSIFIER_FACTORY = ClassificationElementFactory(
    MemoryClassificationElement, {}
)
