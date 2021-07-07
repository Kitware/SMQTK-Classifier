from typing import Set


class NoClassificationError (Exception):
    """
    When a ClassificationElement has no mapping yet set, but an operation
    required it.
    """


class MissingLabelError(Exception):
    """
    Raised by ClassifyDescriptorCollection.classify when requested classifier
    labels are missing from collection.
    """
    def __init__(self, labels: Set[str]):
        """
        :param labels: The labels missing from the collection
        """
        super(MissingLabelError, self).__init__(labels)
        self.labels = labels


class ExistingModelError(Exception):
    """
    Raised by ClassifyDescriptorSupervised and ClassifyImageSupervised when
    a model already exists in an instance to prevent overwriting and existing
    model
    """
