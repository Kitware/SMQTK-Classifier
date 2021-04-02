
class NoClassificationError (Exception):
    """
    When a ClassificationElement has no mapping yet set, but an operation
    required it.
    """


class MissingLabelError(Exception):
    """
    Raised by ClassifierCollection.classify when requested classifier labels
    are missing from collection.
    """
    def __init__(self, labels):
        """
        :param labels: The labels missing from the collection
        :type labels: set[str]
        """
        super(MissingLabelError, self).__init__(labels)
        self.labels = labels
