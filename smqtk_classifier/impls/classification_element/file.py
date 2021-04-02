import pickle
import os.path as osp

from smqtk_dataprovider.utils.file import safe_create_dir
from smqtk_dataprovider.utils.string import partition_string

from smqtk_classifier.exceptions import NoClassificationError
from smqtk_classifier.interfaces.classification_element import ClassificationElement


class FileClassificationElement (ClassificationElement):

    __slots__ = ('save_dir', 'pickle_protocol', 'subdir_split', 'filepath')

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, type_name, uuid, save_dir, subdir_split=None,
                 pickle_protocol=-1):
        """
        Initialize a file-base descriptor element.

        :param type_name: Type of classification. This is usually the name of
            the classifier that generated this result.
        :type type_name: str

        :param uuid: uuid for this classification
        :type uuid: collections.abc.Hashable

        :param save_dir: Directory to save this element's contents. If this path
            is relative, we interpret as relative to the current working
            directory.
        :type save_dir: str | unicode

        :param subdir_split: If a positive integer, this will cause us to store
            the vector file in a subdirectory under the ``save_dir`` that was
            specified. The integer value specifies the number of splits that we
            will make in the stringification of this descriptor's UUID. If
            there happen to be dashes in this stringification, we will
            remove them (as would happen if given an uuid.UUID instance as
            the uuid element).
        :type subdir_split: None | int

        :param pickle_protocol: Pickling protocol to use. We will use -1 by
            default (latest version, probably binary).
        :type pickle_protocol: int

        """
        super(FileClassificationElement, self).__init__(type_name, uuid)

        # TODO: Remove absolute path conversion (allow relative)
        self.save_dir = osp.abspath(osp.expanduser(save_dir))
        self.pickle_protocol = pickle_protocol

        # Saving components
        self.subdir_split = subdir_split
        if subdir_split and int(subdir_split) > 0:
            self.subdir_split = subdir_split = int(subdir_split)
            # Using all but the last split segment. This is so we don't create
            # a whole bunch of directories with a single element in them.
            save_dir = osp.join(self.save_dir,
                                *partition_string(str(uuid).replace('-', ''),
                                                  subdir_split)[:subdir_split-1]
                                )
        else:
            save_dir = self.save_dir

        self.filepath = osp.join(save_dir,
                                 "%s.%s.classification.pickle"
                                 % (self.type_name, str(self.uuid)))

    def __getstate__(self):
        return (
            super(FileClassificationElement, self).__getstate__(),
            self.save_dir,
            self.pickle_protocol,
            self.subdir_split,
            self.filepath,
        )

    def __setstate__(self, state):
        super(FileClassificationElement, self).__setstate__(state[0])
        self.save_dir, \
            self.pickle_protocol, \
            self.subdir_split, \
            self.filepath \
            = state[1:]

    def get_config(self):
        return {
            "save_dir": self.save_dir,
            'subdir_split': self.subdir_split,
            "pickle_protocol": self.pickle_protocol,
        }

    def has_classifications(self):
        return osp.isfile(self.filepath)

    def get_classification(self):
        if not self.has_classifications():
            raise NoClassificationError("No classification values.")
        with open(self.filepath, 'rb') as f:
            return pickle.load(f)

    def set_classification(self, m=None, **kwds):
        m = super(FileClassificationElement, self)\
            .set_classification(m, **kwds)
        safe_create_dir(osp.dirname(self.filepath))
        with open(self.filepath, 'wb') as f:
            pickle.dump(m, f, self.pickle_protocol)
