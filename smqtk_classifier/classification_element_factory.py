from typing import Any, Dict, Hashable, Type, TypeVar

from smqtk_core import Configurable
from smqtk_core.configuration import (
    cls_conf_from_config_dict,
    cls_conf_to_config_dict,
    make_default_config
)
from smqtk_core.dict import merge_dict

from smqtk_classifier.interfaces.classification_element import ClassificationElement


C = TypeVar("C", bound=ClassificationElement)


class ClassificationElementFactory (Configurable):
    """
    Factory class for producing ClassificationElement instances of a specified
    type and configuration.
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, type: Type[C], type_config: Dict[str, Any]):
        """
        Initialize the factory to produce ClassificationElement instances of the
        given type from the given configuration.

        :param type: Python implementation type of the ClassifierElement to
            produce
        :param type_config: Configuration dictionary that will be passed
            ``from_config`` class method of given ``type``.
        """
        self.type = type
        self.type_config = type_config

    #
    # Class methods
    #

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        return make_default_config(ClassificationElement.get_impls())

    # We likely do not expect subclasses for this type base, thus it is OK to
    # use direct type reference in Type and return annotations.
    @classmethod
    def from_config(
        cls: Type["ClassificationElementFactory"],
        config_dict: Dict,
        merge_default: bool = True
    ) -> "ClassificationElementFactory":
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless and instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.

        :return: Constructed instance from the provided config.
        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        ce_type, ce_conf = cls_conf_from_config_dict(
            config_dict,  ClassificationElement.get_impls()
        )
        return ClassificationElementFactory(ce_type, ce_conf)

    def get_config(self) -> Dict[str, Any]:
        return cls_conf_to_config_dict(self.type, self.type_config)

    # noinspection PyShadowingBuiltins
    def new_classification(self, type: str, uuid: Hashable) -> ClassificationElement:
        """
        Create a new ClassificationElement instance of the configured
        implementation.

        :param type: Type of classifier. This is usually the name of the
            classifier that generated this result.
        :param uuid: UUID to associate with the classification.

        :return: New ClassificationElement instance.
        """
        return self.type.from_config(self.type_config, type, uuid)

    # noinspection PyShadowingBuiltins
    def __call__(self, type: str, uuid: Hashable) -> ClassificationElement:
        """
        Create a new ClassificationElement instance of the configured
        implementation.

        :param type: Type of classifier. This is usually the name of the
            classifier that generated this result.
        :param uuid: UUID to associate with the classification.

        :return: New ClassificationElement instance.
        """
        return self.new_classification(type, uuid)
