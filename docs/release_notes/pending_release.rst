Pending Release Notes
=====================


Updates / New Features
----------------------

Interfaces

* Classifier and SupervisedClassifier split into ClassifyImage,
  ClassifyImageSupervised, ClassifyDescriptor, and ClassifyDescriptorSupervised

* Rather than just taking in Descriptors, the SMQTK-Classifier can work
  directly with both Descriptors and Images

* Standardized Image input to follow the format of numpy matrices

Implementations

* Modified `smqtk_classifier.impls.classification_element.postgres` to use the
  helper from `smqtk_dataprovider.utils.postgres`.

Fixes
-----
