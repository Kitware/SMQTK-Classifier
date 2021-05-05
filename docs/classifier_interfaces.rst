Classifier Interfaces
---------------------

Here we list and briefly describe the high level algorithm interfaces which SMQTK-Classifier provides.
There is at least one implementation available for each interface.
Some implementations will require additional dependencies that cannot be packaged with SMQTK-Classifier.


Classifier
++++++++++
This interface represents algorithms that classify ``DescriptorElement`` instances into discrete labels or label confidences.

.. autoclass:: smqtk_classifier.interfaces.classifier.Classifier
   :members:
   :private-members:

Supervised Classifier
+++++++++++++++++++++
This interface is a class of classifiers that are trainable via supervised training.

.. autoclass:: smqtk_classifier.interfaces.supervised.SupervisedClassifier
   :members:

Classification Element
++++++++++++++++++++++
Data structure used by Classifier

.. autoclass:: smqtk_classifier.interfaces.classification_element.ClassificationElement
   :members:
