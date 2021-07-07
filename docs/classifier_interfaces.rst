Classifier Interfaces
---------------------

Here we list and briefly describe the high level algorithm interfaces which SMQTK-Classifier provides.
Some implementations will require additional dependencies that cannot be packaged with SMQTK-Classifier.


ClassifyDescriptor
++++++++++++++++++
This interface represents algorithms that classify ``DescriptorElement`` instances into discrete labels or label confidences.

.. autoclass:: smqtk_classifier.interfaces.classify_descriptor.ClassifyDescriptor
   :members:
   :private-members:

ClassifyDescriptorSupervised
++++++++++++++++++++++++++++
This interface is a class of classifiers that are trainable via supervised training.

.. autoclass:: smqtk_classifier.interfaces.classify_descriptor_supervised.ClassifyDescriptorSupervised
   :members:

ClassifyImage
+++++++++++++
This interface represents algorithms that classify image instances into discrete labels or label confidences. The Images are formatted as
``np.ndarray``.

.. autoclass:: smqtk_classifier.interfaces.classify_image.ClassifyImage
   :members:
   :private-members:

ClassifyImageSupervised
+++++++++++++++++++++++
This interface defines a specialization of image classifiers that are
trainable via supervised learning.

.. autoclass:: smqtk_classifier.interfaces.classify_image_supervised.ClassifyImageSupervised
   :members:

Classification Element
++++++++++++++++++++++
Data structure used by Classifier

.. autoclass:: smqtk_classifier.interfaces.classification_element.ClassificationElement
   :members:
