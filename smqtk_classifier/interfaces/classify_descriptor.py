import abc
from collections import deque
import itertools
import logging
from typing import Deque, Generator, Hashable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from smqtk_core import Plugfigurable
from smqtk_descriptors import DescriptorElement

from smqtk_classifier._defaults import DFLT_CLASSIFIER_FACTORY
from smqtk_classifier.classification_element_factory import ClassificationElementFactory
from smqtk_classifier.interfaces.classification_element import (
    ClassificationElement,
    CLASSIFICATION_DICT_T
)


ARRAY_ITER_T = Union[np.ndarray, Iterable[np.ndarray]]
LOG = logging.getLogger(__name__)


class ClassifyDescriptor (Plugfigurable):
    """
    Interface for algorithms that classify input descriptors into discrete
    labels and/or label confidences.
    """

    @abc.abstractmethod
    def get_labels(self) -> Sequence[Hashable]:
        """
        Get the sequence of class labels that this classifier can classify
        descriptors into. This includes the negative or background label if the
        classifier embodies such a concept.

        :return: Sequence of possible classifier labels.

        :raises RuntimeError: No model loaded.

        """

    @abc.abstractmethod
    def _classify_arrays(self, array_iter: ARRAY_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
        """
        Overridable method for classifying an iterable of descriptor elements
        whose vectors should be classified.

        At this level, all input arrays are guaranteed to be of consistent
        dimensionality.

        *Remember:* A single-pass `Iterator` is a valid `Iterable`. If an
        implementation needs to pass over the input multiple times, either
        ensure you are receiving an ndarray, or *not* and Iterator.

        Each classification mapping should contain confidence values for each
        label the configured model contains.
        Implementations may act in a discrete manner whereby only one label is
        marked with a ``1`` value (others being ``0``), or in a continuous
        manner whereby each label is given a confidence-like value in the
        [0, 1] range.

        :param array_iter:
            Iterable of arrays to be classified.

        :return: Iterator of dictionaries, parallel in association to the input
            descriptor vectors. Each dictionary should map labels to associated
            confidence values.
        """

    @staticmethod
    def _assert_array_dim_consistency(array_iter: ARRAY_ITER_T) -> ARRAY_ITER_T:
        """
        Assert that arrays are consistent in dimensionality across iterated
        arrays.

        Currently we only support iterating single dimension vectors. Arrays
        of more than one dimension (i.e. 2D matrices, etc.) will trigger a
        ValueError.

        Includes a short-cut where if the input is a non-object 2D ndarray,
        dimensionality must already be consistent, so the ndarray (which is an
        Iterable) is just returned. Otherwise, we return a generator that
        checked dimensionality of the input iterable during iteration.

        :param collections.abc.Iterable[numpy.ndarray] | np.ndarray array_iter:
            Iterable numpy arrays.

        :raises AttributeError: Individual arrays are not numpy.ndarray-like.
        :raises ValueError: Not all input arrays were of consistent
            dimensionality.

        :return: Iterable of the same arrays in the same order, but validated
            to be of common dimensionality.
        """
        try:
            # If an ndarray is at least 2 dimensional and not made of just
            # objects then its shape *will* be consistent.
            # * Type-ignoring this line to utilize duck-type detection of
            #   array-type via catching attribute error.
            if array_iter.ndim > 1 and array_iter.dtype != np.object:  # type: ignore
                return array_iter
        except AttributeError:
            # If we don't encounter numpy array-like syntax proceed with
            # generic iterable fallback.
            pass

        # Fall-back: manually checking that iterated arrays are 1D and of
        # consistent size.
        def _inner() -> Generator[np.ndarray, None, None]:
            dim = None
            for a in array_iter:
                if a.ndim > 1:
                    raise ValueError("Input vector had more than one "
                                     f"dimension! (ndim = {a.ndim})")
                elif dim is None:
                    dim = a.size
                elif a.size != dim:
                    raise ValueError("Input vector violated dimension "
                                     f"consistency (basis == {dim}, "
                                     f"violation == {a.size})")
                yield a
        return _inner()

    def classify_arrays(self, array_iter: ARRAY_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
        """
        Classify an input iterable of numpy arrays into a parallel iterable of
        label-to-confidence mappings (dictionaries).

        Each classification mapping should contain confidence values for each
        label the configured model contains.
        Implementations may act in a discrete manner whereby only one label is
        marked with a ``1`` value (others being ``0``), or in a continuous
        manner whereby each label is given a confidence-like value in the
        [0, 1] range.

        :param collections.abc.Iterable[numpy.ndarray] | np.ndarray array_iter:
            Iterable of descriptor vectors, as numpy arrays, to be classified.

        :raises ValueError: Input arrays were not all of consistent
            dimensionality.

        :return: Iterable of dictionaries, parallel in association to the input
            descriptor vectors. Each dictionary should map labels to associated
            confidence values.
        """
        return self._classify_arrays(
            self._assert_array_dim_consistency(array_iter)
        )

    def classify_elements(
        self,
        descr_iter: Iterable[DescriptorElement],
        factory: ClassificationElementFactory = DFLT_CLASSIFIER_FACTORY,
        overwrite: bool = False,
        d_elem_batch: int = 100
    ) -> Iterator[ClassificationElement]:
        """
        Classify an input iterable of descriptor elements into a parallel
        iterable of classification elements.

        Classification element UIDs are inherited from the descriptor element
        it was generated from.

        We invoke ``classify_arrays`` for actual generation of classification
        results. See documentation for this method for further details.
        # We invoke ``classify_arrays`` for factory-generated classification
        # elements that do not yet have classifications stored, or on all input
        # descriptor elements if the ``overwrite`` flag is True.

        **Selective Iteration**
        For situations when it is desired to access specific generator returns,
        like when only one descriptor element is provided in order to get a
        single element out, it is strongly recommended to expand the returned
        generator into a sequence type first.
        For example, expanding out the generator's returns into a list
        (``list(g.generate_elements([e]))[0]``) is recommended over just
        getting the "next" element of the returned generator
        (``next(g.generate_elements([e]))``).
        Expansion into a sequence allows the generator to fully execute, which
        includes any functionality after the final ``yield`` statement in any
        of the underlying iterators that may perform required clean-up.

        **Non-redundant Processing**
        Certain classification element implementations, as dictated by the
        input factory, may be connected to persistent storage in the
        background.
        Because of this, some classification elements may already "have"
        classification results on construction.
        This method, by default, only computes new classification results for
        descriptor elements whose associated classification element does not
        report as already containing results.
        If the ``overwrite`` flag is True then classifications are computed for
        all input descriptor elements and results are set to their respective
        classification elements regardless of existing result storage.

        :param collections.abc.Iterable[DescriptorElement] descr_iter:
            Iterable of DescriptorElement instances to be classified.
        :param smqtk.representation.ClassificationElementFactory factory:
            Classification element factory. The default factory yields
            MemoryClassificationElement instances.
        :param bool overwrite:
            Recompute classification of the input descriptor and set the
            results to the ClassificationElement produced by the factory.
        :param int d_elem_batch:
            The number of descriptor elements to collect before requesting
            the whole batch's vectors at once via
            ``DescriptorElement.get_many_vectors`` method.

        :raises ValueError: Either: (A) one or more input descriptor elements
            did not have a stored vector, or (B) input descriptor element
            arrays were not all of consistent dimensionality.
        :raises IndexError: Implementation of ``_classify_arrays`` either under
            or over produced classifications relative to the number of input
            descriptor vectors.

        :return: Iterator of result ClassificationElement instances. UUIDs of
            generated ClassificationElement instances will reflect the UUID of
            the DescriptorElement it was computed from.
        """
        log_debug = LOG.debug

        if d_elem_batch <= 0:
            LOG.warning("Descriptor element batching value <= 0, defaulting to using value of 1.")
            d_elem_batch = 1

        # Queue populated by ``iter_tocompute_arrays`` with
        #   ClassificationElement instances paired with a flag indicating
        #   whether a classification was to be computed for that element.
        # Using deques so we can efficiently popleft off of them in the below
        #   for-loop. This way we do not retain elements and booleans for
        #   things we have yielded that would otherwise build up if this method
        #   iterated for a long time.
        elem_and_status_q: Deque[Tuple[ClassificationElement, bool]] = deque()

        # Flag for end of data iteration. When not None will be the index of
        # the last descriptor/classification element to be yielded. This will
        # NOT be the number of elements to be yielded, that would be
        # ``end_of_iter[0]+1``.
        end_of_iter: List[Optional[int]] = [None]

        # TODO: Make generator threadsafe?
        # See: https://anandology.com/blog/using-iterators-and-generators/
        def iter_tocompute_arrays() -> Generator[np.ndarray, None, None]:
            """ Yield descriptor vectors for classification elements that need
            computing yet.
            """
            # Force into an iterator.
            descr_iterator = iter(descr_iter)
            # Running var for the index of final data element in input
            # iterator. This will be -1 or the value of the final index in the
            # parallel lists.
            last_i = -1
            # Make successive islices into iterator of descriptor elements to
            # produces batches. We end when there is nothing left being
            # returned by the iterator
            de_batch_list = \
                list(itertools.islice(descr_iterator, d_elem_batch))
            # Fully qualified path to this classifier implementation type. This
            # should be unique among concrete classifier implementations.
            self_name = f"{self.__module__}.{self.__class__.__name__}"
            while de_batch_list:
                # Get vectors from batch using implementation-level batch
                # aggregation methods where applicable.
                de_batch_vecs = \
                    DescriptorElement.get_many_vectors(de_batch_list)

                for d_elem, d_vec in zip(de_batch_list, de_batch_vecs):
                    d_uid = d_elem.uuid()
                    if d_vec is None:
                        raise ValueError("Encountered DescriptorElement with "
                                         "no vector stored! (UID=`{}`)"
                                         .format(d_uid))
                    c_elem_ = factory.new_classification(self_name, d_uid)
                    already_computed = \
                        not overwrite and c_elem_.has_classifications()
                    elem_and_status_q.append((c_elem_, already_computed))
                    if not already_computed:
                        # Classifications should be computed for this
                        # descriptor
                        log_debug("Yielding descriptor array with UID `{}` "
                                  "for classification generation."
                                  .format(d_uid))
                        yield d_vec
                    else:
                        log_debug("Classification already generated for UID "
                                  "`{}`.".format(d_uid))

                last_i += len(de_batch_vecs)

                # Slice out the next batch of descriptor elements. This will be
                # empty if the iterator has been exhausted.
                de_batch_list = list(
                    itertools.islice(descr_iterator, d_elem_batch)
                )

            end_of_iter[0] = last_i

        classification_iter = self.classify_arrays(iter_tocompute_arrays())
        for c_i, c in enumerate(classification_iter):
            # These pops would fail with an IndexError if there is nothing left
            #   left from parallel allocation within ``iter_tocompute_arrays``.
            # This usually means that the ``self.generate_arrays`` is
            #   generating more vectors than there are descr element slots to
            #   fill.
            try:
                c_elem, c_already_computed = elem_and_status_q.popleft()
            except IndexError:
                # Translate index error into one with a more informative
                # message.
                raise IndexError(
                    "Implementation's ``_classify_arrays`` over-produced "
                    "classifications relative to input descriptor vectors."
                )

            # Forwarding the ``classification_iter`` generator should forward
            # the ``iter_tocompute_arrays`` iterator, thus populating the
            # ``elem_and_status_q`` by some amount. The current ``c`` should be
            # used to populate the next ClassificationElement with a False
            # ``already_computed`` flag. Yield classification elements that
            # already had classifications until we hit an element that was
            # flagged for computation.
            while c_already_computed:
                yield c_elem
                # We clearly have a classification from the result of
                # computation so there should logically be some future element
                # in which to store this result.
                c_elem, c_already_computed = elem_and_status_q.popleft()

            # We've arrived at an element that was flagged for computation, so
            # store the result.
            log_debug("Setting computed classification to element UID=`{}`"
                      .format(c_elem.uuid))
            c_elem.set_classification(c)
            yield c_elem

        # At this point, the ``iter_tocompute_arrays`` iterator should have
        #   completed du eto the ``self.classify_arrays`` method iterating
        #   through it completely, resulting in assignment to
        #   ``end_of_iter[0]``.
        # This also indicates that nothing more should be being added to
        #   ``elem_and_status_q``.
        assert end_of_iter[0] is not None, \
            "EoI value has not yet been assigned a value even though " \
            "``classify_arrays`` completed. Implementation of " \
            "``_classify_arrays`` may not have fully iterated through the " \
            "input numpy.ndarray iterable."

        # Finish yielding any "already-computed" classification elements that
        # are past the last computed element index.
        for ce, already_comp in elem_and_status_q:
            # If an element's already-computed state is False at this point,
            # then the implementation's ``_classify_arrays`` method must not
            # have yielded enough arrays to fill the elements that were flagged
            # for classification.
            if not already_comp:
                raise IndexError(
                    "Implementation's ``_classify_arrays`` under-produced "
                    "classifications to fill elements that were flagged for "
                    "computation."
                )
            yield ce

    def classify_one_element(
        self,
        descr_elem: DescriptorElement,
        factory: ClassificationElementFactory = DFLT_CLASSIFIER_FACTORY,
        overwrite: bool = False
    ) -> ClassificationElement:
        """
        Convenience method around ``classify_elements`` for the single-input
        case.

        See documentation for the :meth:`Classifier.classify_elements` method
        for more information.

        :param DescriptorElement descr_elem:
            Iterable of DescriptorElement instances to be classified.
        :param smqtk.representation.ClassificationElementFactory factory:
            Classification element factory. The default factory yields
            MemoryClassificationElement instances.
        :param bool overwrite:
            Recompute classification of the input descriptor and set the
            results to the ClassificationElement produced by the factory.

        :raises ValueError: The input descriptor element did not have a stored
            vector.
        :raises IndexError: Implementation of ``_classify_arrays`` either under
            or over produced classifications relative to the number of input
            descriptor vectors.

        :return: ClassificationElement instances. UUIDs of the generated
            ClassificationElement instance will reflect the UUID of the
            DescriptorElement it was computed from.
        :rtype: smqtk.representation.ClassificationElement
        """
        return list(self.classify_elements(
            [descr_elem], factory=factory, overwrite=overwrite, d_elem_batch=1
        ))[0]
