SMQTK-Classifier Pending Release Notes
======================================


Updates / New Features
----------------------

Dependencies

* Remove dependency on ``setuptool``'s ``pkg_resources`` module.
  Taking the stance of bullet number 5 in from `Python's Packaging User-guide`_
  with regards to getting this package's version.
  The "needs to be installed" requirement from before is maintained.

* Added ``ipython`` (and appropriately supporting version of ``jedi``) as
  development dependencies.
  Minimum versioning is set to support python 3.6 (current versions follow
  `NEP 29`_ and thus require python 3.7+).

Testing

* Added terminal-output coverage report in the standard pytest config in the
  ``pyproject.toml`` file.

Fixes
-----


.. _Python's Packaging User-guide: https://packaging.python.org/guides/single-sourcing-package-version/
.. _NEP 29: https://packaging.python.org/guides/single-sourcing-package-version/
