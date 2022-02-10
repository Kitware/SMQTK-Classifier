Pending Release Notes
=====================

Updates / New Features
----------------------

Algorithms

* Added scikit-learn SVM classifier.

CI

* Added workflow to inherit the smqtk-core publish workflow.

* Updated CI unittests workflow to include codecov reporting and to run
  nightly.

Dependencies

* Added `libsvm-official` 3.15.0 as an extras dependency to provide the backing
  for the existing libsvm-based plugin.

Miscellaneous

* Added a wrapper script to pull the versioning/changelog update helper from
  smqtk-core to use here without duplication.

Fixes
-----

CI

* Modified CI unittests workflow to run for PRs targetting branches that match
  the `release*` glob.

Dependency Versions

* Updated the developer dependency and locked version of ipython to address a
  security vulnerability.

* Removed `jedi = "^0.17"` requirement and updated to `ipython = "^7.17.3"`
  since recent ipython update appropriately addresses the dependency.
