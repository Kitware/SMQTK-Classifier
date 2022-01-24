Pending Release Notes
=====================

Updates / New Features
----------------------

Algorithms

* Add scikit-learn SVM classifier.

CI

* Add workflow to inherit the smqtk-core publish workflow.

Miscellaneous

* Add a wrapper script to pull the versioning/changelog update helper from
  smqtk-core to use here without duplication.

Fixes
-----

CI

* Also run CI unittests for PRs targetting branches that match the `release*`
  glob.

Dependency Versions

* Update the developer dependency and locked version of ipython to address a
  security vulnerability.
