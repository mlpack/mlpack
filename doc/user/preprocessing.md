<object data="../img/pipeline-top-2.svg" type="image/svg+xml" id="pipeline-top">
</object>

# Preprocessing / feature extraction

mlpack provides a number of utilities for data preparation and feature
extraction.  These utilities are generally used just before actually applying
any machine learning [transformations](transformations.md) or
[modeling](modeling.md).

*Note: this section is under construction and not all functionality is
documented yet.*

 * [Normalizing labels](core/normalizing_labels.md): convert labels to/from an
   arbitrary range to `[0, numClasses - 1]`, which is the range that mlpack
   classifiers require.

 * [Dataset splitting](core/split.md): split a dataset into a training and test
   set, optionally including labels.
