<object data="../img/pipeline-top-3.svg" type="image/svg+xml" id="pipeline-top">
</object>

# Transformations

Once data is [loaded](load_save.html) and any necessary
[preprocessing and feature extraction](preprocessing.md) is done,
one of mlpack's transformations can be used to transform data into a
new space.

*Note: this section is under construction and not all functionality is
documented yet.*

## Matrix decompositions

Decompose a matrix into two or more components.

 * [AMF](methods/amf.md): alternating matrix factorization
 * [NMF](methods/nmf.md): non-negative matrix factorization

## Linear transformations

Linearly map a matrix onto a new basis, optionally performing dimensionality
reduction.

 * [PCA](methods/pca.md): principal components analysis
 * [RADICAL](methods/radical.md): an independent components analysis technique

## Metric learning techniques

Learn a [distance metric](core/distances.md) based on a data matrix.

 * [LMNN](methods/lmnn.md): large margin nearest neighbor
 * [NCA](methods/nca.md): neighborhood components analysis

## Coding techniques

Encode data points in a matrix as a combination of points in a dictionary.

 * [LocalCoordinateCoding](methods/local_coordinate_coding.md): local coordinate
   coding with dictionary learning
 * [SparseCoding](methods/sparse_coding.md): sparse coding with dictionary
   learning
