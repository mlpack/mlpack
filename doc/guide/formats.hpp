/*! @page formatdoc File formats in mlpack

@section formatintro Introduction

mlpack supports a wide variety of data and model formats for use in both its
command-line programs and in C++ programs using mlpack via the
mlpack::data::Load() function.  This tutorial discusses the formats that are
supported and how to use them.

@section formattypes Supported dataset types

Datasets in mlpack are represented internally as sparse or dense numeric
matrices (specifically, as \c arma::mat or \c arma::sp_mat or similar).  This
means that when datasets are loaded from file, they must be converted to a
suitable numeric representation.  Therefore, in general, datasets on disk should
contain only numeric features in order to be loaded successfully by mlpack.

The types of datasets that mlpack can load are roughly the same as the types of
matrices that Armadillo can load.  However, the load functionality that mlpack
provides <b>only supports loading dense datasets</b>.  When datasets are loaded
by mlpack, <b>the file's type is detected using the file's extension</b>.
mlpack supports the following file types:

 - csv (comma-separated values), denoted by .csv or .txt
 - tsv (tab-separated values), denoted by .tsv, .csv, or .txt
 - ASCII (raw ASCII, with space-separated values), denoted by .txt
 - Armadillo ASCII (Armadillo's text format with a header), denoted by .txt
 - PGM, denoted by .pgm
 - PPM, denoted by .ppm
 - Armadillo binary, denoted by .bin
 - Raw binary, denoted by .bin \b "(note: this will be loaded as"
   \b "one-dimensional data, which is likely not what is desired.)"
 - HDF5, denoted by .hdf, .hdf5, .h5, or .he5 (<b>note: HDF5 must be enabled"
   in the Armadillo configuration</b>)
 - ARFF, denoted by .arff (<b>note: this is not supported by all mlpack"
   command-line programs </b>; see \ref formatcat )

Datasets that are loaded by mlpack should be stored with <b>one row for
one point</b> and <b>one column for one dimension</b>.  Therefore, a dataset
with three two-dimensional points \f$(0, 1)\f$, \f$(3, 1)\f$, and \f$(5, -5)\f$
would be stored in a csv file as:

\code
0, 1
3, 1
5, -5
\endcode

As noted earlier, the format is automatically detected at load time.  Therefore,
a dataset can be loaded in many ways:

\code
$ mlpack_logistic_regression -t dataset.csv -v
[INFO ] Loading 'dataset.csv' as CSV data.  Size is 32 x 37749.
...

$ mlpack_logistic_regression -t dataset.txt -v
[INFO ] Loading 'dataset.txt' as raw ASCII formatted data.  Size is 32 x 37749.
...

$ mlpack_logistic_regression -t dataset.h5 -v
[INFO ] Loading 'dataset.h5' as HDF5 data.  Size is 32 x 37749.
...
\endcode

Similarly, the format to save to is detected by the extension of the given
filename.

@section formatcpp Loading simple matrices in C++

When C++ is being written, the mlpack::data::Load() and mlpack::data::Save()
functions are used to load and save datasets, respectively.  These functions
should be preferred over the built-in Armadillo \c .load() and \c .save()
functions.

Matrices in mlpack are column-major, meaning that each column should correspond
to a point in the dataset and each row should correspond to a dimension; for
more information, see \ref matrices .  This is at odds with how the data is
stored in files; therefore, a transposition is required during load and save.
The mlpack::data::Load() and mlpack::data::Save() functions do this
automatically (unless otherwise specified), which is why they are preferred over
the Armadillo functions.

To load a matrix from file, the call is straightforward.  After creating a
matrix object, the data can be loaded:

\code
arma::mat dataset; // The data will be loaded into this matrix.
mlpack::data::Load("dataset.csv", dataset);
\endcode

Saving matrices is equally straightforward.  The code below generates a random
matrix with 10 points in 3 dimensions and saves it to a file as HDF5.

\code
// 3 dimensions (rows), with 10 points (columns).
arma::mat dataset = arma::randu<arma::mat>(3, 10);
mlpack::data::Save("dataset.h5", dataset);
\endcode

As with the command-line programs, the type of data to be loaded is
automatically detected from the filename extension.  For more details, see the
mlpack::data::Load() and mlpack::data::Save() documentation.

@section sparseload Dealing with sparse matrices

As mentioned earlier, support for loading sparse matrices in mlpack is not
available at this time.  To use a sparse matrix with mlpack code, you will have
to write a C++ program instead of using any of the command-line tools, because
the command-line tools all use dense datasets internally.  (There is one
exception: the \c mlpack_cf program, for collaborative filtering, loads sparse
coordinate lists.)

In addition, the \c mlpack::data::Load() function does not support loading any
sparse format; so the best idea is to use undocumented Armadillo functionality
to load coordinate lists.  Suppose you have a coordinate list file like the one
below:

\code
$ cat cl.csv
0 0 0.332
1 3 3.126
4 4 1.333
\endcode

This represents a 5x5 matrix with three nonzero elements.  We can load this
using Armadillo:

\code
arma::sp_mat matrix;
matrix.load("cl.csv", arma::coord_ascii);
matrix = matrix.t(); // We must transpose after load!
\endcode

The transposition after loading is necessary if the coordinate list is in
row-major format (that is, if each row in the matrix represents a point and each
column represents a feature).  Be sure that the matrix you use with mlpack
methods has points as columns and features as rows!  See \ref matrices for more
information.

@section formatcat Categorical features and command line programs

In some situations it is useful to represent data not just as a numeric matrix
but also as categorical data (i.e. with numeric but unordered categories).  This
support is useful for, e.g., decision trees and other models that support
categorical features.

In some machine learning situations, such as, e.g., decision trees, categorical
data can be used.  Categorical data might look like this (in CSV format):

\code
0, 1, "true", 3
5, -2, "false", 5
2, 2, "true", 4
3, -1, "true", 3
4, 4, "not sure", 0
0, 7, "false", 6
\endcode

In the example above, the third dimension (which takes values "true", "false",
and "not sure") is categorical.  mlpack can load and work with this data, but
the strings must be mapped to numbers, because all dataset in mlpack are
represented by Armadillo matrix objects.

From the perspective of an mlpack command-line program, this support is
transparent; mlpack will attempt to load the data file, and if it detects
entries in the file that are not numeric, it will map them to numbers and then
print, for each dimension, the number of mappings.  For instance, if we run the
\c mlpack_hoeffding_tree program (which supports categorical data) on the
dataset above (stored as dataset.csv), we receive this output during loading:

\code
$ mlpack_hoeffding_tree -t dataset.csv -l dataset.labels.csv -v
[INFO ] Loading 'dataset.csv' as CSV data.  Size is 6 x 4.
[INFO ] 0 mappings in dimension 0.
[INFO ] 0 mappings in dimension 1.
[INFO ] 3 mappings in dimension 2.
[INFO ] 0 mappings in dimension 3.
...
\endcode

Currently, only the \c mlpack_hoeffding_tree program supports loading
categorical data, and this is also the only program that supports loading an
ARFF dataset.

@section formatcatcpp Categorical features and C++

When writing C++, loading categorical data is slightly more tricky: the mappings
from strings to integers must be preserved.  This is the purpose of the
mlpack::data::DatasetInfo class, which stores these mappings and can be used and
load and save time to apply and de-apply the mappings.

When loading a dataset with categorical data, the overload of
mlpack::data::Load() that takes an mlpack::data::DatasetInfo object should be
used.  An example is below:

\code
arma::mat dataset; // Load into this matrix.
mlpack::data::DatasetInfo info; // Store information about dataset in this.

// Load the ARFF dataset.
mlpack::data::Load("dataset.arff", dataset, info);
\endcode

After this load completes, the \c info object will hold the information about
the mappings necessary to load the dataset.  It is possible to re-use the
\c DatasetInfo object to load another dataset with the same mappings.  This is
useful when, for instance, both a training and test set are being loaded, and it
is necessary that the mappings from strings to integers for categorical features
are identical.  An example is given below.

\code
arma::mat trainingData; // Load training data into this matrix.
mlpack::data::DatasetInfo info; // This will store the mappings.

// Load the training data, and create the mappings in the 'info' object.
mlpack::data::Load("training_data.arff", trainingData, info);

// Load the test data, but re-use the 'info' object with the already initialized
// mappings.  This means that the same mappings will be applied to the test set.
mlpack::data::Load("test_data.arff", trainingData, info);
\endcode

When saving data, pass the same DatasetInfo object it was loaded with in order
to unmap the categorical features correctly.  The example below demonstrates
this functionality: it loads the dataset, increments all non-categorical
features by 1, and then saves the dataset with the same DatasetInfo it was
loaded with.

\code
arma::mat dataset; // Load data into this matrix.
mlpack::data::DatasetInfo info; // This will store the mappings.

// Load the dataset.
mlpack::data::Load("dataset.tsv", dataset, info);

// Loop over all features, and add 1 to all non-categorical features.
for (size_t i = 0; i < info.Dimensionality(); ++i)
{
  // The Type() function returns whether or not the data is numeric or
  // categorical.
  if (info.Type(i) != mlpack::data::Datatype::categorical)
    dataset.row(i) += 1.0;
}

// Save the modified dataset using the same DatasetInfo.
mlpack::data::Save("dataset-new.tsv", dataset, info);
\endcode

There is more functionality to the DatasetInfo class; for more information, see
the mlpack::data::DatasetInfo documentation.

@section formatmodels Loading and saving models

Using \c boost::serialization, mlpack is able to load and save machine learning
models with ease.  These models can currently be saved in three formats:

 - binary (.bin); this is not human-readable, but it is small
 - text (.txt); this is sort of human-readable and relatively small
 - xml (.xml); this is human-readable but very verbose and large

The type of file to save is determined by the given file extension, as with the
other loading and saving functionality in mlpack.  Below is an example where a
dataset stored as TSV and labels stored as ASCII text are used to train a
logistic regression model, which is then saved to model.xml.

\code
$ mlpack_logistic_regression -t training_dataset.tsv -l training_labels.txt \
> -M model.xml
\endcode

Many mlpack command-line programs have support for loading and saving models
through the \c --input_model_file (\c -m) and \c --output_model_file (\c -M)
options; for more information, see the documentation for each program
(accessible by passing \c --help as a parameter).

@section formatmodels Loading and saving models in C++

mlpack uses the \c boost::serialization library internally to perform loading
and saving of models, and provides convenience overloads of mlpack::data::Load()
and mlpack::data::Save() to load and save these models.

To be serializable, a class must implement the method

\code
template<typename Archive>
void Serialize(Archive& ar, const unsigned int version);
\endcode

\note
For more information on this method and how it works, see the
boost::serialization documentation at http://www.boost.org/libs/serialization/doc/
.  Note that mlpack uses a \c Serialize()
method and not a \c serialize() method, and also mlpack uses the
mlpack::data::CreateNVP() method instead of \c BOOST_SERIALIZATION_NVP() ; this
is for coherence with the mlpack style guidelines, and is done via a
particularly complex bit of template metaprogramming in
src/mlpack/core/data/serialization_shim.hpp (read that file if you want your
head to hurt!).

\note
Examples of Serialize() methods can be found in most classes; one fairly
straightforward example is found \ref mlpack::math::Range::Serialize()
"in the mlpack::math::Range class".  A more complex example is found \ref
mlpack::tree::BinarySpaceTree::Serialize()
"in the mlpack::tree::BinarySpaceTree class".

Using the mlpack::data::Load() and mlpack::data::Save() classes is easy if the
type being saved has a \c Serialize() method implemented: simply call either
function with a filename, a name for the object to save, and the object itself.
The example below, for instance, creates an mlpack::math::Range object and saves
it as range.txt.  Then, that range is loaded from file into another
mlpack::math::Range object.

\code
// Create range and save it.
mlpack::math::Range r(0.0, 5.0);
mlpack::data::Save("range.txt", "range", r);

// Load into new range.
mlpack::math::Range newRange;
mlpack::data::Load("range.txt", "range", newRange);
\endcode

It is important to be sure that you load the appropriate type; if you save, for
instance, an mlpack::regression::LogisticRegression object and attempt to load
it as an mlpack::math::Range object, the load will fail and an exception will be
thrown.  (When the object is saved as binary (.bin), it is possible that the
load will not fail, but instead load with mangled data, which is perhaps even
worse!)

@section formatfinal Final notes

If the examples here are unclear, it would be worth looking into the ways that
mlpack::data::Load() and mlpack::data::Save() are used in the code.  Some
example files that may be useful to this end:

 - src/mlpack/methods/logistic_regression/logistic_regression_main.cpp
 - src/mlpack/methods/hoeffding_trees/hoeffding_tree_main.cpp
 - src/mlpack/methods/neighbor_search/knn_main.cpp

If you are interested in adding support for more data types to mlpack, it would
be preferable to add the support upstream to Armadillo instead, so that may be a
better direction to go first.  Then very little code modification for mlpack
will be necessary.

*/
