/**
 * @file python_quickstart.hpp
 * @author Ryan Curtin

@page python_quickstart mlpack in Python quickstart guide

@section python_quickstart_intro Introduction

This page describes how you can quickly get started using mlpack from Python and
gives a few examples of usage, and pointers to deeper documentation.

This quickstart guide is also available for @ref cli_quickstart "the command-line".

@section python_quickstart_install Installing mlpack

Installing the mlpack bindings for Python is straightforward.  It's easy to use
conda or pip to do this:

@code{.sh}
pip install mlpack/mlpack3
@endcode

@code{.sh}
conda install -c mlpack mlpack
@endcode

Otherwise, we can build the Python bindings from scratch, as follows.  First we
have to install the dependencies (the code below is for Ubuntu), then we can
build and install mlpack.  You can copy-paste the commands into your shell.

@code{.sh}
sudo apt-get install libboost-all-dev g++ cmake libarmadillo-dev python-pip wget
sudo pip install cython setuptools distutils numpy pandas
wget http://www.mlpack.org/files/mlpack-3.0.2.tar.gz
tar -xvzpf mlpack-3.0.2.tar.gz
mkdir -p mlpack-3.0.2/build/ && cd mlpack-3.0.2/build/
cmake ../ && make -j4 && sudo make install
@endcode

More information on the build process and details can be found on the @ref build
page.  You may also need to set the environment variable @c LD_LIBRARY_PATH to
include @c /usr/local/lib/ on most Linux systems.

@code
export LD_LIBRARY_PATH=/usr/local/lib/
@endcode

You can also use the mlpack Docker image on Dockerhub, which has all of the
Python bindings pre-installed:

@code
docker run -it mlpack/mlpack /bin/bash
@endcode

@section python_quickstart_example Simple mlpack quickstart example

As a really simple example of how to use mlpack from Python, let's do some
simple classification on a subset of the standard machine learning @c covertype
dataset.  We'll first split the dataset into a training set and a testing set,
then we'll train an mlpack random forest on the training data, and finally we'll
print the accuracy of the random forest on the test dataset.

You can copy-paste this code directly into Python to run it.

@code{.py}
import mlpack
import pandas as pd
import numpy as np

# Load the dataset from an online URL.  Replace with 'covertype.csv.gz' if you
# want to use on the full dataset.
df = pd.read_csv('http://www.mlpack.org/datasets/covertype-small.csv.gz')

# Split the labels.
labels = df['label']
dataset = df.drop('label', 1)

# Split the dataset using mlpack.  The output comes back as a dictionary,
# which we'll unpack for clarity of code.
output = mlpack.preprocess_split(input=dataset,
                                 input_labels=labels,
                                 test_ratio=0.3)
training_set = output['training']
training_labels = output['training_labels']
test_set = output['test']
test_labels = output['test_labels']

# Train a random forest.
output = mlpack.random_forest(training=training_set,
                              labels=training_labels,
                              print_training_accuracy=True,
                              num_trees=10,
                              minimum_leaf_size=3)
random_forest = output['output_model']

# Predict the labels of the test points.
output = mlpack.random_forest(input_model=random_forest,
                              test=test_set)

# Now print the accuracy.  The 'probabilities' output could also be used
# to generate an ROC curve.
correct = np.sum(output['predictions'] == test_labels)
print(str(correct) + ' correct out of ' + str(len(test_labels)) + ' (' +
    str(100 * float(correct) / float(len(test_labels))) + '%).')
@endcode

We can see that we achieve reasonably good accuracy on the test dataset (80%+);
if we use the full @c covertype.csv.gz, the accuracy should increase
significantly (but training will take longer).

It's easy to modify the code above to do more complex things, or to use
different mlpack learners, or to interface with other machine learning toolkits.

@section python_quickstart_whatelse What else does mlpack implement?

The example above has only shown a little bit of the functionality of mlpack.
Lots of other commands are available with different functionality.  Below is a
list of all the mlpack functionality offered through Python, split into some
categories.

 - Classification techniques: <tt><a
   href="http://www.mlpack.org/docs/mlpack-git/python/adaboost.html">adaboost()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/decision_stump.html">decision_stump()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/decision_tree.html">decision_tree()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/hmm_train.html">hmm_train()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/hmm_generate.html">hmm_generate()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/hmm_loglik.html">hmm_loglik()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/hmm_viterbi.html">hmm_viterbi()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/hoeffding_tree.html">hoeffding_tree()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/logistic_regression.html">logistic_regression()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/nbc.html">nbc()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/perceptron.html">perceptron()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/random_forest.html">random_forest()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/softmax_regression.html">softmax_regression()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/cf.html">cf()</a></tt>

 - Distance-based problems: <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/approx_kfn.html">approx_kfn()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/emst.html">emst()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/fastmks.html">fastmks()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/kfn.html">kfn()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/knn.html">knn()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/krann.html">krann()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/lsh.html">lsh()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/det.html">det()</a></tt>

 - Clustering: <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/kmeans.html">kmeans()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/mean_shift.html">mean_shift()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/gmm_train.html">gmm_train()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/gmm_generate.html">gmm_generate()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/gmm_probability.html">gmm_probability()</a></tt>

 - Transformations: <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/pca.html">pca()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/radical.html">radical()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/local_coordinate_coding.html">local_coordinate_coding()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/sparse_coding.html">sparse_coding()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/nca.html">nca()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/kernel_pca.html">kernel_pca()</a></tt>

 - Regression: <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/linear_regression.html">linear_regression()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/lars.html">lars()</a></tt>

 - Preprocessing/other: <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/preprocess_binarize.html">preprocess_binarize()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/preprocess_split.html">preprocess_split()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/preprocess_describe.html">preprocess_describe()</a></tt>, <tt><a href="http://www.mlpack.org/docs/mlpack-git/python/nmf.html">nmf()</a></tt>

For more information on what mlpack does, see http://www.mlpack.org/about.html.
Next, let's go through another example for providing movie recommendations with
mlpack.

@section python_quickstart_movierecs Using mlpack for movie recommendations

In this example, we'll train a collaborative filtering model using mlpack's
<tt><a href="http://www.mlpack.org/docs/mlpack-git/python/cf.html">cf()</a></tt> method.  We'll train this on the MovieLens dataset from
https://grouplens.org/datasets/movielens/, and then we'll use the model that we
train to give recommendations.

You can copy-paste this code directly into Python to run it.

@code{.py}
import mlpack
import pandas as pd
import numpy as np

# First, load the MovieLens dataset.  This is taken from files.grouplens.org/
# but reposted on mlpack.org as unpacked and slightly preprocessed data.
ratings = pd.read_csv('http://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz')
movies = pd.read_csv('http://www.mlpack.org/datasets/ml-20m/movies.csv.gz')

# Hold out 10% of the dataset into a test set so we can evaluate performance.
output = mlpack.preprocess_split(input=ratings, test_ratio=0.1, verbose=True)
ratings_train = output['training']
ratings_test = output['test']

# Train the model.  Change the rank to increase/decrease the complexity of the
# model.
output = mlpack.cf(training=ratings_train,
                   test=ratings_test,
                   rank=10,
                   verbose=True,
                   algorithm='RegSVD')
cf_model = output['output_model']

# Now query the 5 top movies for user 1.
output = mlpack.cf(input_model=cf_model,
                   query=[[1]],
                   recommendations=10,
                   verbose=True)

# Get the names of the movies for user 1.
print("Recommendations for user 1:")
for i in range(10):
  print("  " + str(i) + ": " + str(movies.loc[movies['movieId'] ==
      output['output'][0, i]].iloc[0]['title']))
@endcode

Here is some example output, showing that user 1 seems to have good taste in
movies:

@code{.unparsed}
Recommendations for user 1:
  0: Casablanca (1942)
  1: Pan's Labyrinth (Laberinto del fauno, El) (2006)
  2: Godfather, The (1972)
  3: Answer This! (2010)
  4: Life Is Beautiful (La Vita è bella) (1997)
  5: Adventures of Tintin, The (2011)
  6: Dark Knight, The (2008)
  7: Out for Justice (1991)
  8: Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
  9: Schindler's List (1993)
@endcode

@section python_quickstart_nextsteps Next steps with mlpack

Now that you have done some simple work with mlpack, you have seen how it can
easily plug into a data science workflow in Python.  A great thing to do next
would be to look at more documentation for the Python mlpack bindings:

 - <a href="http://www.mlpack.org/docs/mlpack-git/python.html">Python mlpack
   binding documentation</a>

Also, mlpack is much more flexible from C++ and allows much greater
functionality.  So, more complicated tasks are possible if you are willing to
write C++ (or perhaps Cython).  To get started learning about mlpack in C++, the
following resources might be helpful:

 - <a href="http://www.mlpack.org/docs/mlpack-git/doxygen/tutorials.html">mlpack
   C++ tutorials</a>
 - <a href="http://www.mlpack.org/docs/mlpack-git/doxygen/build.html">mlpack
   build and installation guide</a>
 - <a href="http://www.mlpack.org/docs/mlpack-git/doxygen/sample.html">Simple
   sample C++ mlpack programs</a>
 - <a href="http://www.mlpack.org/docs/mlpack-git/doxygen/index.html">mlpack
   Doxygen documentation homepage</a>

 */
