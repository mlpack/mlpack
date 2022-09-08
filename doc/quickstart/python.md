# mlpack in Python quickstart guide

This page describes how you can quickly get started using mlpack from Python and
gives a few examples of usage, and pointers to deeper documentation.

This quickstart guide is also available for [C++](cpp.md),
[the command line](cli.md), [Julia](julia.md), [R](R.md), and [Go](go.md).

## Installing mlpack

Installing the mlpack bindings for Python is straightforward.  It's easy to use
`conda` or `pip` to do this:

```sh
pip install mlpack
```

```sh
conda install -c conda-forge mlpack
```

You can also use the mlpack Docker image on Dockerhub, which has all of the
Python bindings pre-installed:

```sh
docker run -it mlpack/mlpack /bin/bash
```

Otherwise, you can build the Python bindings from scratch using the
documentation in the [main README](../../README.md).

## Simple mlpack quickstart example

As a really simple example of how to use mlpack from Python, let's do some
simple classification on a subset of the standard machine learning `covertype`
dataset.  We'll first split the dataset into a training set and a testing set,
then we'll train an mlpack random forest on the training data, and finally we'll
print the accuracy of the random forest on the test dataset.

You can copy-paste this code directly into Python to run it.

```py
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
correct = np.sum(
    output['predictions'] == np.reshape(test_labels, (test_labels.shape[0],)))
print(str(correct) + ' correct out of ' + str(len(test_labels)) + ' (' +
    str(100 * float(correct) / float(len(test_labels))) + '%).')
```

We can see that we achieve reasonably good accuracy on the test dataset (80%+);
if we use the full `covertype.csv.gz`, the accuracy should increase
significantly (but training will take longer).

It's easy to modify the code above to do more complex things, or to use
different mlpack learners, or to interface with other machine learning toolkits.

## Using mlpack for movie recommendations

In this example, we'll train a collaborative filtering model using mlpack's
[`cf()`](https://www.mlpack.org/doc/stable/python_documentation.html#cf) method.
We'll train this on the
[MovieLens dataset](https://grouplens.org/datasets/movielens/), and then we'll
use the model that we train to give recommendations.

You can copy-paste this code directly into Python to run it.

```py
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
```

Here is some example output, showing that user 1 seems to have good taste in
movies:

```
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
```

## Next steps with mlpack

Now that you have done some simple work with mlpack, you have seen how it can
easily plug into a data science workflow in Python.  But the two examples above
have only shown a little bit of the functionality of mlpack.  Lots of other
commands are available with different functionality.  A full list of each of
these commands and full documentation can be found on the following page:

 - [Python documentation](https://www.mlpack.org/doc/stable/python_documentation.html)

Also, mlpack is much more flexible from C++ and allows much greater
functionality.  So, more complicated tasks are possible if you are willing to
write C++ (or perhaps Cython).  To get started learning about mlpack in C++, the
[C++ quickstart](cpp.md) would be a good place to go.
