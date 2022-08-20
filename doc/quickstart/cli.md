# mlpack command-line quickstart guide

This page describes how you can quickly get started using mlpack from the
command-line and gives a few examples of usage, and pointers to deeper
documentation.

This quickstart guide is also available for [C++](cpp.md), [Python](python.md),
[R](R.md), [Julia](julia.md), and [Go](go.md).

## Installing mlpack

Installing mlpack is straightforward and can be done with your system's package
manager.  For instance, for Ubuntu or Debian the command is simply

```sh
sudo apt-get install mlpack-bin
```

On Fedora or Red Hat:

```sh
sudo dnf install mlpack
```

If you use a different distribution, mlpack may be packaged under a different
name.  And if it is not packaged, you can use a Docker image from Dockerhub:

```sh
docker run -it mlpack/mlpack /bin/bash
```

This Docker image has mlpack's command-line bindings already built and
installed.

If you prefer to build mlpack from scratch, see the
[main README](../../README.md).

## Simple quickstart example

As a really simple example of how to use mlpack from the command-line, let's do
some simple classification on a subset of the standard machine learning
`covertype` dataset.  We'll first split the dataset into a training set and a
testing set, then we'll train an mlpack random forest on the training data, and
finally we'll print the accuracy of the random forest on the test dataset.

You can copy-paste this code directly into your shell to run it.

```sh
# Get the dataset and unpack it.
wget https://www.mlpack.org/datasets/covertype-small.data.csv.gz
wget https://www.mlpack.org/datasets/covertype-small.labels.csv.gz
gunzip covertype-small.data.csv.gz covertype-small.labels.csv.gz

# Split the dataset; 70% into a training set and 30% into a test set.
# Each of these options has a shorthand single-character option but here we type
# it all out for clarity.
mlpack_preprocess_split                                       \
    --input_file covertype-small.data.csv                     \
    --input_labels_file covertype-small.labels.csv            \
    --training_file covertype-small.train.csv                 \
    --training_labels_file covertype-small.train.labels.csv   \
    --test_file covertype-small.test.csv                      \
    --test_labels_file covertype-small.test.labels.csv        \
    --test_ratio 0.3                                          \
    --verbose

# Train a random forest.
mlpack_random_forest                                  \
    --training_file covertype-small.train.csv         \
    --labels_file covertype-small.train.labels.csv    \
    --num_trees 10                                    \
    --minimum_leaf_size 3                             \
    --print_training_accuracy                         \
    --output_model_file rf-model.bin                  \
    --verbose

# Now predict the labels of the test points and print the accuracy.
# Also, save the test set predictions to the file 'predictions.csv'.
mlpack_random_forest                                    \
    --input_model_file rf-model.bin                     \
    --test_file covertype-small.test.csv                \
    --test_labels_file covertype-small.test.labels.csv  \
    --predictions_file predictions.csv                  \
    --verbose
```

We can see by looking at the output that we achieve reasonably good accuracy on
the test dataset (80%+).  The file `predictions.csv` could also be used by
other tools; for instance, we can easily calculate the number of points that
were predicted incorrectly:

```sh
$ diff -U 0 predictions.csv covertype-small.test.labels.csv | grep '^@@' | wc -l
```

It's easy to modify the code above to do more complex things, or to use
different mlpack learners, or to interface with other machine learning toolkits.

## Using mlpack for movie recommendations

In this example, we'll train a collaborative filtering model using mlpack's
`mlpack_cf` program.  We'll train this on the
[MovieLens dataset](https://grouplens.org/datasets/movielens/), and then we'll
use the model that we train to give recommendations.

You can copy-paste this code directly into the command line to run it.

```sh
wget https://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz
wget https://www.mlpack.org/datasets/ml-20m/movies.csv.gz
gunzip ratings-only.csv.gz
gunzip movies.csv.gz

# Hold out 10% of the dataset into a test set so we can evaluate performance.
mlpack_preprocess_split                 \
    --input_file ratings-only.csv       \
    --training_file ratings-train.csv   \
    --test_file ratings-test.csv        \
    --test_ratio 0.1                    \
    --verbose

# Train the model.  Change the rank to increase/decrease the complexity of the
# model.
mlpack_cf                             \
    --training_file ratings-train.csv \
    --test_file ratings-test.csv      \
    --rank 10                         \
    --algorithm RegSVD                \
    --output_model_file cf-model.bin  \
    --verbose

# Now query the 5 top movies for user 1.
echo "1" > query.csv;
mlpack_cf                             \
    --input_model_file cf-model.bin   \
    --query_file query.csv            \
    --recommendations 10              \
    --output_file recommendations.csv \
    --verbose

# Get the names of the movies for user 1.
echo "Recommendations for user 1:"
for i in `seq 1 10`; do
    item=`cat recommendations.csv | awk -F',' '{ print $'$i' }'`;
    head -n $(($item + 2)) movies.csv | tail -1 | \
        sed 's/^[^,]*,[^,]*,//' | \
        sed 's/\(.*\),.*$/\1/' | sed 's/"//g';
done
```

Here is some example output, showing that user 1 seems to have good taste in
movies:

```
Recommendations for user 1:
Casablanca (1942)
Pan's Labyrinth (Laberinto del fauno, El) (2006)
Godfather, The (1972)
Answer This! (2010)
Life Is Beautiful (La Vita è bella) (1997)
Adventures of Tintin, The (2011)
Dark Knight, The (2008)
Out for Justice (1991)
Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
Schindler's List (1993)
```
## Next steps with mlpack

Now that you have done some simple work with mlpack, you have seen how it can
easily plug into a data science production workflow for the command line.  But
these two examples have only shown a little bit of the functionality of mlpack.
Lots of other commands are available with different functionality.  A full list
of commands and full documentation for each can be found on the following page:

 - [CLI program documentation](https://www.mlpack.org/doc/stable/cli_documentation.html)

Also, mlpack is much more flexible from C++ and allows much greater
functionality.  So, more complicated tasks are possible if you are willing to
write C++.  To get started learning about mlpack in C++, the
[C++ quickstart](cpp.md) is a good place to start.
