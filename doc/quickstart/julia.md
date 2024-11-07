# mlpack in Julia quickstart guide

This page describes how you can quickly get started using mlpack from Julia and
gives a few examples of usage, and pointers to deeper documentation.

This quickstart guide is also available for [C++](cpp.md), [Python](python.md),
[the command line](cli.md), [R](r.md), and [Go](go.md).

## Installing mlpack

Installing the mlpack bindings for Julia is straightforward; you can just use
`Pkg`:

```julia
using Pkg
Pkg.add("mlpack")
```

Building the Julia bindings from scratch is a little more in-depth, though.  For
information on that, follow the instructions in the
[installation guide](../user/install.md#compile-bindings-manually).

## Simple quickstart example

As a really simple example of how to use mlpack from Julia, let's do some
simple classification on a subset of the standard machine learning `covertype`
dataset.  We'll first split the dataset into a training set and a testing set,
then we'll train an mlpack random forest on the training data, and finally we'll
print the accuracy of the random forest on the test dataset.

You can copy-paste this code directly into Julia to run it.  You may need to add
some extra packages with, e.g., `using Pkg; Pkg.add("CSV");
Pkg.add("DataFrames"); Pkg.add("Libz")`.

```julia
using CSV
using DataFrames
using Libz
using mlpack

# Load the dataset from an online URL.  Replace with 'covertype.csv.gz' if you
# want to use on the full dataset.
df = CSV.read(ZlibInflateInputStream(open(download(
        "http://www.mlpack.org/datasets/covertype-small.csv.gz"))))

# Split the labels.
labels = df[!, :label][:]
dataset = select!(df, Not(:label))

# Split the dataset using mlpack.
test, test_labels, train, train_labels = mlpack.preprocess_split(
    dataset,
    input_labels=labels,
    test_ratio=0.3)

# Train a random forest.
rf_model, _, _ = mlpack.random_forest(training=train,
                              labels=train_labels,
                              print_training_accuracy=true,
                              num_trees=10,
                              minimum_leaf_size=3)

# Predict the labels of the test points.
_, predictions, _ = mlpack.random_forest(input_model=rf_model,
                                         test=test)

# Now print the accuracy.  The third return value ('probabilities'), which we
# ignored here, could also be used to generate an ROC curve.
correct = sum(predictions .== test_labels)
print("$(correct) out of $(length(test_labels)) test points correct " *
    "($(correct / length(test_labels) * 100.0)%).\n")
```

We can see that we achieve reasonably good accuracy on the test dataset (80%+);
if we use the full `covertype.csv.gz`, the accuracy should increase
significantly (but training will take longer).

It's easy to modify the code above to do more complex things, or to use
different mlpack learners, or to interface with other machine learning toolkits.

## Using mlpack for movie recommendations

In this example, we'll train a collaborative filtering model using mlpack's
[`cf()`](../user/bindings/julia.md#cf) method.
We'll train this on the
[MovieLens dataset](https://grouplens.org/datasets/movielens/), and then we'll
use the model that we train to give recommendations.

You can copy-paste this code directly into Julia to run it.

```julia
using CSV
using mlpack
using Libz
using DataFrames

# Load the dataset from an online URL.  Replace with 'covertype.csv.gz' if you
# want to use on the full dataset.
ratings = CSV.read(ZlibInflateInputStream(open(download(
        "http://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz"))))
movies = CSV.read(ZlibInflateInputStream(open(download(
        "http://www.mlpack.org/datasets/ml-20m/movies.csv.gz"))))

# Hold out 10% of the dataset into a test set so we can evaluate performance.
ratings_test, _, ratings_train, _ = mlpack.preprocess_split(ratings;
    test_ratio=0.1, verbose=true)

# Train the model.  Change the rank to increase/decrease the complexity of the
# model.
_, cf_model = mlpack.cf(training=ratings_train,
                        test=ratings_test,
                        rank=10,
                        verbose=true,
                        algorithm="RegSVD")

# Now query the 5 top movies for user 1.
output, _ = mlpack.cf(input_model=cf_model,
                      query=[1],
                      recommendations=10,
                      verbose=true,
                      max_iterations=10)

print("Recommendations for user 1:\n")
for i in 1:10
  print("  $(i): $(movies[output[i], :][3])\n")
end
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
easily plug into a data science workflow in Julia.  But the two examples above
have only shown a little bit of the functionality of mlpack.  Lots of other
functions are available with different functionality.  A full list of each of
these commands and full documentation can be found on the following page:

 - [Julia documentation](../user/bindings/julia.md)

You can also use the Julia REPL to explore the `mlpack` module and its
functions; every function comes with comprehensive documentation.

Also, mlpack is much more flexible from C++ and allows much greater
functionality.  So, more complicated tasks are possible if you are willing to
write C++ (or perhaps CxxWrap.jl).  To get started learning about mlpack in C++,
the [C++ quickstart](cpp.md) would be a good place to start.
