<object data="../img/pipeline-top-1.svg" type="image/svg+xml" id="pipeline-top">
</object>

# Data loading and I/O

mlpack provides the `data::Load()` and `data::Save()` functions to load and save
[Armadillo matrices](matrices.md) (e.g. numeric and categorical datasets) and
any mlpack object via the [cereal](https://uscilab.github.io/cereal/)
serialization toolkit.  A number of other utilities related to loading and
saving data and objects are also available. To enable you to start quickly with
mlpack data loading / saving interface, here is how the current function
signature looks like:

## `data::Load()`

 - `data::Load(filename, object)`
 - `data::Load(filename, object, Option1 + Option2 + ...)`
 - `data::Load(filename, object, options)`

 - `data::Save(filename, object)`
 - `data::Save(filename, object, Option1 + Option2 + ...)`
 - `data::Save(filename, object, options)`

   * Load / save an object (either an mlpack model or a matrix) from `filename`.
   * By default, the format is autodetected and part of our data options.
     The format can be specified with [DataOptions](#dataoptions)
   * Options can be specified as standalone or an
     DataOptions object, [see below](#dataoptions)

Usage Examples:

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat x;
mlpack::data::Load("iris.csv", x);

std::cout << "Loaded iris.csv; size " << x.n_rows << " x " << x.n_cols << "."
    << std::endl;
```

If you would like to specify the type of the file for example:

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat x;
mlpack::data::Load("iris.csv", x, CSV);

std::cout << "Loaded iris.csv; size " << x.n_rows << " x " << x.n_cols << "."
    << std::endl;
```

For more information regarding loading / saving each data type, please visit
the following pages:

 * [Numeric data](#data/load_save_numeric.md)
 * [Categorical data](#data/load_save_categorical.md)
 * [Image data](#data/load_save_images.md)
 * [mlpack objects / models](#data/load_save_models.md)
 * [Data Options](#data/data_options.md)
