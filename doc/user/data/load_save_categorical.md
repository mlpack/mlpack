## Mixed categorical data

mlpack support loading / saving mixed categorical data, e.g., data where some
dimensions take only categorical values (e.g. `0`, `1`, `2`, etc.).  When using
mlpack, string data and other non-numerical data must be mapped to categorical
values and represented as part of an `arma::mat`.  Category information is
stored in an auxiliary `data::TextOptions::DatasetInfo()` object.

### `data::TextOptions::DatasetInfo`

<!-- TODO: also document in core.md? -->

mlpack represents categorical data via the use of the auxiliary
`data::TextOptions::DatasetInfo` object, which stores information about which
dimensions are numeric or categorical and allows conversion from the original
category values to the numeric values used to represent those categories.


#### Accessing and setting properties

 - `data::DatasetInfo info = opts.DatasetInfo();`

 - `info.Type(d)`
   * Get the type (categorical or numeric) of dimension `d`.
   * Returns a `data::Datatype`, either `data::Datatype::numeric` or
     `data::Datatype::categorical`.
   * Calling `info.Type(d) = t` will set a dimension to type `t`, but this
     should only be done before `info` is used with `data::Load()` or
     `data::Save()`.

 - `info.NumMappings(d)`
   * Get the number of categories in dimension `d` as a `size_t`.
   * Returns `0` if dimension `d` is numeric.

 - `info.Dimensionality()`
   * Return the dimensionality of the object as a `size_t`.

---

#### Map to and from numeric values

 - `info.MapString<double>(value, d)`
   * Given `value` (a `std::string`), return the `double` representing the
     categorical mapping (an integer value) of `value` in dimension `d`.
   * If a mapping for `value` does not exist in dimension `d`, a new mapping is
     created, and `info.NumMappings(d)` is increased by one.
   * If dimension `d` is numeric and `value` cannot be parsed as a numeric
     value, then dimension `d` is changed to categorical and a new mapping is
     returned.

 - `info.UnmapString(mappedValue, d)`
   * Given `mappedValue` (a `size_t`), return the `std::string` containing the
     original category that mapped to the value `mappedValue` in dimension `d`.
   * If dimension `d` is not categorical, a `std::invalid_argument` is thrown.

---

### Loading categorical data

With a `data::DatasetInfo` object, categorical data can be loaded / saved in a
similar way to numeric data. However, the `Categorical` flag needs to be set:

 - `TextOptions opts;`

 - `opts.Categorical() = true;`

 - `data::Load(filename, matrix, opts)`
 - `data::Save(filename, matrix, opts)`

   * `filename` is a `std::string` with a path to the file to be loaded.

   * The format is auto-detected based on the extension of the filename and the
     contents of the file:
     - `.csv`, `.tsv`, or `.txt` for CSV/TSV (tab-separated)/ASCII
       (space-separated)
     - `.arff` for [ARFF](https://ml.cms.waikato.ac.nz/weka/arff.html)

   * `matrix` is an `arma::mat&`, `arma::Mat<size_t>&`, or similar (e.g., a
     reference to an Armadillo object that data will be loaded into or saved
     from).

   * `opts` is a `data::TextOptions` object.  Internally, `DatasetInfo` will
     be populated with the category information of the file when loading, and
     used to unmap values when saving. It can be accessed using
     `opts.DatasetInfo()`

   * A `bool` is returned indicating whether the operation was successful.

Another simplified signature can be used by specifying const boolean options
directly and add them using the `+` operator.

 - `data::Load(filename, matrix, NoFatal + Transpose + Categorical)`
 - `data::Save(filename, matrix, NoFatal + Transpose + Categorical)`

   * `filename` is a `std::string` with a path to the file to be loaded.

   * The format is auto-detected based on the extension of the filename and the
     contents of the file:
     - `.csv`, `.tsv`, or `.txt` for CSV/TSV (tab-separated)/ASCII
       (space-separated)
     - `.arff` for [ARFF](https://ml.cms.waikato.ac.nz/weka/arff.html)

   * `matrix` is an `arma::mat&`, `arma::Mat<size_t>&`, `arma::sp_mat&`, or
     similar (e.g., a reference to an Armadillo object that data will be loaded
     into or saved from).

   * `NoFatal` a const boolean that returns a warning if  `MLPACK_PRINT_WARN`
     is defined during compile time.
   
   * `Fatal` a const boolean that throws a `std::runtime_error` on failure.

   * `NoTranspose` a const boolean that loads the matrix as a row-major matrix,
     usually not desired in mlpack.

   * `Transpose` a const boolean that transposes the matrix during loading in
     order to have a colunm-major matrix. This is the desired format for (CSV/TSV/ASCII).

   * `Categotrical` a const boolean indicates that the data contains categorical
     values and needs to be mapped. Using this flag will not allow the user to
     access the mapped values.

   * A `bool` is returned indicating whether the operation was successful.

   * A full list of these options is available: [DataOptions](#data-options),
     [MatrixOptions](#matrix-options) and [TextOptions](#text-options). 

---

Example usage to load and manipulate an ARFF file.

```c++
// Load a categorical dataset.
arma::mat dataset;

// Define a Text Options to load categorical data.
mlpack::data::TextOptions opts;
opts.Fatal() = true;
opts.NoTranspose() = false; 
opts.Categorical() = true;

// See https://datasets.mlpack.org/covertype.train.arff.
mlpack::data::Load("covertype.train.arff", dataset, opts);

arma::Row<size_t> labels;
// See https://datasets.mlpack.org/covertype.train.labels.csv.
mlpack::data::Load("covertype.train.labels.csv", labels, opts);

// Print information about the data.
std::cout << "The data in 'covertype.train.arff' has: " << std::endl;
std::cout << " - " << dataset.n_cols << " points." << std::endl;
std::cout << " - " << opts.DatasetInfo().Dimensionality() << " dimensions."
    << std::endl;

// Print information about each dimension.
for (size_t d = 0; d < opts.DatasetInfo().Dimensionality(); ++d)
{
  if (opts.DatasetInfo.Type(d) == mlpack::data::Datatype::categorical)
  {
    std::cout << " - Dimension " << d << " is categorical with "
        << opts.DatasetInfo().NumMappings(d) << " categories." << std::endl;
  }
  else
  {
    std::cout << " - Dimension " << d << " is numeric." << std::endl;
  }
}

// Modify the 5th point.  Increment any numeric values, and set any categorical
// values to the string "hooray!".
for (size_t d = 0; d < opts.DatasetInfo.Dimensionality(); ++d)
{
  if (opts.DatasetInfo().Type(d) == mlpack::data::Datatype::categorical)
  {
    // This will create a new mapping if the string "hooray!" does not already
    // exist as a category for dimension d..
    dataset(d, 4) = opts.DatasetInfo().MapString<double>("hooray!", d);
  }
  else
  {
    dataset(d, 4) += 1.0;
  }
}
```

---

Example usage to manually create a `data::DatasetOptions` object.

```c++
// This will manually create the following data matrix (shown as it would appear
// in a CSV):
//
// 1, TRUE, "good", 7.0, 4
// 2, FALSE, "good", 5.6, 3
// 3, FALSE, "bad", 6.1, 4
// 4, TRUE, "bad", 6.1, 1
// 5, TRUE, "unknown", 6.3, 0
// 6, FALSE, "unknown", 5.1, 2
//
// Although the last dimension is numeric, we will take it as a categorical
// dimension.

arma::mat dataset(5, 6); // 6 data points in 5 dimensions.
mlpack::data::DatasetInfo info(5);

// Set types of dimensions.  By default they are numeric so we only set
// categorical dimensions.
info.Type(1) = mlpack::data::Datatype::categorical;
info.Type(2) = mlpack::data::Datatype::categorical;
info.Type(4) = mlpack::data::Datatype::categorical;

// The first dimension is numeric.
dataset(0, 0) = 1;
dataset(0, 1) = 2;
dataset(0, 2) = 3;
dataset(0, 3) = 4;
dataset(0, 4) = 5;
dataset(0, 5) = 6;

// The second dimension is categorical.
dataset(1, 0) = info.MapString<double>("TRUE", 1);
dataset(1, 1) = info.MapString<double>("FALSE", 1);
dataset(1, 2) = info.MapString<double>("FALSE", 1);
dataset(1, 3) = info.MapString<double>("TRUE", 1);
dataset(1, 4) = info.MapString<double>("TRUE", 1);
dataset(1, 5) = info.MapString<double>("FALSE", 1);

// The third dimension is categorical.
dataset(2, 0) = info.MapString<double>("good", 2);
dataset(2, 1) = info.MapString<double>("good", 2);
dataset(2, 2) = info.MapString<double>("bad", 2);
dataset(2, 3) = info.MapString<double>("bad", 2);
dataset(2, 4) = info.MapString<double>("unknown", 2);
dataset(2, 5) = info.MapString<double>("unknown", 2);

// The fourth dimension is numeric.
dataset(3, 0) = 7.0;
dataset(3, 1) = 5.6;
dataset(3, 2) = 6.1;
dataset(3, 3) = 6.1;
dataset(3, 4) = 6.3;
dataset(3, 5) = 5.1;

// The fifth dimension is categorical.  Note that `info` will choose to assign
// category values in the order they are seen, even if the category can be
// parsed as a number.  So, here, the value '4' will be assigned category '0',
// since it is seen first.
dataset(4, 0) = info.MapString<double>("4", 4);
dataset(4, 1) = info.MapString<double>("3", 4);
dataset(4, 2) = info.MapString<double>("4", 4);
dataset(4, 3) = info.MapString<double>("1", 4);
dataset(4, 4) = info.MapString<double>("0", 4);
dataset(4, 5) = info.MapString<double>("2", 4);

// Print the dataset with mapped categories.
dataset.print("Dataset with mapped categories");

// Print the mappings for the third dimension.
std::cout << "Mappings for dimension 3: " << std::endl;
for (size_t i = 0; i < info.NumMappings(2); ++i)
{
  std::cout << " - \"" << info.UnmapString(i, 2) << "\" maps to " << i << "."
      << std::endl;
}

// Now `dataset` is ready for use with an mlpack algorithm that supports
// categorical data.
```

---

