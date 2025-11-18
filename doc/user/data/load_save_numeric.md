## Numeric data

Numeric data or general numeric matrices can be loaded or saved with the
following functions.

 - `data::Load(filename, matrix, opts)`
 - `data::Save(filename, matrix, opts)`

Specifying options using `opts` object can be done as follows:

 - `opts.Fatal() = false;`
 - `opts.Transpose() = true;`

   * `opts` can be the following:
       * Either a `DataOptions` object, that defines two options `Fatal()`. and
         `Format()`.
       * Or a `MatrixOptions` object, that defines `Transpose()` in addition to
         the previous options two options `Fatal()`. and `Format()`.
       * Or a `TextOptions` object, that defines `HasHeaders()`, `SemiColon()`
         `MissingToNan()` and `Categorical()`, in addition to `Fatal()`,
         `Format()` and `Transpose()`.

   * The behavior of the options (`Fatal()`, `Transpose()` etc..) is defined
     simirarly to `Fatal` and `Transpose` const options. For more information, please refer to 
     [DataOptions](#data-options), [MatrixOptions](#matrix-options) and [TextOptions](#text-options).

Another simplified signature can be used by specifying const boolean options
directly and add them using the `+` operator.

 - `data::Load(filename, matrix, NoFatal + Transpose + Autodetect)`
 - `data::Save(filename, matrix, NoFatal + Transpose + Autodetect)`

   * `filename` is a `std::string` with a path to the file to be loaded.

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

   * `Autodetect` a const boolean that automaticaly detects the file type based
     on extension. The type can be explicitly specifed;
     see [DataOptions](#data-options).

   * A `bool` is returned indicating whether the operation was successful.

   * A full list of these options is available: [DataOptions](#data-options),
     [MatrixOptions](#matrix-options) and [TextOptions](#text-options). 

---

Example usage:

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::data::Load("satellite.train.csv", dataset, Fatal + Transpose);

// See https://datasets.mlpack.org/satellite.train.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("satellite.train.labels.csv", labels, Fatal + Transpose);

// Print information about the data.
std::cout << "The data in 'satellite.train.csv' has: " << std::endl;
std::cout << " - " << dataset.n_cols << " points." << std::endl;
std::cout << " - " << dataset.n_rows << " dimensions." << std::endl;

std::cout << "The labels in 'satellite.train.labels.csv' have: " << std::endl;
std::cout << " - " << labels.n_elem << " labels." << std::endl;
std::cout << " - A maximum label of " << labels.max() << "." << std::endl;
std::cout << " - A minimum label of " << labels.min() << "." << std::endl;

// Modify and save the data.  Add 2 to the data and drop the last column.
dataset += 2;
dataset.shed_col(dataset.n_cols - 1);
labels.shed_col(labels.n_cols - 1);

mlpack::data::Save("satellite.train.mod.csv", dataset, NoFatal + Transpose);
mlpack::data::Save("satellite.train.labels.mod.csv", labels, NoFatal +
    Transpose);
```

Example usage if the dataset has a semicolon separator: 

```c++
std::fstream f;
f.open("semicolon.csv", fstream::out);
f << "1; 2; 3; 4" << std::endl;
f << "5; 6; 7; 8" << std::endl;
f << "9; 10; 11; 12" << std::endl;

arma::mat dataset;
data::TextOptions opts;
opts.Fatal() = false;
opts.NoTranspose() = false;
opts.Semicolon() = true;

data::Load("semicolon.csv", dataset, NoFatal + Transpose + Semicolon);
std::cout << "The data in 'missing_to_nan.csv' has: " << std::endl;
std::cout << " - " << dataset.n_cols << " points." << std::endl;
std::cout << " - " << dataset.n_rows << " dimensions." << std::endl;
```

Example usage if the dataset has a missing elements, these would be replaced
with NaN:

```c++
std::fstream f;
f.open("missing_to_nan.csv", fstream::out);
// Missing 2 value in the first row.
f << "1, , 3, 4" << std::endl;
f << "5, 6, 7, 8" << std::endl;
f << "9, 10, 11, 12" << std::endl;

arma::mat dataset;
data::TextOptions opts;
opts.Fatal() = false;
opts.NoTranspose() = true;
opts.MissingToNan() = true;

// Also we can write instead of opts, NoFatal + NoTranspose + MIssingToNan.
data::Load("missing_to_nan.csv", dataset, opts);
// Print information about the data.
std::cout << "The data in 'missing_to_nan.csv' has: " << std::endl;
std::cout << " - " << dataset.n_cols << " points." << std::endl;
std::cout << " - " << dataset.n_rows << " dimensions." << std::endl;
std::cout << "Is the value replaced with Nan: "<< std::isnan(dataset.at(0, 1))
    << std::endl;
```

Example usage if the dataset has a header and a semicolon separator.

```c++
fstream f;
f.open("header_semicolon.csv", fstream::out);
f << "a;b;c;d" << std::endl;
f << "1;2;3;4" << std::endl;
f << "5;6;7;8" << std::endl;

arma::mat dataset;
// We define opts so we can access the values of the header.
data::TextOptions opts;
opts.Fatal() = true;
opts.NoTranspose() = false;
opts.Semicolon() = true;
opts.HasHeaders() = true;

data::Load("header_semicolon.csv", dataset, opts);

// The headers are returned as an Armadillo filed.
arma::field<std::string> headers = opts.Headers();

std::cout << "The data in 'header_semicolon.csv' has: " << std::endl;
std::cout << " - " << dataset.n_cols << " points." << std::endl;
std::cout << " - " << dataset.n_rows << " dimensions." << std::endl;

std::cout << "header values are: " << headers.at(0) << "," << headers.at(1)
    << "," << headers.at(2) << "," << headers.at(3) << std::endl;
```
---

