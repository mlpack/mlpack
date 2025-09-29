<object data="../img/pipeline-top-1.svg" type="image/svg+xml" id="pipeline-top">
</object>

# Data loading and I/O

mlpack provides the `data::Load()` and `data::Save()` functions to load and save
[Armadillo matrices](matrices.md) (e.g. numeric and categorical datasets) and
any mlpack object via the [cereal](https://uscilab.github.io/cereal/)
serialization toolkit.  A number of other utilities related to loading and
saving data and objects are also available.

 * [Data Options](#data-option)
 * [Numeric data](#numeric-data)
 * [Mixed categorical data](#mixed-categorical-data)
   - [`data::DatasetInfo`](#datadatasetinfo)
   - [Loading categorical data](#loading-categorical-data)
 * [Image data](#image-data)
   - [`data::ImageInfo`](#dataimageinfo)
   - [Loading images](#loading-images)
 * [mlpack objects](#mlpack-objects): load or save any mlpack object
 * [Formats](#formats): supported formats for each load/save variant

## Data Options
It is a generic class that allows the user to specify the
`data::Load()` and `data::Save()` options when loading and saving dataset
files. mlpack has an identical data load API in whether we
are trying to load an image or a csv file. However, we need to specify the
relevant options for each case. Currently mlpack supports the following:

1. `data::DataOptions`: provide settings if load is fatal or not, and allow
   specify the Format of the File we are trying to load.
2. `data::MatrixOptions` provide settings to transpose matrix when loading /
   saving. Inherits directly `DataOptions`
2. `data::TextOptions`: provide settings related to time series data. Inherits
   directly `MatrixOptions`
3. `data::ImageOptions`: provide setttings when loading images. (e,g., Height,
   Width, etc). Inherits directly `DataOptions`

The settings related to these classes are simplfied to easy interaction. This
is done by using the `+` operator between these settings when loading /
saving. More details are provided below with examples how to use
`data::Load / data::Save` functions.


## DataOptions

|-------------------------------------------------------------------------------------------------------|
| Operator | Function | Type  | Comment | 
|-------------------------------------------------------------------------------------------------------|
| `Fatal`  | `.Fatal() = true` | bool | A  `std::runtime_error` will be thrown on failure.              |
|-------------------------------------------------------------------------------------------------------|
| `NoFatal`  | `.Fatal() = false` | bool | A false will be returned on failure.A warning will be printed if the user enabled `MLPACK_PRINT_WARN`|
|-------------------------------------------------------------------------------------------------------|
| `CSV` | `.Format() = FileType::CSVASCII` | enum | (autodetect extensions `.csv`, `.tsv`): CSV format  |
|       |                                  |      | with no header.  If loading a sparse matrix and the |
|       |                                  |      | CSV has three columns, the data is interpreted as a |
|       |                                  |      | [coordinate list](https://arma.sourceforge.net/docs.html#save_load_mat). |
|-----------------------------------------------------------------------------------------------------|
| `PGM` | `.Format() = FileType::PGMBinary` | enum | (autodetect extension `.pgm`): PGM image format. |
|-----------------------------------------------------------------------------------------------------|
| `PPM` | `.Format() = FileType::PPMBinary` | enum | (autodetect extension `.ppm`):PPM image format.  |
|-----------------------------------------------------------------------------------------------------|
| `HDF5` | `.Format() = FileType::HDF5Binary` | enum | (autodetect extensions `.h5`, `.hdf5`, `.hdf`,
|        |                                    |      |  `.he5`): [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) |
|        |                                    |      |   binary format; only available if Armadillo is configured with |
|        |                                    |      | [HDF5 support](https://arma.sourceforge.net/docs.html#config_hpp). |
|--------------------------------------------------------------------------------------------------------------|
| `ArmaAscii` | `.Format() = FileType::ArmaASCII` | enum | (autodetect extension `.txt`): space-separated |
|             |                                   |      | values as saved by Armadillo with the          |
|             |                                   |      | [`arma_ascii`](https://arma.sourceforge.net/docs.html#save_load_mat) |
|             |                                   |      | format. |
|--------------------------------------------------------------------------------------------------------------|
| `ArmaBin` | `.Format() = FileType::ArmaBinary` | enum |  (autodetect extension `.bin`): Armadillo's     |
|           |                                    |      |  efficient binary matrix format                 |
|           |                                    |      |  ([`arma_binary`](https://arma.sourceforge.net/docs.html#save_load_mat)). |
|--------------------------------------------------------------------------------------------------------------|
| `RawAscii` | `.Format() = FileType::RawASCII` | enum | (autodetect extensions `.csv`, `.txt`):          |
|            |                                  |      | space-separated values or tab-separated values (TSV) |
|            |                                  |      | with no header.                                  |
|--------------------------------------------------------------------------------------------------------------------|
| `BinAscii` | `.Format() = FileType::RawBinary` | enum | (autodetect extension `.bin`): packed binary data     | 
|            |                                   |      | with no header and no size information; data will be  |
|            |                                   |      | loaded as a single column vector _(not recommended)_. |
|--------------------------------------------------------------------------------------------------------------------|
| `CoordAscii` | `.Format() = FileType::` | enum | (autodetect extensions `.txt`, `.tsv`; must be loading a sparse |
|              |                          |      | matrix type): coordinate list format for sparse data (see       |
|              |                          |      | [`coord_ascii`](https://arma.sourceforge.net/docs.html#save_load_mat)). |
|------------------------------------------------------------------------------------------------------------|
| `AutoDetect` | `.Format() = FileType::AutoDetect` | enum | auto-detects the format as one of the  |
|              |                                    |      | formats above using the extension of the |
|              |                                    |      | filename and inspecting the file contents.|
|----------------------------------------------------------------------------------------------------------------|

***Notes:***

   - ASCII formats (`CSVAscii`, `RawAscii`, `ArmaAscii`) are human-readable but
     large; to reduce dataset size, consider a binary format such as
      `ArmaBinary` or `HDF5`.
   - Sparse data (`arma::sp_mat`, `arma::sp_fmat`, etc.) must be saved in a
     binary format (`ArmaBinary` or `HDF5`) or as a coordinate list
     (`CoordAscii`).

### [mlpack objects](#mlpack-objects)

By default, load/save format for mlpack objects is autodetected. However, 
if necessary you can specify the format of the serialization of the objects (ml
models). Currently mlpack supports three serialization formats: Binary, JSON,
and XML. Those can be specified as follows:

|-------------------------------------------------------------------------------|
| operator |  Function | Type  | Comment                                        |
|-------------------------------------------------------------------------------|
| `JSON` | `.Format() = FileType::JSON` | enum | (autodetect extension `.json`) |
|-------------------------------------------------------------------------------|
| `XML` | `.Format() = FileType::XML`   | enum | (autodetect extension `.xml`)  |
|-------------------------------------------------------------------------------|
| `BIN` | `.Format() = FileType::BIN`   | enum | (autodetect extension `.bin`)  |
|-------------------------------------------------------------------------------|

***Notes:***

 - `FileType::JSON` (`.json`) and `FileType::XML` (`.xml`) produce human-readable
   files, but they may be quite large.
 - `FileType::BIN` (`.bin`) is recommended for the sake of size; objects in
   binary FileType may be an order of magnitude or more smaller than JSON!

### [images](#images)

By default, loading and saving images is auto detected. If the user does not
want to specify the type, they can indicate that this an image by pasing
`Image` in the data option field. However, it is possible
to specify the file format we are trying to load / save. mlpack supports the
following formats only:

|-------------------------------------------------------------------------------|
| operator |  Function | Type  | Comment                                        |
|-------------------------------------------------------------------------------|
| `PNG` | `.Format() = FileType::PNG`   | enum | (autodetect extension `.png`)  |
|-------------------------------------------------------------------------------|
| `JPG` | `.Format() = FileType::JPG`   | enum | (autodetect extension `.jpg`)  |
|-------------------------------------------------------------------------------|
| `TGA` | `.format() = filetype::TGA`   | enum | (autodetect extension `.tga`)  |
|-------------------------------------------------------------------------------|
| `BMP` | `.format() = filetype::BMP`   | enum | (autodetect extension `.bmp`)  |
|-------------------------------------------------------------------------------|
| `gif` | `.format() = filetype::GIF`   | enum | (autodetect extension `.gif`)  |
|-------------------------------------------------------------------------------|
| `pic` | `.format() = filetype::PIC`   | enum | (autodetect extension `.pic`)  |
|-------------------------------------------------------------------------------|
| `pnm` | `.format() = filetype::PNM`   | enum | (autodetect extension `.pnm`)  |
|-------------------------------------------------------------------------------|
| `Image` | `.format() = filetype::ImageType` | enum | Not specifying the type  |
|-------------------------------------------------------------------------------|


## MatrixOptions

During standard load / save operation we usually would like to Load the matrix
in column major (transposed). For this, MatrixOptions is going to be used when
we are Transpose operator as follows:

During standard load /save operations for plaintext formats (CSV/TSV/ASCII),
the following options allows the matrix to be transposed on load or save.
(Keep this `true` if you want | a column-major matrix to be loaded or saved
with points as rows and | dimensions as columns; that is generally what is desired.) 

|-------------------------------------------------------------------------------------------------------|
| Operator | Function | Type  | Comment | 
|-------------------------------------------------------------------------------------------------------|
| `Transpose`  | `.transpose() = true` | bool | The matrix will be transposed when load / save.         |
|-------------------------------------------------------------------------------------------------------|
| `NoTranspose`  | `.transpose() = false` | bool | The matrix will not be transposed when load / save.  |
|-------------------------------------------------------------------------------------------------------|

## TextOptions

This class allows to specify settings related to the characteristic of the
matrix we are loading. For instance, does it contain categorical values? or
does the dataset has headers. The supported options are as following:

|-------------------------------------------------------------------------------------------------------|
| Operator | Function | Type  | Comment | 
|-------------------------------------------------------------------------------------------------------|
| `HasHeaders`   | `.HasHeaders()`  | bool | Set `true`, if the CSV file has a header, the header can   |
|                |                  |      | be accessible using `Headers()` function                   |
|--------------------------------------------------------------------------------------------------------
| `SemiColon`    | `.SemiColon()`   | bool | Set `true` for plaintext formats (CSV/TSV/ASCII) if the    |
|                |                  |      | separator is a semicolon instead of a comma                |
|--------------------------------------------------------------------------------------------------------
| `MissingToNan` | `.MissingToNan()`| bool | Set `true`, if there is missing data elements and you want |
|                |                  |      | them to be replaced with NaN                               |
|--------------------------------------------------------------------------------------------------------
| `Categorical`  | `.Categorical()` | bool | Set `true`, if the dataset contains categorical data.      |
|--------------------------------------------------------------------------------------------------------


## Numeric data

Numeric data or general numeric matrices can be loaded or saved with the
following functions.

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
     see [DataOptions](#DataOptions).

Another signature can be used by specifying the options before calling the
function. For instance

 - `opts.Fatal() = false;`
 - `opts.Transpose() = true;`

Then call the load function as follows:

 - `data::Load(filename, matrix, opts)`
 - `data::Save(filename, matrix, opts)`

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
     [DataOptions](#DataOptions), [MatrixOptions](#MatrixOptions) and [TextOptions](#TextOptions).
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

## Mixed categorical data

Some mlpack techniques support mixed categorical data, e.g., data where some
dimensions take only categorical values (e.g. `0`, `1`, `2`, etc.).  When using
mlpack, string data and other non-numerical data must be mapped to categorical
values and represented as part of an `arma::mat`.  Category information is
stored in an auxiliary `data::DatasetInfo` object.

### `data::DatasetInfo`

<!-- TODO: also document in core.md? -->

mlpack represents categorical data via the use of the auxiliary
`data::DatasetInfo` object, which stores information about which dimensions are
numeric or categorical and allows conversion from the original category values
to the numeric values used to represent those categories.

---

#### Constructors

 - `info = data::DatasetInfo()`
   * Create an empty `data::DatasetInfo` object.
   * Use this constructor if you intend to populate the `data::DatasetInfo` via
     a `data::Load()` call.

 - `info = data::DatasetInfo(dimensionality)`
   * Create a `data::DatasetInfo` object with the given dimensionality
   * All dimensions are assumed to be numeric (not categorical).

---

#### Accessing and setting properties

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

With a `data::DatasetInfo` object, categorical data can be loaded:

 - `data::Load(filename, matrix, info, fatal=false, transpose=true)`
   * `filename` is a `std::string` with a path to the file to be loaded.

   * The format is auto-detected based on the extension of the filename and the
     contents of the file:
     - `.csv`, `.tsv`, or `.txt` for CSV/TSV (tab-separated)/ASCII
       (space-separated)
     - `.arff` for [ARFF](https://ml.cms.waikato.ac.nz/weka/arff.html)

   * `matrix` is an `arma::mat&`, `arma::Mat<size_t>&`, or similar (e.g., a
     reference to an Armadillo object that data will be loaded into or saved
     from).

   * `info` is a `data::DatasetInfo&` object.  This will be populated with the
     category information of the file when loading, and used to unmap values
     when saving.

   * If `fatal` is `true`, a `std::runtime_error` will be thrown on failure.

   * If `transpose` is `true`, then for plaintext formats (CSV/TSV/ASCII), the
     matrix will be transposed on save.  (Keep this `true` if you want a
     column-major matrix to be saved with points as rows and dimensions as
     columns; that is generally what is desired.)

   * A `bool` is returned indicating whether the operation was successful.

Saving should be performed with the [numeric](#numeric-data) `data::Load()`
variant.

---

Example usage to load and manipulate an ARFF file.

```c++
// Load a categorical dataset.
arma::mat dataset;
mlpack::data::DataOptions opts;
opts.Fatal() = true;
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
for (size_t d = 0; d < info.Dimensionality(); ++d)
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

#### [Mixed categorical data](#mixed-categorical-data)

The format of mixed categorical data is detected automatically based on the
file extension and inspecting the file contents:

 - `.csv`, `.txt`, or `.tsv` indicates CSV/TSV/ASCII format
 - `.arff` indicates [ARFF](https://ml.cms.waikato.ac.nz/weka/arff.html)

