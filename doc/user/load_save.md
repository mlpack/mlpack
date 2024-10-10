# Loading and saving in mlpack

mlpack provides the `data::Load()` and `data::Save()` functions to load and save
[Armadillo matrices](matrices.md) (e.g. numeric and categorical datasets) and
any mlpack object via the [cereal](https://uscilab.github.io/cereal/)
serialization toolkit.  A number of other utilities related to loading and
saving data and objects are also available.

 * [Numeric data](#numeric-data)
 * [Mixed categorical data](#mixed-categorical-data)
   - [`data::DatasetInfo`](#datadatasetinfo)
   - [Loading categorical data](#loading-categorical-data)
 * [Image data](#image-data)
   - [`data::ImageInfo`](#dataimageinfo)
   - [Loading images](#loading-images)
 * [mlpack objects](#mlpack-objects): load or save any mlpack object
 * [Formats](#formats): supported formats for each load/save variant

## Numeric data

Numeric data or general numeric matrices can be loaded or saved with the
following functions.

 - `data::Load(filename, matrix, fatal=false, transpose=true, format=FileType::AutoDetect)`
 - `data::Save(filename, matrix, fatal=false, transpose=true, format=FileType::AutoDetect)`
   * `filename` is a `std::string` with a path to the file to be loaded.

   * By default the format is auto-detected based on the file extension, but can
     be explicitly specified with `format`; see [Formats](#formats).

   * `matrix` is an `arma::mat&`, `arma::Mat<size_t>&`, `arma::sp_mat&`, or
     similar (e.g., a reference to an Armadillo object that data will be loaded
     into or saved from).

   * If `fatal` is `true`, a `std::runtime_error` will be thrown on failure.

   * If `transpose` is `true`, then for plaintext formats (CSV/TSV/ASCII), the
     matrix will be transposed on load or save.  (Keep this `true` if you want a
     column-major matrix to be loaded or saved with points as rows and
     dimensions as columns; that is generally what is desired.)

   * A `bool` is returned indicating whether the operation was successful.

---

Example usage:

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::data::Load("satellite.train.csv", dataset, true);

// See https://datasets.mlpack.org/satellite.train.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("satellite.train.labels.csv", labels, true);

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

mlpack::data::Save("satellite.train.mod.csv", dataset);
mlpack::data::Save("satellite.train.labels.mod.csv", labels);
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
mlpack::data::DatasetInfo info;
// See https://datasets.mlpack.org/covertype.train.arff.
mlpack::data::Load("covertype.train.arff", dataset, info, true);

arma::Row<size_t> labels;
// See https://datasets.mlpack.org/covertype.train.labels.csv.
mlpack::data::Load("covertype.train.labels.csv", labels, true);

// Print information about the data.
std::cout << "The data in 'covertype.train.arff' has: " << std::endl;
std::cout << " - " << dataset.n_cols << " points." << std::endl;
std::cout << " - " << info.Dimensionality() << " dimensions." << std::endl;

// Print information about each dimension.
for (size_t d = 0; d < info.Dimensionality(); ++d)
{
  if (info.Type(d) == mlpack::data::Datatype::categorical)
  {
    std::cout << " - Dimension " << d << " is categorical with "
        << info.NumMappings(d) << " categories." << std::endl;
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
  if (info.Type(d) == mlpack::data::Datatype::categorical)
  {
    // This will create a new mapping if the string "hooray!" does not already
    // exist as a category for dimension d..
    dataset(d, 4) = info.MapString<double>("hooray!", d);
  }
  else
  {
    dataset(d, 4) += 1.0;
  }
}
```

---

Example usage to manually create a `data::DatasetInfo` object.

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

## Image data

If the STB image library is available on the system (`stb_image.h` and
`stb_image_write.h` must be available on the compiler's include search path),
then mlpack will define the `MLPACK_HAS_STB` macro, and support for loading
individual images or sets of images will be available.

Supported formats for loading are `jpg`, `png`, `tga`, `bmp`, `psd`, `gif`, `hdr`, `pic`, and `pnm`.

Supported formats for saving are `jpg`, `png`, `tga`, `bmp`, and `hdr`.

When loading images, each image is represented as a flattened single column
vector in a data matrix; each row of the resulting vector will correspond to a
single pixel value in a single channel.  An auxiliary `data::ImageInfo` class is
used to store information about the images.

### `data::ImageInfo`

The `data::ImageInfo` class contains the metadata of the images.

---

#### Constructors

 - `info = data::ImageInfo()`
   * Create a `data::ImageInfo` object with no data.
   * Use this constructor if you intend to populate the `data::ImageInfo` via a
     `data::Load()` call.

 - `info = data::ImageInfo(width, height, channels)`
   * Create a `data::ImageInfo` object with the given image specifications.
   * `width` and `height` are specified as pixels.

---

#### Accessing and modifying image metadata

 - `info.Quality() = q` will set the compression quality (e.g. for saving JPEGs)
   to `q`.
   * `q` should take values between `0` and `100`.
   * The quality value is ignored unless calling `data::Save()` with `info`.

 - Calling `info.Channels() = 1` before loading will cause images to be loaded
   in grayscale.

 - Metadata stored in the `data::ImageInfo` can be accessed with the following
   members:
   * `info.Width()` returns the image width in pixels.
   * `info.Height()` returns the image height in pixels.
   * `info.Channels()` returns the number of color channels in the image.
   * `info.Quality()` returns the compression quality that will be used to save
     images (between 0 and 100).

---

### Loading images

With a `data::ImageInfo` object, image data can be loaded or saved, handling
either one or multiple images at a time:

<!-- TODO: add parameter to force use of what's in `info` -->

 - `data::Load(filename, matrix, info, fatal=false)`
   * Load a ***single image*** from `filename` into `matrix`.
     - Format is chosen by extension (e.g. `image.png` will load as PNG).

   * `matrix` will have one column representing the image as a flattened vector.

   * `info` will be populated with information from the image in `filename`.

   * If `fatal` is `true`, a `std::runtime_error` will be thrown upon load
     failure.

   * Returns a `bool` indicating the success of the operation.

---

 - `data::Load(files, matrix, info, fatal=false)`
   * Load ***multiple images*** from `files` into `matrix`.
     - `files` is of type `std::vector<std::string>` and should contain the list
       of images to be loaded.
     - `matrix` will have `files.size()` columns, each representing the
       corresponding image as a flattened vector.

   * `info` will be populated with information from the images in `files`.

   * If `fatal` is `true`, a `std::runtime_error` will be thrown if any files
     fail to load.

   * Returns a `bool` indicating the success of the operation.

---

 - `data::Save(filename, matrix, info, fatal=false)`
   * Save a ***single image*** from `matrix` into the file `filename`.
     - Format is chosen by extension (e.g. `image.png` will save as PNG).

   * `matrix` is expected to have only one column representing the image as a
     flattened vector.

   * If `fatal` is `true`, a `std::runtime_error` will be thrown in the event of
     save failure.

   * Returns a `bool` indicating the success of the operation.

---

 - `data::Save(files, matrix, info, fatal=false)`
   * Save ***multiple images*** from `matrix` into `files`.
     - `files` is of type `std::vector<std::string>` and should contain the list
       of files to save to.
     - The format of each file is chosen by extension (e.g. `image.png` will
       save as PNG); it is allowed for filenames in `files` to have different
       extensions.

   * `matrix` is expected to have `files.size()` columns representing images as
     flattened vectors.

   * If `fatal` is `true`, a `std::runtime_error` will be thrown if any images
     fail to save.

   * Returns a `bool` indicating the success of the operation.

---

Images are flattened along rows, with channel values interleaved, starting from
the top left.  Thus, the value of the pixel at position `(x, y)` in channel `c`
will be contained in element/row `y * (width * channels) + x * (channels) + c`
of the flattened vector.

Pixels take values between 0 and 255.

---

Example of loading and saving a single image:

```c++
// See https://www.mlpack.org/static/img/numfocus-logo.png.
mlpack::data::ImageInfo info;
arma::mat matrix;
mlpack::data::Load("numfocus-logo.png", matrix, info, true);

// `matrix` should now contain one column.

// Print information about the image.
std::cout << "Information about the image in 'numfocus-logo.png': "
    << std::endl;
std::cout << " - " << info.Width() << " pixels in width." << std::endl;
std::cout << " - " << info.Height() << " pixels in height." << std::endl;
std::cout << " - " << info.Channels() << " color channels." << std::endl;

std::cout << "Value at pixel (x=3, y=4) in the first channel: ";
const size_t index = (4 * info.Width() * info.Channels()) +
    (3 * info.Channels());
std::cout << matrix[index] << "." << std::endl;

// Increment each pixel value, but make sure they are still within the bounds.
matrix += 1;
matrix = arma::clamp(matrix, 0, 255);

mlpack::data::Save("numfocus-logo-mod.png", matrix, info);
```

---

Example of loading and saving multiple images:

```c++
// Load some favicons from websites associated with mlpack.
std::vector<std::string> images;
// See the following files:
// - https://datasets.mlpack.org/images/mlpack-favicon.png
// - https://datasets.mlpack.org/images/ensmallen-favicon.png
// - https://datasets.mlpack.org/images/armadillo-favicon.png
// - https://datasets.mlpack.org/images/bandicoot-favicon.png
images.push_back("mlpack-favicon.png");
images.push_back("ensmallen-favicon.png");
images.push_back("armadillo-favicon.png");
images.push_back("bandicoot-favicon.png");

mlpack::data::ImageInfo info;
info.Channels() = 1; // Force loading in grayscale.

arma::mat matrix;
mlpack::data::Load(images, matrix, info, true);

// Print information about what we loaded.
std::cout << "Loaded " << matrix.n_cols << " images.  Images are of size "
    << info.Width() << " x " << info.Height() << " with " << info.Channels()
    << " color channel." << std::endl;

// Invert images.
matrix = (255.0 - matrix);

// Save as compressed JPEGs with low quality.
info.Quality() = 75;
std::vector<std::string> outImages;
outImages.push_back("mlpack-favicon-inv.jpeg");
outImages.push_back("ensmallen-favicon-inv.jpeg");
outImages.push_back("armadillo-favicon-inv.jpeg");
outImages.push_back("bandicoot-favicon-inv.jpeg");

mlpack::data::Save(outImages, matrix, info);
```

## mlpack objects

All mlpack objects can be saved with `data::Save()` and loaded with
`data::Load()`.  Serialization is performed using the
[cereal](https://uscilab.github.io/cereal/) serialization toolkit.
Each object must be given a logical name.

 - `data::Load(filename, name, object, fatal=false, format=data::format::autodetect)`
 - `data::Save(filename, name, object, fatal=false, format=data::format::autodetect)`
   * Load/save `object` to/from `filename` with the logical name `name`.

   * If `fatal` is `true`, a `std::runtime_error` will be thrown in the event of
     load or save failure.

   * The format is autodetected based on extension (`.bin`, `.json`, or `.xml`),
     but can be manually specified:
     - `data::format::binary`: binary blob (smallest and fastest).  No checks;
       assumes all data is correct.
     - `data::format::json`: JSON.
     - `data::format::xml`: XML (largest and slowest).

   * For JSON and XML types, when loading, `name` must match the name used to
     save the object.

   * Returns a `bool` indicating the success of the operation.

***Note:*** when loading an object that was saved as a binary blob, the C++ type
of the object must be ***exactly the same*** (including template parameters) as
the type used to save the object.  If not, undefined behavior will occur---most
likely a crash.

---

Simple example: create a `math::Range` object, then save and load it.

```c++
mlpack::math::Range r(3.0, 6.0);

// Save the Range to 'range.bin', using the name "range".
mlpack::data::Save("range.bin", "range", r, true);

// Load the range into a new object.
mlpack::math::Range r2;
mlpack::data::Load("range.bin", "range", r2, true);

std::cout << "Loaded range: [" << r2.Lo() << ", " << r2.Hi() << "]."
    << std::endl;

// Modify and save the range as JSON.
r2.Lo() = 4.0;
mlpack::data::Save("range.json", "range", r2, true);

// Now 'range.json' will contain the following:
//
// {
//     "range": {
//         "cereal_class_version": 0,
//         "hi": 6.0,
//         "lo": 4.0
//     }
// }
```

---

## Formats

mlpack's `data::Load()` and `data::Save()` functions support a variety of
different formats in different contexts.

---

#### [Numeric data](#numeric-data)

By default, load/save format is ***autodetected***, but can be manually
specified with the `format` parameter using one of the options below:

 - `FileType::AutoDetect` (default): auto-detects the format as one of the
   formats below using the extension of the filename and inspecting the file
   contents.

 - `FileType::CSVASCII` (autodetect extensions `.csv`, `.tsv`): CSV format
   with no header.  If loading a sparse matrix and the CSV has three columns,
   the data is interpreted as a
   [coordinate list](https://arma.sourceforge.net/docs.html#save_load_mat).

 - `FileType::RawASCII` (autodetect extensions `.csv`, `.txt`):
   space-separated values or tab-separated values (TSV) with no header.

 - `FileType::ArmaASCII` (autodetect extension `.txt`): space-separated
   values as saved by Armadillo with the
   [`arma_ascii`](https://arma.sourceforge.net/docs.html#save_load_mat)
   format.

 - `FileType::CoordASCII` (autodetect extensions `.txt`, `.tsv`; must be
   loading a sparse matrix type): coordinate list format for sparse data (see
   [`coord_ascii`](https://arma.sourceforge.net/docs.html#save_load_mat)).

 - `FileType::ArmaBinary` (autodetect extension `.bin`): Armadillo's
   efficient binary matrix format
   ([`arma_binary`](https://arma.sourceforge.net/docs.html#save_load_mat)).

 - `FileType::HDF5Binary` (autodetect extensions `.h5`, `.hdf5`, `.hdf`,
  `.he5`): [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)
   binary format; only available if Armadillo is configured with
   [HDF5 support](https://arma.sourceforge.net/docs.html#config_hpp).

 - `FileType::RawBinary` (autodetect extension `.bin`): packed binary data
   with no header and no size information; data will be loaded as a single
   column vector _(not recommended)_.

 - `FileType::PGMBinary` (autodetect extension `.pgm`): PGM image format

***Notes:***

   - ASCII formats (`CSVASCII`, `RawASCII`, `ArmaASCII`) are human-readable but
     large; to reduce dataset size, consider a binary format such as
      `ArmaBinary` or `HDF5Binary`.
   - Sparse data (`arma::sp_mat`, `arma::sp_fmat`, etc.) should be saved in a
     binary format (`ArmaBinary` or `HDF5Binary`) or as a coordinate list
     (`CoordASCII`).

---

#### [Mixed categorical data](#mixed-categorical-data)

The format of mixed categorical data is detected automatically based on the
file extension and inspecting the file contents:

 - `.csv`, `.txt`, or `.tsv` indicates CSV/TSV/ASCII format
 - `.arff` indicates [ARFF](https://ml.cms.waikato.ac.nz/weka/arff.html)

---

#### [Image data](#image-data)

The format of images are detected automatically based on the file extension.

 - The following formats are supported for loading: `.jpg`, `.jpeg`, `.png`,
   `.tga`, `.bmp`, `.psd`, `.gif`, `.hdr`, `.pic`, `.pnm`

 - The following formats are supported for saving: `.jpg`, `.png`, `.tga`,
   `.bmp`, `.hdr`

---

#### [mlpack objects](#mlpack-objects)

By default, load/save format for mlpack objects is autodetected, but can be
manually specified with the `format` parameter using one of the options below:

 - `format::autodetect` (default): auto-detects the format as one of the
    formats below using the extension of the filename
 - `format::json` (autodetect extension `.json`)
 - `format::xml` (autodetect extension `.xml`)
 - `format::binary` (autodetect extension `.bin`)

***Notes:***

 - `format::json` (`.json`) and `format::xml` (`.xml`) produce human-readable
   files, but they may be quite large.
 - `format::binary` (`.bin`) is recommended for the sake of size; objects in
   binary format may be an order of magnitude or more smaller than JSON!
