<object data="../img/pipeline-top-1.svg" type="image/svg+xml" id="pipeline-top">
</object>

# Data loading and I/O

mlpack provides the `data::Load()` and `data::Save()` functions to load and save
[Armadillo matrices](matrices.md) (e.g. numeric and categorical datasets) and
any mlpack object via the [cereal](https://uscilab.github.io/cereal/)
serialization toolkit.  A number of other utilities related to loading and
saving data and objects are also available.

 * [Numeric data](#numeric-data)
 * [Mixed categorical data](#mixed-categorical-data)
   - [`data::DatasetInfo`](#datadatasetinfo)
   - [Loading categorical data](#loading-categorical-data)
 * [Data Options](#data-option)

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

### Resize images

It is possible to resize images in mlpack with the following function:

- `ResizeImages(images, info, newWidth, newHeight)`
   * `images` is a [column-major matrix](matrices.md) containing a set of
      images; each image is represented as a flattened vector in one column.

   * `info` is a [`data::ImageInfo&`](#dataimageinfo) containing details about
     the images in `images`, and will be modified to contain the new size of the
     images.

   * `newWidth` and `newHeight` (of type `size_t`) are the desired new
     dimensions of the resized images.

   * This function returns `void` and modifies `info` and `images`.

   * ***NOTE:*** if the element type of `images` is not `unsigned char` or
     `float` (e.g. if `image` is not `arma::Mat<unsigned char>` or
     `arma::fmat`), the matrix will be temporarily converted during resizing;
     therefore, using `unsigned char` or `float` as the element type is the most
     efficient.

   * This function expects all the images to have identical
     dimensions. If this is not the case, iteratively call `ResizeImages()` with
     a single image/column in `images`.
    
Example usage of the `ResizeImages()` function on a set of images with
different dimensions:

```c++
// See https://datasets.mlpack.org/sheep.tar.bz2
arma::Mat<unsigned char> image;
mlpack::data::ImageInfo info;

// The images are located in our test/data directory. However, any image could
// be used instead.
std::vector<std::string> files =
    {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
     "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
     "sheep_9.jpg"};

// The resized images will be saved locally. We are declaring the vector that
// contains the names of the resized images.
std::vector<std::string> reSheeps =
    {"re_sheep_1.jpg", "re_sheep_2.jpg", "re_sheep_3.jpg", "re_sheep_4.jpg",
     "re_sheep_5.jpg", "re_sheep_6.jpg", "re_sheep_7.jpg", "re_sheep_8.jpg",
     "re_sheep_9.jpg"};

// Load and Resize each one of them individually, because they do not have
// the same dimensions. The `info` will contain the dimension for each one.
for (size_t i = 0; i < files.size(); i++)
{
  mlpack::data::Load(files.at(i), image, info, false);
  mlpack::data::ResizeImages(image, info, 320, 320);
  mlpack::data::Save(reSheeps.at(i), image, info, false);
}
```

Example usage of `ResizeImages()` function on a set of images that have the
same dimensions.

```c++
// All images have the same dimension, It would be possible to load all of
// them into one matrix

// See https://datasets.mlpack.org/sheep.tar.bz2
arma::Mat<unsigned char> images;
mlpack::data::ImageInfo info;

std::vector<std::string> reSheeps =
    {"re_sheep_1.jpg", "re_sheep_2.jpg", "re_sheep_3.jpg", "re_sheep_4.jpg",
     "re_sheep_5.jpg", "re_sheep_6.jpg", "re_sheep_7.jpg", "re_sheep_8.jpg",
     "re_sheep_9.jpg"};

mlpack::data::Load(reSheeps, images, info, false);

// Now let us resize all these images at once, to specific dimensions.
mlpack::data::ResizeImages(images, info, 160, 160);

// The resized images will be saved locally. We are declaring the vector that
// contains the names of the resized images.
std::vector<std::string> smSheeps =
    {"sm_sheep_1.jpg", "sm_sheep_2.jpg", "sm_sheep_3.jpg", "sm_sheep_4.jpg",
     "sm_sheep_5.jpg", "sm_sheep_6.jpg", "sm_sheep_7.jpg", "sm_sheep_8.jpg",
     "sm_sheep_9.jpg"};

mlpack::data::Save(smSheeps, images, info, false);
```

### Resize and crop images

In addition to resizing images, mlpack also provides resize-and-crop
functionality.  This is useful when the desired aspect ratio of an image differs
largely from the original image.

The resize-and-crop operation, given a target size `outputWidth` x
`outputHeight`, first resizes the image while preserving the aspect ratio such
that the width and height of the image both no smaller than `outputWidth` and
`outputHeight`.  Then, the image is cropped to have size `outputWidth` by
`outputHeight`, keeping the center pixels only.  This process is shown below.

*Original image:*

<p align="center">
  <img src="../img/cat.jpg" alt="cat">
</p>

*Original image with target size of* `220`x`220` *pixels:*

<p align="center">
  <img src="../img/cat_rect.jpg" alt="cat with rectangle overlaid">
</p>

*First step: resize while preserving aspect ratio:*

<p align="center">
  <img src="../img/cat_scaled_rect.jpg"
       alt="scaled cat with rectangle overlaid">
</p>

*Second step: crop to desired final size:*

<p align="center">
  <img src="../img/cat_cropped.jpg" alt="cropped cat">
</p>

- `ResizeCropImages(images, info, newWidth, newHeight)`
   * `images` is a [column-major matrix](matrices.md) containing a set of
      images; each image is represented as a flattened vector in one column.

   * `info` is a [`data::ImageInfo&`](#dataimageinfo) containing details about
     the images in `images`.

   * `images` and `info` are modified in-place.

   * `newWidth` and `newHeight` (of type `size_t`) are the desired new
     dimensions of the resized images.
     - If the output size is larger than the input image size, the images will
       be upscaled the minimum amount necessary before cropping.
     - If the aspect ratio is not changed from the input aspect ratio, no
       cropping is performed.

   * ***NOTE:*** if the element type of `images` is not `unsigned char` or
     `float` (e.g. if `image` is not `arma::Mat<unsigned char>` or
     `arma::fmat`), the matrix will be temporarily converted during resizing;
     therefore, using `unsigned char` or `float` as the element type is the most
     efficient.

   * This function expects all the images to have identical dimensions. If this
     is not the case, iteratively call `ResizeCropImages()` with a single
     image/column in `images`.

Example usage of the `ResizeCropImages()` function on a set of images with
different dimensions:

```c++
// See https://datasets.mlpack.org/sheep.tar.bz2.
arma::Mat<unsigned char> image;
mlpack::data::ImageInfo info;

// The images are located in our test/data directory. However, any image could
// be used instead.
std::vector<std::string> files =
    {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
     "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
     "sheep_9.jpg"};

// The resized images will be saved locally. We are declaring the vector that
// contains the names of the resized and cropped images.
std::vector<std::string> cropSheeps =
    {"crop_sheep_1.jpg", "crop_sheep_2.jpg", "crop_sheep_3.jpg",
     "crop_sheep_4.jpg", "crop_sheep_5.jpg", "crop_sheep_6.jpg",
     "crop_sheep_7.jpg", "crop_sheep_8.jpg", "crop_sheep_9.jpg"};

// Load and resize-and-crop each image individually, because they do not have
// the same dimensions. The `info` will contain the dimension for each one.
for (size_t i = 0; i < files.size(); i++)
{
  mlpack::data::Load(files.at(i), image, info, false);
  mlpack::data::ResizeCropImages(image, info, 320, 320);
  mlpack::data::Save(cropSheeps.at(i), image, info, false);
  std::cout << "Resized and cropped " << files.at(i) << " to "
      << cropSheeps.at(i) << " with output size 320x320." << std::endl;
}
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

### Resize images

It is possible to resize images in mlpack with the following function:

- `ResizeImages(images, info, newWidth, newHeight)`
   * `images` is a [column-major matrix](matrices.md) containing a set of
      images; each image is represented as a flattened vector in one column.

   * `info` is a [`data::ImageInfo&`](#dataimageinfo) containing details about
     the images in `images`, and will be modified to contain the new size of the
     images.

   * `newWidth` and `newHeight` (of type `size_t`) are the desired new
     dimensions of the resized images.

   * This function returns `void` and modifies `info` and `images`.

   * ***NOTE:*** if the element type of `images` is not `unsigned char` or
     `float` (e.g. if `image` is not `arma::Mat<unsigned char>` or
     `arma::fmat`), the matrix will be temporarily converted during resizing;
     therefore, using `unsigned char` or `float` as the element type is the most
     efficient.

   * This function expects all the images to have identical
     dimensions. If this is not the case, iteratively call `ResizeImages()` with
     a single image/column in `images`.
    
Example usage of the `ResizeImages()` function on a set of images with
different dimensions:

```c++
// See https://datasets.mlpack.org/sheep.tar.bz2
arma::Mat<unsigned char> image;
mlpack::data::ImageInfo info;

// The images are located in our test/data directory. However, any image could
// be used instead.
std::vector<std::string> files =
    {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
     "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
     "sheep_9.jpg"};

// The resized images will be saved locally. We are declaring the vector that
// contains the names of the resized images.
std::vector<std::string> reSheeps =
    {"re_sheep_1.jpg", "re_sheep_2.jpg", "re_sheep_3.jpg", "re_sheep_4.jpg",
     "re_sheep_5.jpg", "re_sheep_6.jpg", "re_sheep_7.jpg", "re_sheep_8.jpg",
     "re_sheep_9.jpg"};

// Load and Resize each one of them individually, because they do not have
// the same dimensions. The `info` will contain the dimension for each one.
for (size_t i = 0; i < files.size(); i++)
{
  mlpack::data::Load(files.at(i), image, info, false);
  mlpack::data::ResizeImages(image, info, 320, 320);
  mlpack::data::Save(reSheeps.at(i), image, info, false);
}
```

Example usage of `ResizeImages()` function on a set of images that have the
same dimensions.

```c++
// All images have the same dimension, It would be possible to load all of
// them into one matrix

// See https://datasets.mlpack.org/sheep.tar.bz2
arma::Mat<unsigned char> images;
mlpack::data::ImageInfo info;

std::vector<std::string> reSheeps =
    {"re_sheep_1.jpg", "re_sheep_2.jpg", "re_sheep_3.jpg", "re_sheep_4.jpg",
     "re_sheep_5.jpg", "re_sheep_6.jpg", "re_sheep_7.jpg", "re_sheep_8.jpg",
     "re_sheep_9.jpg"};

mlpack::data::Load(reSheeps, images, info, false);

// Now let us resize all these images at once, to specific dimensions.
mlpack::data::ResizeImages(images, info, 160, 160);

// The resized images will be saved locally. We are declaring the vector that
// contains the names of the resized images.
std::vector<std::string> smSheeps =
    {"sm_sheep_1.jpg", "sm_sheep_2.jpg", "sm_sheep_3.jpg", "sm_sheep_4.jpg",
     "sm_sheep_5.jpg", "sm_sheep_6.jpg", "sm_sheep_7.jpg", "sm_sheep_8.jpg",
     "sm_sheep_9.jpg"};

mlpack::data::Save(smSheeps, images, info, false);
```

### Resize and crop images

In addition to resizing images, mlpack also provides resize-and-crop
functionality.  This is useful when the desired aspect ratio of an image differs
largely from the original image.

The resize-and-crop operation, given a target size `outputWidth` x
`outputHeight`, first resizes the image while preserving the aspect ratio such
that the width and height of the image both no smaller than `outputWidth` and
`outputHeight`.  Then, the image is cropped to have size `outputWidth` by
`outputHeight`, keeping the center pixels only.  This process is shown below.

*Original image:*

<p align="center">
  <img src="../img/cat.jpg" alt="cat">
</p>

*Original image with target size of* `220`x`220` *pixels:*

<p align="center">
  <img src="../img/cat_rect.jpg" alt="cat with rectangle overlaid">
</p>

*First step: resize while preserving aspect ratio:*

<p align="center">
  <img src="../img/cat_scaled_rect.jpg"
       alt="scaled cat with rectangle overlaid">
</p>

*Second step: crop to desired final size:*

<p align="center">
  <img src="../img/cat_cropped.jpg" alt="cropped cat">
</p>

- `ResizeCropImages(images, info, newWidth, newHeight)`
   * `images` is a [column-major matrix](matrices.md) containing a set of
      images; each image is represented as a flattened vector in one column.

   * `info` is a [`data::ImageInfo&`](#dataimageinfo) containing details about
     the images in `images`.

   * `images` and `info` are modified in-place.

   * `newWidth` and `newHeight` (of type `size_t`) are the desired new
     dimensions of the resized images.
     - If the output size is larger than the input image size, the images will
       be upscaled the minimum amount necessary before cropping.
     - If the aspect ratio is not changed from the input aspect ratio, no
       cropping is performed.

   * ***NOTE:*** if the element type of `images` is not `unsigned char` or
     `float` (e.g. if `image` is not `arma::Mat<unsigned char>` or
     `arma::fmat`), the matrix will be temporarily converted during resizing;
     therefore, using `unsigned char` or `float` as the element type is the most
     efficient.

   * This function expects all the images to have identical dimensions. If this
     is not the case, iteratively call `ResizeCropImages()` with a single
     image/column in `images`.

Example usage of the `ResizeCropImages()` function on a set of images with
different dimensions:

```c++
// See https://datasets.mlpack.org/sheep.tar.bz2.
arma::Mat<unsigned char> image;
mlpack::data::ImageInfo info;

// The images are located in our test/data directory. However, any image could
// be used instead.
std::vector<std::string> files =
    {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
     "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
     "sheep_9.jpg"};

// The resized images will be saved locally. We are declaring the vector that
// contains the names of the resized and cropped images.
std::vector<std::string> cropSheeps =
    {"crop_sheep_1.jpg", "crop_sheep_2.jpg", "crop_sheep_3.jpg",
     "crop_sheep_4.jpg", "crop_sheep_5.jpg", "crop_sheep_6.jpg",
     "crop_sheep_7.jpg", "crop_sheep_8.jpg", "crop_sheep_9.jpg"};

// Load and resize-and-crop each image individually, because they do not have
// the same dimensions. The `info` will contain the dimension for each one.
for (size_t i = 0; i < files.size(); i++)
{
  mlpack::data::Load(files.at(i), image, info, false);
  mlpack::data::ResizeCropImages(image, info, 320, 320);
  mlpack::data::Save(cropSheeps.at(i), image, info, false);
  std::cout << "Resized and cropped " << files.at(i) << " to "
      << cropSheeps.at(i) << " with output size 320x320." << std::endl;
}
```

### Changing the memory layout of images

When loading images using `data::Load()` channels are interleaved, i.e.
the underlying vector contains the values `[r, g, b, r, g, b, ... ]`
(for an image with 3 channels). mlpack has functionality such as `Convolution`
that requires channels be grouped, e.g `[r, r, ..., g, g, ..., b, b]`.
The same is true when using `data::Save()`, the channels are expected to be
interleaved.

To convert the layout of your image from interleaved channels to grouped
channels and vice versa, you can use `data::GroupChannels()` and
`data::InterleaveChannels()`.

***NOTE***: Other image related functions (such as
[`ResizeImages`](#resize-images) etc) require channels be interleaved. If you
need to use `GroupChannels()` make sure to resize or crop your images first
beforehand.

---

#### `data::GroupChannels()`

 * `data::GroupChannels(images, info)`
    - `images` must be a matrix where each column is an image. Each image is
      expected to be interleaved, i.e. in the format `[r, g, b, r, g, b ... ]`.

    - `info` describes the shape of each image.

    - Returns a matrix where each image from `images` are in the
      format `[r, r, ... , g, g, ... , b, b]`.

---

#### `data::InterleaveChannels()`

 * `data::InterleaveChannels(images, info)`
    - Performs the reverse of `data::GroupChannels()`.

    - `images` must be a matrix where each column is an image. Each image is
      expected to be grouped, i.e. in the format `[r, r, ..., g, g, ..., b, b]`.

    - `info` describes the shape of each image.

    - Returns a matrix where each image from `images` are in the
      format `[r, g, b, r, g, b ... ]`.

---

#### Example

An example that loads an image converts the layout such that channels are
grouped together in preparation for a convolutional neural network. Then convert
back to interleaved channels and save the image.

```c++
// Download: https://datasets.mlpack.org/images/mlpack-favicon.png
arma::mat image;
mlpack::data::ImageInfo info;
mlpack::data::Load("mlpack-favicon.png", image, info, true);

std::vector<std::string> colors =
     { "\033[31m", "\033[32m", "\033[34m", "\033[37m" };

// Display input before grouping channels (Load returns channels interleaved).
std::cout << "Original Image (channels interleaved):" << std::endl;
for (size_t i = 0; i < image.n_rows; i += info.Channels())
{
  for (size_t j = 0; j < info.Channels(); j++)
    std::cout << colors[j] << image.at(i + j, 0) << "\033[0m" << ", ";
}
std::cout << std::endl << std::endl;

// Group channels.
image = mlpack::data::GroupChannels(image, info);

// Display submatrix of input after grouping channels
std::cout << "Grouped channels:" << std::endl;
for (size_t i = 0; i < info.Channels(); i++)
{
  for (size_t j = 0; j < image.n_rows / info.Channels(); j++)
    std::cout << colors[i] <<
      image.at(i * image.n_rows / info.Channels() + j, 0) << "\033[0m" << ", ";
}
std::cout << std::endl << std::endl;

// Do some computation here, for example a convolutional neural network.

// Interleave channels to prepare for saving.
image = mlpack::data::InterleaveChannels(image, info);

// Display input after interleaving channels
// Should be identical to original.
std::cout << "Interleaved channels (indentical to original):" << std::endl;
for (size_t i = 0; i < image.n_rows; i += info.Channels())
{
  for (size_t j = 0; j < info.Channels(); j++)
    std::cout << colors[j] << image.at(i + j, 0) << "\033[0m" << ", ";
}
std::cout << std::endl << std::endl;

mlpack::data::Save("mlpack-favicon.png", image, info, true);
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


## Matrix Options

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

## Text Options

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
