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

 - `data::Load(filename, X)`
   * Load `X` from the given file `filename` with default options:
     - the format of the file is auto-detected, and
     - an exception is *not* thrown on an error.
   * Returns a `bool` indicating whether the load was a success.
   * `X` can be [any supported load type](#load-types).
   
 - `data::Load(filename, object, Option1 + Option2 + ...)`
   * Load `X` from the given file `filename` with the given options.
   * Returns a `bool` indicating whether the load was a success.
   * `X` can be [any supported load type](#load-types).
   * The given options must be from the [list of standalone options](#data-options) and be appropriate for the type of `X`.

 
 - `data::Load(filename, object, opts)`
   * Load `X` from the given file `filename` with the given options specified in `opts`.
   * Returns a `bool` indicating whether the load was a success.
   * `X` can be [any supported load type](#load-types).
   * `opts` is a [`DataOptions` object](#data-options) whose subtype matches the type of `X`.

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
||||||| 672a5f7235
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
=======
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

### Letterbox transform

The letterbox transform resizes an image's dimensions to `width x height` but
keeps the aspect ratio of the original image. Whitespace is then filled in
with `fillValue`.

*Original image with size of* `640`x`326` *pixels:*

<p align="center">
  <img src="../img/cat.jpg" alt="cat">
</p>

*Image with target size of* `416`x`416` *pixels after letterbox:*

<p align="center">
  <img src="../img/cat_square_letterbox.jpg"
       alt="cat with square letterbox transform">
</p>

*Image with target size of* `300`x`208` *pixels after letterbox:*

<p align="center">
  <img src="../img/cat_rect_letterbox.jpg"
       alt="cat with rectangular letterbox transform">
</p>

- `LetterboxImages(src, opt, width, height, fillValue)`
  * `src` is a [column-major matrix](matrices.md) containing a single image,
    where the image is represented as a flattened vector in one column.
  * `opt` is a [`data::imageOptions&`](#dataimageinfo) containing info on
    the dimensions of the image.
  * `width` and `height` are `const size_t`s determining the new width and
    height of `src`.
  * `fillValue` is the white space value that pads out the resized image.
    Each channel will be filled in with this value, i.e., if `fillValue` is 127
    then each RGB channel will be 127.
  * Only images with 1 or 3 channels can be used.

#### Example

An example that loads an image, resizes the image to some square image
while keeping the aspect ratio using `LetterboxImages()`.

```c++
// Download: https://datasets.mlpack.org/jurassic-park.png
arma::mat image;
mlpack::data::ImageOptions opt;
mlpack::data::Load("jurassic-park.png", image, opt, true);
mlpack::data::LetterboxImages(image, opt, 416, 416, 127.0);
// Image dimensions are now 416x416.
mlpack::data::Save("jurassic-park-letterbox.png", image, opt, true);

std::cout << "Dimensions: " << opt.Width() << " x " << opt.Height()
          << " x " << opt.Channels() << "\n";
std::cout << "Total size: " << image.n_rows << "\n";
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
```

For more information regarding loading / saving each data type, please visit
the following pages:

 * [Numeric data](data/load_save_numeric.md)
 * [Categorical data](data/load_save_categorical.md)
 * [Image data](data/load_save_images.md)
 * [mlpack objects / models](data/load_save_models.md)
 * [Data Options](data/data_options.md)
