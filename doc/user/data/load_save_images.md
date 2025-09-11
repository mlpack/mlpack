# Image data loading

If the STB image library is available on the system (`stb_image.h` and
`stb_image_write.h` must be available on the compiler's include search path),
then mlpack will define the `MLPACK_HAS_STB` macro, and support for loading
individual images or sets of images will be available.

Supported formats for loading are `jpg`, `png`, `tga`, `bmp`, `psd`, `gif`, `pic`, and `pnm`.

Supported formats for saving are `jpg`, `png`, `tga`, and `bmp`.

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
