# Image data loading

If the STB image library is available on the system (`stb_image.h` and
`stb_image_write.h` must be available on the compiler's include search path),
then mlpack will define the `MLPACK_HAS_STB` macro, and support for loading
individual images or sets of images will be available.

Supported formats for loading are `jpg`, `png`, `tga`, `bmp`, `psd`, `gif`, `pic`, and `pnm`.

Supported formats for saving are `jpg`, `png`, `tga`, and `bmp`.

When loading images, each image is represented as a flattened single column
vector in a data matrix; each row of the resulting vector will correspond to a
single pixel value in a single channel.  An auxiliary `data::ImageOptions` class
is used to store information about the images.

### `data::ImageOptions`

The `data::ImageOptions` class contains the metadata of the images.

---

#### Constructors

 - `opts = data::ImageOptions()`
   * Create a `data::ImageOptions` object with no data.
   * Use this constructor if you intend to populate the `data::ImageOptions` via a
     `data::Load()` call.

 - `opts = data::ImageOptions(width, height, channels)`
   * Create a `data::ImageOptions` object with the given image specifications.
   * `width` and `height` are specified as pixels.

---

#### Accessing and modifying image metadata

 - `opts.Quality() = q` will set the compression quality (e.g. for saving JPEGs)
   to `q`.
   * `q` should take values between `0` and `100`.
   * The quality value is ignored unless calling `data::Save()` with `opts`.

 - Calling `opts.Channels() = 1` before loading will cause images to be loaded
   in grayscale.

 - Metadata stored in the `data::ImageOptions` can be accessed with the following
   members:
   * `opts.Width()` returns the image width in pixels.
   * `opts.Height()` returns the image height in pixels.
   * `opts.Channels()` returns the number of color channels in the image.
   * `opts.Quality()` returns the compression quality that will be used to save
     images (between 0 and 100).

---

### Loading images

With a `data::ImageOptions` object, image data can be loaded or saved, handling
either one or multiple images at a time. This object need to be defined before
calling `data::Load` or `data::Save`:

<!-- TODO: add parameter to force use of what's in `info` -->

 - `data::ImageOptions opts;`

 - `data::Load(filename, matrix, opts)`
   * Load a ***single image*** from `filename` into `matrix`.
     - Format is chosen by extension (e.g. `image.png` will load as PNG).

   * `matrix` will have one column representing the image as a flattened vector.

   * `opts` will be populated with information from the image in `filename`.

   * To make the function to throw a `std::runtime_error` on failure, please
     set `opts.Fatal() = true;`

   * Returns a `bool` indicating the success of the operation.

   * ***NOTE:*** if the element type of `images` is not `unsigned char`
     (e.g. if `image` is not `arma::Mat<unsigned char>`, the matrix will be
     temporarily converted during loading to `unsigned char` and then
     converted back to the original element type at the end of the loading
     process.

---

 - `data::Load(files, matrix, opts)`
   * Load ***multiple images*** from `files` into `matrix`.
     - `files` is of type `std::vector<std::string>` and should contain the list
       of images to be loaded.
     - `matrix` will have `files.size()` columns, each representing the
       corresponding image as a flattened vector.

   * `opts` will be populated with information from the images in `files`.

   * To make the function to throw a `std::runtime_error` on failure, please
     set `opts.Fatal() = true;`

   * Returns a `bool` indicating the success of the operation.

   * ***NOTE:*** if the element type of `images` is not `unsigned char`
     (e.g. if `image` is not `arma::Mat<unsigned char>`, the matrix will be
     temporarily converted during loading to `unsigned char` and then
     converted back to the original element type at the end of the loading
     process.

   * This function expects all the images to have identical
     dimensions. If this is not the case, iteratively call the above overload 
     `Load(filename)` with a single image/column in `images`.

---

 - `data::Save(filename, matrix, opts)`
   * Save a ***single image*** from `matrix` into the file `filename`.
     - Format is chosen by extension (e.g. `image.png` will save as PNG).

   * `matrix` is expected to have only one column representing the image as a
     flattened vector.

   * `opts` must have the dimension information from the images in `filename`.

   * To make the function to throw a `std::runtime_error` on failure, please
     set `opts.Fatal() = true;`

   * Returns a `bool` indicating the success of the operation.

   * ***NOTE:*** if the element type of `images` is not `unsigned char`
     (e.g. if `image` is not `arma::Mat<unsigned char>`, the matrix will be
     temporarily converted during loading to `unsigned char` and then
     converted back to the original element type at the end of the loading
     process.

---

 - `data::Save(files, matrix, opts)`
   * Save ***multiple images*** from `matrix` into `files`.
     - `files` is of type `std::vector<std::string>` and should contain the list
       of files to save to.
     - The format of each file is chosen by extension (e.g. `image.png` will
       save as PNG); it is possible for filenames in `files` to have different
       extensions, but this is not recommended.

   * `matrix` is expected to have `files.size()` columns representing images as
     flattened vectors.

   * `opts` must have the dimension information from the images in `filename`.

   * To make the function to throw a `std::runtime_error` on failure, please
     set `opts.Fatal() = true;`

   * Returns a `bool` indicating the success of the operation.

   * ***NOTE:*** if the element type of `images` is not `unsigned char`
     (e.g. if `image` is not `arma::Mat<unsigned char>`, the matrix will be
     temporarily converted during loading to `unsigned char` and then
     converted back to the original element type at the end of the loading
     process.

   * This function expects all the images to have identical
     dimensions. If this is not the case, iteratively call the above overload 
     `Save(filename)` with a single image/column in `images`.

---

The above functions could be used with the following simplified signature:

```
    // To load a png image immediately, but no need to know the dimensions.
    mlpack::data::load(filename, matrix, PNG);
    
    // The same would apply for JPG as well, if we want to throw an exception.
    // You can follow the same logic to load other image formats.
    mlpack::data::load(filename, matrix, JPG + Fatal);

    // Another Images is specified instead of the extension. The format will be
    decided based on the extension. For instance:
    // This will load a PNG
    mlpack::data::load("myimage.png", matrix, Image);
```

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
mlpack::data::ImageOptions opts;
opts.Fatal() = true;
arma::mat matrix;
mlpack::data::Load("numfocus-logo.png", matrix, opts);

// `matrix` should now contain one column.

// Print optsrmation about the image.
std::cout << "Information about the image in 'numfocus-logo.png': "
    << std::endl;
std::cout << " - " << opts.Width() << " pixels in width." << std::endl;
std::cout << " - " << opts.Height() << " pixels in height." << std::endl;
std::cout << " - " << opts.Channels() << " color channels." << std::endl;

std::cout << "Value at pixel (x=3, y=4) in the first channel: ";
const size_t index = (4 * opts.Width() * opts.Channels()) +
    (3 * opts.Channels());
std::cout << matrix[index] << "." << std::endl;

// Increment each pixel value, but make sure they are still within the bounds.
matrix += 1;
matrix = arma::clamp(matrix, 0, 255);

mlpack::data::Save("numfocus-logo-mod.png", matrix, opts);
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

mlpack::data::ImageOptions opts;
opts.Channels() = 1; // Force loading in grayscale.

arma::mat matrix;
mlpack::data::Load(images, matrix, opts, true);

// Print optsrmation about what we loaded.
std::cout << "Loaded " << matrix.n_cols << " images.  Images are of size "
    << opts.Width() << " x " << opts.Height() << " with " << opts.Channels()
    << " color channel." << std::endl;

// Invert images.
matrix = (255.0 - matrix);

// Save as compressed JPEGs with low quality.
opts.Quality() = 75;
std::vector<std::string> outImages;
outImages.push_back("mlpack-favicon-inv.jpeg");
outImages.push_back("ensmallen-favicon-inv.jpeg");
outImages.push_back("armadillo-favicon-inv.jpeg");
outImages.push_back("bandicoot-favicon-inv.jpeg");

mlpack::data::Save(outImages, matrix, opts);
```

### Resize images

It is possible to resize images in mlpack with the following function:

- `ResizeImages(images, info, newWidth, newHeight)`
   * `images` is a [column-major matrix](matrices.md) containing a set of
      images; each image is represented as a flattened vector in one column.

   * `opts` is a [`data::ImageOptions&`](#dataimageinfo) containing details about
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
mlpack::data::ImageOptions opts;

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
// the same dimensions. The `opts` will contain the dimension for each one.
for (size_t i = 0; i < files.size(); i++)
{
  mlpack::data::Load(files.at(i), image, opts, false);
  mlpack::data::ResizeImages(image, opts, 320, 320);
  mlpack::data::Save(reSheeps.at(i), image, opts, false);
}
```

Example usage of `ResizeImages()` function on a set of images that have the
same dimensions.

```c++
// All images have the same dimension, It would be possible to load all of
// them into one matrix

// See https://datasets.mlpack.org/sheep.tar.bz2
arma::Mat<unsigned char> images;
mlpack::data::ImageOptions opts;

std::vector<std::string> reSheeps =
    {"re_sheep_1.jpg", "re_sheep_2.jpg", "re_sheep_3.jpg", "re_sheep_4.jpg",
     "re_sheep_5.jpg", "re_sheep_6.jpg", "re_sheep_7.jpg", "re_sheep_8.jpg",
     "re_sheep_9.jpg"};

mlpack::data::Load(reSheeps, images, opts, false);

// Now let us resize all these images at once, to specific dimensions.
mlpack::data::ResizeImages(images, opts, 160, 160);

// The resized images will be saved locally. We are declaring the vector that
// contains the names of the resized images.
std::vector<std::string> smSheeps =
    {"sm_sheep_1.jpg", "sm_sheep_2.jpg", "sm_sheep_3.jpg", "sm_sheep_4.jpg",
     "sm_sheep_5.jpg", "sm_sheep_6.jpg", "sm_sheep_7.jpg", "sm_sheep_8.jpg",
     "sm_sheep_9.jpg"};

mlpack::data::Save(smSheeps, images, opts, false);
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

   * `opts` is a [`data::ImageOptions&`](#dataimageinfo) containing details about
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
mlpack::data::ImageOptions opts;

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
// the same dimensions. The `opts` will contain the dimension for each one.
for (size_t i = 0; i < files.size(); i++)
{
  mlpack::data::Load(files.at(i), image, opts, false);
  mlpack::data::ResizeCropImages(image, opts, 320, 320);
  mlpack::data::Save(cropSheeps.at(i), image, opts, false);
  std::cout << "Resized and cropped " << files.at(i) << " to "
      << cropSheeps.at(i) << " with output size 320x320." << std::endl;
}
```
