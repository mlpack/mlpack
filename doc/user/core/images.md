# Image preprocessing

mlpack provides a set of functions to preprocess images for training and
testing.

 * [`ResizeImages()`](#resizing-images): resize a matrix with a set of images.
   Images need to have identical dimensions.
 * [`ResizeCropImages()`](#resize-and-crop-images): resize the images and then crop
   to keep the same aspect ratio.
 * [`GroupChannels()`](#groupchannels): Convert image pixel order from
   `[r, g, b, r, g, b, ... ]` to `[r, r, ..., g, g, ..., b, b]`.
 * [`InterleaveChannels()`](#interleavechannels): Convert image pixel order
   from `[r, r, ..., g, g, ..., b, b]` to  `[r, g, b, r, g, b, ... ]`.
 * [`LetterboxImages()`](#letterbox-transform): resize the images and keep the
   same aspect ratio without cropping.

## Resizing images

The `ResizeImages()` function can be used to resize image data:

- `ResizeImages(images, opts, newWidth, newHeight)`
   * `images` is a [column-major matrix](../matrices.md) containing a set of
      images; each image is represented as a flattened vector in one column.

   * `opts` is a [`ImageOptions&`](../load_save.md#imageoptions) containing details about
     the images in `images`, and will be modified to contain the new size of the
     images.

   * `newWidth` and `newHeight` (of type `size_t`) are the desired new
     dimensions of the resized images.

   * This function returns `void` and modifies `opts` and `images`.

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
mlpack::ImageOptions opts;
opts.Fatal() = false;

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
  mlpack::Load(files.at(i), image, opts);
  mlpack::ResizeImages(image, opts, 320, 320);
  mlpack::Save(reSheeps.at(i), image, opts);
}
```

Example usage of `ResizeImages()` function on a set of images that have the
same dimensions.

```c++
// All images have the same dimension, It would be possible to load all of
// them into one matrix

// See https://datasets.mlpack.org/sheep.tar.bz2
arma::Mat<unsigned char> images;
mlpack::ImageOptions opts;
opts.Fatal() = false;

std::vector<std::string> reSheeps =
    {"re_sheep_1.jpg", "re_sheep_2.jpg", "re_sheep_3.jpg", "re_sheep_4.jpg",
     "re_sheep_5.jpg", "re_sheep_6.jpg", "re_sheep_7.jpg", "re_sheep_8.jpg",
     "re_sheep_9.jpg"};

mlpack::Load(reSheeps, images, opts);

// Now let us resize all these images at once, to specific dimensions.
mlpack::ResizeImages(images, opts, 160, 160);

// The resized images will be saved locally. We are declaring the vector that
// contains the names of the resized images.
std::vector<std::string> smSheeps =
    {"sm_sheep_1.jpg", "sm_sheep_2.jpg", "sm_sheep_3.jpg", "sm_sheep_4.jpg",
     "sm_sheep_5.jpg", "sm_sheep_6.jpg", "sm_sheep_7.jpg", "sm_sheep_8.jpg",
     "sm_sheep_9.jpg"};

mlpack::Save(smSheeps, images, opts);
```

## Resize and crop images

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
  <img src="../../img/cat.jpg" alt="cat">
</p>

*Original image with target size of* `220`x`220` *pixels:*

<p align="center">
  <img src="../../img/cat_rect.jpg" alt="cat with rectangle overlaid">
</p>

*First step: resize while preserving aspect ratio:*

<p align="center">
  <img src="../../img/cat_scaled_rect.jpg"
       alt="scaled cat with rectangle overlaid">
</p>

*Second step: crop to desired final size:*

<p align="center">
  <img src="../../img/cat_cropped.jpg" alt="cropped cat">
</p>

- `ResizeCropImages(images, opts, newWidth, newHeight)`
   * `images` is a [column-major matrix](../matrices.md) containing a set of
      images; each image is represented as a flattened vector in one column.

   * `opts` is a [`ImageOptions&`](../load_save.md#imageoptions) containing details about
     the images in `images`.

   * `images` and `opts` are modified in-place.

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
mlpack::ImageOptions opts;
opts.Fatal() = false;

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
  mlpack::Load(files.at(i), image, opts);
  mlpack::ResizeCropImages(image, opts, 320, 320);
  mlpack::Save(cropSheeps.at(i), image, opts);
  std::cout << "Resized and cropped " << files.at(i) << " to "
      << cropSheeps.at(i) << " with output size 320x320." << std::endl;
}
```

## Changing the memory layout of images

When loading images using `Load()` channels are interleaved, i.e.
the underlying vector contains the values `[r, g, b, r, g, b, ... ]`
(for an image with 3 channels). mlpack has functionality such as `Convolution`
that requires channels be grouped, e.g `[r, r, ..., g, g, ..., b, b]`.
The same is true when using `Save()`, the channels are expected to be
interleaved.

To convert the layout of your image from interleaved channels to grouped
channels and vice versa, you can use `GroupChannels()` and
`InterleaveChannels()`.

***NOTE***: Other image related functions (such as
[`ResizeImages`](#resizing-images) etc) require channels be interleaved. If you
need to use `GroupChannels()` make sure to resize or crop your images first
beforehand.

---

#### `GroupChannels()`

 * `GroupChannels(images, opts)`
    - `images` must be a matrix where each column is an image. Each image is
      expected to be interleaved, i.e. in the format `[r, g, b, r, g, b ... ]`.

    - `opts` ImageOptions object describes the shape of each image.

    - Returns a matrix where each image from `images` are in the
      format `[r, r, ... , g, g, ... , b, b]`.

---

#### `InterleaveChannels()`

 * `InterleaveChannels(images, opts)`
    - Performs the reverse of `GroupChannels()`.

    - `images` must be a matrix where each column is an image. Each image is
      expected to be grouped, i.e. in the format `[r, r, ..., g, g, ..., b, b]`.

    - `opts` ImagesOptions object describes the shape of each image.

    - Returns a matrix where each image from `images` are in the
      format `[r, g, b, r, g, b ... ]`.

#### Example

This example loads an image, and converts the layout such that channels are
grouped together in preparation for a convolutional neural network. Then, the
image is converted back to interleaved channels and saved.

```c++
// Download: https://datasets.mlpack.org/images/mlpack-favicon.png
arma::mat image;
mlpack::ImageOptions opts;
opts.Fatal() = true;
mlpack::Load("mlpack-favicon.png", image, opts);

std::vector<std::string> colors =
     { "\033[31m", "\033[32m", "\033[34m", "\033[37m" };

// Display input before grouping channels (Load() returns channels interleaved).
std::cout << "Original Image (channels interleaved):" << std::endl;
for (size_t i = 0; i < image.n_rows; i += opts.Channels())
{
  for (size_t j = 0; j < opts.Channels(); j++)
    std::cout << colors[j] << image.at(i + j, 0) << "\033[0m" << ", ";
}
std::cout << std::endl << std::endl;

// Group channels.
image = mlpack::GroupChannels(image, opts);

// Display submatrix of input after grouping channels
std::cout << "Grouped channels:" << std::endl;
for (size_t i = 0; i < opts.Channels(); i++)
{
  for (size_t j = 0; j < image.n_rows / opts.Channels(); j++)
    std::cout << colors[i] <<
      image.at(i * image.n_rows / opts.Channels() + j, 0) << "\033[0m" << ", ";
}
std::cout << std::endl << std::endl;

// Do some computation here; for example, a convolutional neural network.

// Interleave channels to prepare for saving.
image = mlpack::InterleaveChannels(image, opts);

// Display input after interleaving channels.
// This should be identical to the original image.
std::cout << "Interleaved channels (identical to original):" << std::endl;
for (size_t i = 0; i < image.n_rows; i += opts.Channels())
{
  for (size_t j = 0; j < opts.Channels(); j++)
    std::cout << colors[j] << image.at(i + j, 0) << "\033[0m" << ", ";
}
std::cout << std::endl << std::endl;

mlpack::Save("mlpack-favicon.png", image, opts);
```

### Letterbox transform

The letterbox transform resizes an image's dimensions to `width x height` but
keeps the aspect ratio of the original image. Whitespace is then filled in
with `fillValue`.

*Original image with size of* `640`x`326` *pixels:*

<p align="center">
  <img src="../../img/cat.jpg" alt="cat">
</p>

*Image with target size of* `416`x`416` *pixels after letterbox:*

<p align="center">
  <img src="../../img/cat_square_letterbox.jpg"
       alt="cat with square letterbox transform">
</p>

*Image with target size of* `300`x`208` *pixels after letterbox:*

<p align="center">
  <img src="../../img/cat_rect_letterbox.jpg"
       alt="cat with rectangular letterbox transform">
</p>

- `LetterboxImages(src, opt, width, height, fillValue)`
  * `src` is a [column-major matrix](../matrices.md) containing a single image,
    where the image is represented as a flattened vector in one column.
  * `opt` is an [`ImageOptions&`](../load_save.md#imageoptions) containing info on
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
mlpack::ImageOptions opts;
opts.Fatal() = true;
mlpack::Load("jurassic-park.png", image, opts);
mlpack::LetterboxImages(image, opts, 416, 416, 127.0);
// Image dimensions are now 416x416.
mlpack::Save("jurassic-park-letterbox.png", image, opts);

std::cout << "Dimensions: " << opts.Width() << " x " << opts.Height()
          << " x " << opts.Channels() << "\n";
std::cout << "Total size: " << image.n_rows << "\n";
```

## Draw bounding boxes for object detection

mlpack comes with a utility function to draw bounding boxes onto images when
doing tasks such as object detection.

<p align="center">
  <img src="../img/jurassic-park-logo-box.png" alt="jurassic park logo">
</p>

You can do this through the `BoundingBoxImage()` function.

---

#### `BoundingBoxImage()`

  * `BoundingBoxImage(src, opts, bbox, color, borderSize = 1, className = "", letterSize = 1)`
    - `src` is [image data](#image-data) (e.g. a column vector) where the bounding box
      will be drawn. The channels of the image must be interleaved,
      which you can read about [here](#changing-the-memory-layout-of-images).
      There must be at most one image, otherwise an exception will be thrown.
      Pixel values are expected to be in the 0-255 range.

    - `opts` is the [`ImageOptions`](#imageoptions) object containing metadata relating
      to the image.

    - `bbox` is a [column vector](matrices.md#representing-data-in-mlpack) representing the bounding box to be drawn as a four-element vector: `(x1, y1, x2, y2)`.
      * Elements after the fourth in `bbox` are ignored. There must be at least four elements,
        otherwise an exception will be thrown.
      * The area of the bounding box must be greater than 0.
      * If `x1 >= x2` or `y1 >= y2` an exception will be thrown.
      * Bounding boxes larger than the image will be clipped and their borders will lie along
        the image's edge.

    - `color` is a [column vector](matrices.md#representing-data-in-mlpack) representing the
      color of the bounding box. It must have the same number of elements as `opts.Channels()`.

    - `borderSize` is a `size_t` representing the width of the
      bounding box in pixels. If border size is set to 0, no bounding box will
      be drawn. The default border size is 1.

    - `className` is a string representing the class name given to the bounding
      box. If the string is empty (the default), no name will be drawn.
      Letters will be truncated if they do not fit entirely onto the image. The font
      included uses the `font8x8_basic` from
      [https://github.com/dhepper/font8x8](https://github.com/dhepper/font8x8).

    - `letterSize` represents the size of each letter. When set to 1 (the default), each
      letter is 8x8 pixels. `letterSize` is a multiplier, so when set to 2
      each letter is 16x16 pixels. If letterSize is 0, no class name will be
      printed.

#### Example

An example that draws a red bounding box onto an image, with the class
name `Jurassic Park Logo`.

```c++
// See https://datasets.mlpack.org/jurassic-park.png
arma::mat image;
mlpack::data::ImageOptions opts;
opts.Fatal() = true;
mlpack::data::Load("jurassic-park.png", image, opts);

arma::vec bbox({90, 80, 510, 370});
arma::vec color({255, 0, 0}); // Use red for the border.
std::string className = "Jurassic Park Logo";
size_t borderSize = 2;
size_t letterSize = 2;
mlpack::data::BoundingBoxImage(image, opts, bbox, color, borderSize,
                               className, letterSize);
mlpack::data::Save("jurassic-park-box.png", image, opts, true);
```

