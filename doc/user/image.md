# Working with Images in mlpack

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