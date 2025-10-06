# Image Preprocessing

When loading images using `data::Load()` channels are interleaved, i.e.
`[r, g, b, r, g, b, ... ]`. mlpack has functionality such as `Convolution`
that requires channels be grouped, e.g `[r, r, ..., g, g, ..., b, b]`. The
same is true when using `data::Save()`, the channels are expected to be
interleaved.

To convert the layout of your image from interleaved channels to grouped
channels and vice versa, you can use `data::GroupChannels()` and
`data::InterleaveChannels()`.

---

## `data::GroupChannels()`

 * `data::GroupChannels(images, info)`
    - `images` must be a matrix where each column is an image. Each image is
      expected to be interleaved, i.e. in the format `[r, g, b, r, g, b ... ]`.

    - `info` is an [`ImageInfo`](#dataimageinfo) that describes the shape of each image.

    - Returns a matrix where each image from `images` are in the
      format `[r, r, ... , g, g, ... , b, b]`.  The size of the matrix is the same as the size of `images`.

---

## `data::InterleaveChannels()`

 * `data::InterleaveChannels(images, info)`
    - Performs the reverse of `data::GroupChannels()`.

    - `images` must be a matrix where each column is an image. Each image is
      expected to be grouped, i.e. in the format `[r, r, ..., g, g, ..., b, b]`.

    - `info` describes the shape of each image.

    - Returns a matrix where each image from `images` are in the
      format `[r, g, b, r, g, b ... ]`.

---

## Example

An example that loads an image, normalizes its values between 0-1 and
converts the layout such that channels are grouped together in preparation for
a convolutional neural network. Then convert back to interleaved channels
and save the image.

```c++

arma::mat image;
data::ImageInfo info;
data::Load("example.jpg", image, info, true);

// Convert range to 0-1 and group channels.
image /= 255;
image = data::GroupChannels(image, info);

// Do some computation here, for example data augmentation or convolutional
// neural network.

// Convert range back to 0-255 and interleave channels to prepare for saving.
image *= 255;
image = data::InterleaveChannels(image, info);
data::Save("output.jpg", image, info, true);

```
