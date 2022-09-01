# Image Utilities Tutorial

Image datasets are becoming increasingly popular in deep learning.

mlpack's image saving/loading functionality is based on [stb/](https://github.com/nothings/stb).

## Model API

Image utilities supports loading and saving of images.

It supports filetypes `jpg`, `png`, `tga`, `bmp`, `psd`, `gif`, `hdr`, `pic`,
`pnm` for loading and `jpg`, `png`, `tga`, `bmp`, `hdr` for saving.

The datatype associated is unsigned char to support RGB values in the range
1-255. To feed data into the network typecast of `arma::Mat` may be required.
Images are stored in matrix as `(width * height * channels, numberOfImages)`.
Therefore `imageMatrix.col(0)` would be the first image if images are loaded in
`imageMatrix`.

## `ImageInfo`

The `ImageInfo` class contains the metadata of the images.

```c++
/**
 * Instantiate the ImageInfo object with the image width, height, channels.
 *
 * @param width Image width.
 * @param height Image height.
 * @param channels number of channels in the image.
 */
ImageInfo(const size_t width,
          const size_t height,
          const size_t channels);
```

Other public members include the quality compression of the image if saved as
`jpg` (0-100).

## Loading

Standalone loading of images can be done with the function below.

```c++
/**
 * Load the image file into the given matrix.
 *
 * @param filename Name of the image file.
 * @param matrix Matrix to load the image into.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, flips the image, same as transposing the
 *    matrix after loading.
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose);
```

Loading a test image is shown below. It also fills up the `ImageInfo` class
object.

```c++
data::ImageInfo info;
data::Load("test_image.png", matrix, info, false, true);
```

`ImageInfo` requires height, width, number of channels of the image.

```c++
size_t height = 64, width = 64, channels = 1;
data::ImageInfo info(width, height, channels);
```

More than one image can be loaded into the same matrix.

Loading multiple images can be done using the function below.

```c++
/**
 * Load the image file into the given matrix.
 *
 * @param files A vector consisting of filenames.
 * @param matrix Matrix to save the image from.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, flips the image, same as transposing the
 *    matrix after loading.
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose);
```

```c++
data::ImageInfo info;
std::vector<std::string>> files{"test_image1.bmp","test_image2.bmp"};
data::load(files, matrix, info, false, true);
```

## Saving

Saving images expects a matrix of type unsigned char in the form `(width *
height * channels, NumberOfImages)`.  Just like loading, it can be used to save
one image or multiple images. Besides image data it also expects the shape of
the image as input `(width, height, channels)`.

Saving one image can be done with the function below:

```c++
/**
 * Save the image file from the given matrix.
 *
 * @param filename Name of the image file.
 * @param matrix Matrix to save the image from.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, flips the image, same as transposing the
 *    matrix after loading.
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Save(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose);
```

```c++
data::ImageInfo info;
info.width = info.height = 25;
info.channels = 3;
info.quality = 90;
data::Save("test_image.bmp", matrix, info, false, true);
```

If the matrix contains more than one image, only the first one is saved.

Saving multiple images can be done with the function below.

```c++
/**
 * Save the image file from the given matrix.
 *
 * @param files A vector consisting of filenames.
 * @param matrix Matrix to save the image from.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, Flips the image, same as transposing the
 *    matrix after loading.
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Save(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose);
```

```c++
data::ImageInfo info;
info.width = info.height = 25;
info.channels = 3;
info.quality = 90;
std::vector<std::string>> files{"test_image1.bmp", "test_image2.bmp"};
data::Save(files, matrix, info, false, true);
```

Multiple images are saved according to the vector of filenames specified.
