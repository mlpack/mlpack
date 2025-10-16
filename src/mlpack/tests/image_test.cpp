/**
 * @file tests/image_load_test.cpp
 * @author Mehul Kumar Nirala
 *
 * Tests for loading and saving images.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include "serialization.hpp"
#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

/**
 * Test if an image with an unsupported extension throws an expected
 * exception.
 */
TEST_CASE("LoadInvalidExtensionFile", "[ImageLoadTest]")
{
  arma::Mat<unsigned char> matrix;
  data::ImageInfo info;

  REQUIRE_THROWS_AS(data::Load("invalidExtension.p4ng", matrix, info,
      true),  std::runtime_error);
}

/**
 * Test that the image is loaded correctly into the matrix using the API.
 */
TEST_CASE("LoadImageAPITest", "[ImageLoadTest]")
{
  arma::Mat<unsigned char> matrix;
  data::ImageInfo info;

  REQUIRE(data::Load("test_image.png", matrix, info, false) == true);
  // width * height * channels.
  REQUIRE(matrix.n_rows == 50 * 50 * 3);
  REQUIRE(info.Height() == 50);
  REQUIRE(info.Width() == 50);
  REQUIRE(info.Channels() == 3);
  REQUIRE(matrix.n_cols == 1);
}

/**
 * Test that the image is loaded correctly into the matrix using the new API.
 */
TEST_CASE("LoadImageNewAPITest", "[ImageLoadTest]")
{
  arma::Mat<unsigned char> matrix;
  data::ImageOptions opts;
  opts.Fatal() = false;

  REQUIRE(data::Load("test_image.png", matrix, opts) == true);
  // width * height * channels.
  REQUIRE(matrix.n_rows == 50 * 50 * 3);
  REQUIRE(opts.Height() == 50);
  REQUIRE(opts.Width() == 50);
  REQUIRE(opts.Channels() == 3);
  REQUIRE(matrix.n_cols == 1);
}

/**
 * Test that the image is loaded correctly when specifying the type.
 */
TEST_CASE("LoadImageSpecifyTypeTest", "[ImageLoadTest]")
{
  arma::Mat<unsigned char> matrix;
  data::ImageOptions opts;
  opts.Fatal() = false;
  opts.Format() = FileType::PNG;

  REQUIRE(data::Load("test_image.png", matrix, opts) == true);
  // width * height * channels.
  REQUIRE(matrix.n_rows == 50 * 50 * 3);
  REQUIRE(opts.Height() == 50);
  REQUIRE(opts.Width() == 50);
  REQUIRE(opts.Channels() == 3);
  REQUIRE(matrix.n_cols == 1);
}

/**
 * Test that the image is loaded correctly if the type is specified in the
 * function.
 */
TEST_CASE("LoadPNGImageTest", "[ImageLoadTest]")
{
  arma::Mat<unsigned char> matrix;
  REQUIRE(data::Load("test_image.png", matrix, PNG + Fatal) == true);
}

/**
 * Test that the image is loaded correctly if the type is not specified in the
 * function.
 */
TEST_CASE("LoadPNGImageTestNoFormat", "[ImageLoadTest]")
{
  arma::Mat<unsigned char> matrix;
  REQUIRE(data::Load("test_image.png", matrix, Fatal) == true);
}

/**
 * Test when loading an image with the wrong data options
 */
TEST_CASE("LoadWrongDataOptions", "[ImageLoadTest]")
{
  arma::Mat<unsigned char> matrix;
  TextOptions opts;
  opts.Fatal() = true;
  REQUIRE(data::Load("test_image.png", matrix, opts) == true);
}

/**
 * Test if the image is saved correctly using API.
 */
TEST_CASE("SaveImageAPITest", "[ImageLoadTest]")
{
  data::ImageInfo info(5, 5, 3, 90);

  arma::Mat<unsigned char> im1;
  size_t dimension = info.Width() * info.Height() * info.Channels();
  im1 = arma::randi<arma::Mat<unsigned char>>(dimension, 1);
  REQUIRE(data::Save("APITest.bmp", im1, info, false) == true);

  arma::Mat<unsigned char> im2;
  REQUIRE(data::Load("APITest.bmp", im2, info, false) == true);

  REQUIRE(im1.n_cols == im2.n_cols);
  REQUIRE(im1.n_rows == im2.n_rows);
  for (size_t i = 0; i < im1.n_elem; ++i)
    REQUIRE(im1[i] == im2[i]);
  remove("APITest.bmp");
}


/**
 * Test if the image is saved correctly using the new API.
 */
TEST_CASE("SaveImageNewAPITest", "[ImageLoadTest]")
{
  data::ImageInfo opts(5, 5, 3, 90);
  opts.Fatal() = false;

  arma::Mat<unsigned char> im1;
  size_t dimension = opts.Width() * opts.Height() * opts.Channels();
  im1 = arma::randi<arma::Mat<unsigned char>>(dimension, 1);

  REQUIRE(data::Save("APITest.bmp", im1, opts) == true);

  arma::Mat<unsigned char> im2;
  REQUIRE(data::Load("APITest.bmp", im2, opts) == true);

  REQUIRE(im1.n_cols == im2.n_cols);
  REQUIRE(im1.n_rows == im2.n_rows);
  for (size_t i = 0; i < im1.n_elem; ++i)
    REQUIRE(im1[i] == im2[i]);
  remove("APITest.bmp");
}

/**
 * Test if an image with a wrong dimesion throws an expected
 * exception while saving.
 */
TEST_CASE("SaveImageWrongInfo", "[ImageLoadTest]")
{
  data::ImageInfo info(5, 5, 3, 90);

  arma::Mat<unsigned char> im1;
  im1 = arma::randi<arma::Mat<unsigned char>>(24 * 25 * 7, 1);
  REQUIRE_THROWS_AS(data::Save("APITest.bmp", im1, info, true),
      std::runtime_error);
}

/**
 * Test if an image with a wrong dimesion throws an expected
 * exception while loading.
 */
TEST_CASE("LoadImageWrongInfo", "[ImageLoadTest]")
{
  data::ImageInfo info(5, 5, 3, 90);

  arma::Mat<unsigned char> im1;
  im1 = arma::randi<arma::Mat<unsigned char>>(24 * 25 * 7, 1);
  REQUIRE_THROWS_AS(data::Load("APITest.bmp", im1, info, true),
      std::runtime_error);
}

/**
 * Test if loading a set of images with different dimensions will fail..
 */
TEST_CASE("LoadSetOfImagesNoInfo", "[ImageLoadTest]")
{
  std::vector<std::string> files =
      {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
       "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
       "sheep_9.jpg"};

  arma::Mat<unsigned char> im1;
  REQUIRE_THROWS_AS(data::Load(files, im1, JPG + Fatal),
      std::runtime_error);
}

TEST_CASE("LoadSetOfImagesWrongInfo", "[ImageLoadTest]")
{
  data::ImageOptions opts(5, 5, 3, 90);
  opts.Fatal() = true;
  std::vector<std::string> files =
      {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
       "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
       "sheep_9.jpg"};

  arma::Mat<unsigned char> im1;
  REQUIRE_THROWS_AS(data::Load(files, im1, opts),
      std::runtime_error);
}

/**
 * Test that the image is loaded correctly into the matrix using the API
 * for vectors.
 */
TEST_CASE("LoadVectorImageAPITest", "[ImageLoadTest]")
{
  arma::Mat<unsigned char> matrix;
  data::ImageInfo info;
  std::vector<std::string> files = {"test_image.png", "test_image.png"};
  REQUIRE(data::Load(files, matrix, info, false) == true);
  // width * height * channels.
  REQUIRE(matrix.n_rows == 50 * 50 * 3);
  REQUIRE(info.Height() == 50);
  REQUIRE(info.Width() == 50);
  REQUIRE(info.Channels() == 3);
  REQUIRE(matrix.n_cols == 2);
}

/**
 * Test resize the image if this is done correctly.  Try it with a few different
 * types.
 */
TEMPLATE_TEST_CASE("ImagesResizeTest", "[ImageTest]", unsigned char, size_t,
    float, double)
{
  typedef TestType eT;

  arma::Mat<eT> image, images;
  data::ImageInfo info, resizedInfo, resizedInfo2;
  std::vector<std::string> files =
      {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
       "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
       "sheep_9.jpg"};
  std::vector<std::string> reSheeps =
      {"re_sheep_1.jpg", "re_sheep_2.jpg", "re_sheep_3.jpg", "re_sheep_4.jpg",
       "re_sheep_5.jpg", "re_sheep_6.jpg", "re_sheep_7.jpg", "re_sheep_8.jpg",
       "re_sheep_9.jpg"};
  std::vector<std::string> smSheeps =
      {"sm_sheep_1.jpg", "sm_sheep_2.jpg", "sm_sheep_3.jpg", "sm_sheep_4.jpg",
       "sm_sheep_5.jpg", "sm_sheep_6.jpg", "sm_sheep_7.jpg", "sm_sheep_8.jpg",
       "sm_sheep_9.jpg"};

  // Load and Resize each one of them individually, because they do not have
  // the same sizes, and then the resized images, will be used in the next
  // test.
  for (size_t i = 0; i < files.size(); i++)
  {
    info.Reset();
    REQUIRE(data::Load(files.at(i), image, info, true) == true);
    ResizeImages(image, info, 320, 320);
    REQUIRE(data::Save(reSheeps.at(i), image, info, true) == true);
  }

  // Since they are all resized, this should passes
  REQUIRE(data::Load(reSheeps, images, resizedInfo, false) == true);

  REQUIRE(info.Width() == resizedInfo.Width());
  REQUIRE(info.Height() == resizedInfo.Height());

  REQUIRE(data::Load(reSheeps, images, info, false) == true);

  ResizeImages(images, info, 160, 160);

  REQUIRE(data::Save(smSheeps, images, info, false) == true);

  REQUIRE(data::Load(smSheeps, images, resizedInfo2, false) == true);

  REQUIRE(info.Width() == resizedInfo2.Width());
  REQUIRE(info.Height() == resizedInfo2.Height());

  // cleanup generated images.
  for (size_t i = 0; i < reSheeps.size(); ++i)
  {
    remove(reSheeps.at(i).c_str());
    remove(smSheeps.at(i).c_str());
  }
}

/**
 * Test resize the image if this is done correctly.  Try it with a few different
 * types.
 */
TEMPLATE_TEST_CASE("ImagesResizeCropTest", "[ImageTest]", unsigned char,
    size_t, float, double)
{
  typedef TestType eT;

  arma::Mat<eT> image, images;
  data::ImageInfo info, resizedInfo, resizedInfo2;
  std::vector<std::string> files =
      {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
       "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
       "sheep_9.jpg"};
  std::vector<std::string> reSheeps =
      {"re_sheep_1.jpg", "re_sheep_2.jpg", "re_sheep_3.jpg", "re_sheep_4.jpg",
       "re_sheep_5.jpg", "re_sheep_6.jpg", "re_sheep_7.jpg", "re_sheep_8.jpg",
       "re_sheep_9.jpg"};
  std::vector<std::string> smSheeps =
      {"sm_sheep_1.jpg", "sm_sheep_2.jpg", "sm_sheep_3.jpg", "sm_sheep_4.jpg",
       "sm_sheep_5.jpg", "sm_sheep_6.jpg", "sm_sheep_7.jpg", "sm_sheep_8.jpg",
       "sm_sheep_9.jpg"};

  // Load and Resize each one of them individually, because they do not have
  // the same sizes, and then the resized images, will be used in the next
  // test.
  for (size_t i = 0; i < files.size(); i++)
  {
    info.Reset();
    REQUIRE(data::Load(files.at(i), image, info, true) == true);
    ResizeCropImages(image, info, 320, 320);
    REQUIRE(data::Save(reSheeps.at(i), image, info, false) == true);
  }

  // Since they are all resized, this should passes
  REQUIRE(data::Load(reSheeps, images, resizedInfo, false) == true);

  REQUIRE(info.Width() == resizedInfo.Width());
  REQUIRE(info.Height() == resizedInfo.Height());

  REQUIRE(data::Load(reSheeps, images, info, false) == true);

  ResizeCropImages(images, info, 160, 160);

  REQUIRE(data::Save(smSheeps, images, info, false) == true);

  REQUIRE(data::Load(smSheeps, images, resizedInfo2, false) == true);

  REQUIRE(info.Width() == resizedInfo2.Width());
  REQUIRE(info.Height() == resizedInfo2.Height());

  // cleanup generated images.
  for (size_t i = 0; i < reSheeps.size(); ++i)
  {
    remove(reSheeps.at(i).c_str());
    remove(smSheeps.at(i).c_str());
  }
}

/**
 * Test if we resize to the same original dimension we will get the same pixels
 * and no modification to the image.  Try it with a few different types.
 */
TEMPLATE_TEST_CASE("IdenticalResizeTest", "[ImageTest]", unsigned char, size_t,
    float, double)
{
  typedef TestType eT;

  arma::Mat<eT> image;
  data::ImageInfo info;
  std::vector<std::string> files =
      {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
       "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
       "sheep_9.jpg"};

  for (size_t i = 0; i < files.size(); i++)
  {
    info.Reset();
    REQUIRE(data::Load(files.at(i), image, info, false) == true);
    arma::Mat<eT> originalImage = image;
    ResizeImages(image, info, info.Width(), info.Height());
    if (std::is_same_v<eT, float> || std::is_same_v<eT, double>)
    {
      REQUIRE(arma::approx_equal(originalImage, image, "absdiff", 1e-3));
    }
    else
    {
      REQUIRE(arma::approx_equal(originalImage, image, "absdiff", 1e-7));
    }
  }
}

/**
 * Test if we resize to the same original dimension we will get the same pixels
 * and no modification to the image.  Try it with a few different types.
 */
TEMPLATE_TEST_CASE("IdenticalResizeCropTest", "[ImageTest]", unsigned char,
    size_t, float, double)
{
  typedef TestType eT;

  arma::Mat<eT> image;
  data::ImageInfo info;
  std::vector<std::string> files =
      {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
       "sheep_5.jpg", "sheep_6.jpg", "sheep_7.jpg", "sheep_8.jpg",
       "sheep_9.jpg"};

  for (size_t i = 0; i < files.size(); i++)
  {
    info.Reset();
    REQUIRE(data::Load(files.at(i), image, info, false) == true);
    arma::Mat<eT> originalImage = image;
    ResizeCropImages(image, info, info.Width(), info.Height());
    if (std::is_same_v<eT, float> || std::is_same_v<eT, double>)
    {
      REQUIRE(arma::approx_equal(originalImage, image, "absdiff", 1e-3));
    }
    else
    {
      REQUIRE(arma::approx_equal(originalImage, image, "absdiff", 1e-7));
    }
  }
}

/**
 * Test that if we resize an image, we get the pixels that we expect.
 */
TEMPLATE_TEST_CASE("ResizeCropPixelTest", "[ImageTest]", unsigned char, size_t,
    float, double)
{
  typedef TestType eT;

  // Load cat.jpg, which has a strange aspect ratio.
  arma::Mat<eT> image;
  data::ImageInfo info;
  REQUIRE(data::Load("cat.jpg", image, info, false) == true);

  // When we crop to match the height of the image, no resizing is needed and we
  // can compare pixels directly.
  const size_t inputWidth = info.Width();
  const size_t inputHeight = info.Height();
  const size_t inputChannels = info.Channels();
  const size_t leftOffset = (info.Width() - info.Height()) / 2;
  arma::Mat<eT> oldImage(image);
  ResizeCropImages(image, info, inputHeight, inputHeight);

  REQUIRE(info.Height() == inputHeight);
  REQUIRE(info.Width() == inputHeight);
  REQUIRE(info.Channels() == inputChannels);
  REQUIRE(image.n_elem == info.Height() * info.Width() * info.Channels());

  // Now make sure that all of the pixels are the same as from the center of the
  // image.
  for (size_t i = 0; i < image.n_elem; ++i)
  {
    const size_t channel = i % info.Channels();
    const size_t pixel = (i / info.Channels());
    const size_t x = pixel % info.Width();
    const size_t y = pixel / info.Width();

    const size_t inputPixel = y * (inputWidth * inputChannels) +
        (x + leftOffset) * inputChannels + channel;
    const size_t outputPixel = y * (info.Width() * info.Channels()) +
        x * info.Channels() + channel;

    REQUIRE(oldImage[inputPixel] == Approx(image[outputPixel]).epsilon(1e-7));
  }
}

/**
 * Test that images can be upscaled if desired.
 */
TEMPLATE_TEST_CASE("ResizeCropUpscaleTest", "[ImageTest]", unsigned char,
    size_t, float, double)
{
  typedef TestType eT;

  // Load cat.jpg, which has a strange aspect ratio.
  arma::Mat<eT> image;
  data::ImageInfo info;
  REQUIRE(data::Load("cat.jpg", image, info, false) == true);

  // When we crop to match the height of the image, no resizing is needed and we
  // can compare pixels directly.
  const size_t inputChannels = info.Channels();
  ResizeCropImages(image, info, 1000, 1000);

  // Here we just check that the output image has the correct size.
  REQUIRE(info.Height() == 1000);
  REQUIRE(info.Width() == 1000);
  REQUIRE(info.Channels() == inputChannels);
  REQUIRE(image.n_elem == info.Height() * info.Width() * info.Channels());
}

/**
 * Test that groups channels from interleaved channels.
 */
TEST_CASE("GroupChannels", "[ImageTest]")
{
  arma::mat image = arma::regspace(0, 26);
  data::ImageInfo info(3, 3, 3);

  arma::mat newLayout = GroupChannels(image, info);

  std::vector<double> expectedOutput = {
    0, 3, 6, 9, 12, 15, 18, 21, 24,
    1, 4, 7, 10, 13, 16, 19, 22, 25,
    2, 5, 8, 11, 14, 17, 20, 23, 26,
  };

  arma::mat expectedImage(expectedOutput);
  CheckMatrices(newLayout, expectedImage);
}

/**
 * Test that interleaves channels from grouped channels.
 */
TEST_CASE("InterleaveChannels", "[ImageTest]")
{
  data::ImageInfo info(3, 3, 3);
  std::vector<double> data = {
    0, 3, 6, 9, 12, 15, 18, 21, 24,
    1, 4, 7, 10, 13, 16, 19, 22, 25,
    2, 5, 8, 11, 14, 17, 20, 23, 26,
  };
  arma::mat image(data);

  arma::mat newLayout = InterleaveChannels(image, info);
  arma::mat expectedImage = arma::regspace(0, 26);
  CheckMatrices(newLayout, expectedImage);
}

/**
 * Test that groups channels from 2 images whose channels are interleaved.
 */
TEST_CASE("GroupChannels2Images", "[ImageTest]")
{
  data::ImageInfo info(3, 3, 3);
  arma::mat images = arma::reshape(arma::regspace(0, 53), 27, 2);
  arma::mat newImages = GroupChannels(images, info);

  std::vector<double> expectedOutput = {
    0, 3, 6, 9, 12, 15, 18, 21, 24,
    1, 4, 7, 10, 13, 16, 19, 22, 25,
    2, 5, 8, 11, 14, 17, 20, 23, 26,
    27, 30, 33, 36, 39, 42, 45, 48, 51,
    28, 31, 34, 37, 40, 43, 46, 49, 52,
    29, 32, 35, 38, 41, 44, 47, 50, 53,
  };

  arma::mat expectedImages = arma::reshape(arma::mat(expectedOutput), 27, 2);
  CheckMatrices(newImages, expectedImages);
}

/**
 * Test that interleaves channels from 2 images whose channels are grouped.
 */
TEST_CASE("InterleaveChannels2Images", "[ImageTest]")
{
  data::ImageInfo info(3, 3, 3);
  std::vector<double> input = {
    0, 3, 6, 9, 12, 15, 18, 21, 24,
    1, 4, 7, 10, 13, 16, 19, 22, 25,
    2, 5, 8, 11, 14, 17, 20, 23, 26,
    27, 30, 33, 36, 39, 42, 45, 48, 51,
    28, 31, 34, 37, 40, 43, 46, 49, 52,
    29, 32, 35, 38, 41, 44, 47, 50, 53,
  };
  arma::mat images = arma::reshape(arma::mat(input), 27, 2);

  arma::mat newLayout = InterleaveChannels(images, info);
  arma::mat expectedImage = arma::reshape(arma::regspace(0, 53), 27, 2);
  CheckMatrices(newLayout, expectedImage);
}

/**
 * Test grouping channels on empty image.
 */
TEST_CASE("GroupChannelsEmptyImage", "[ImageTest]")
{
  arma::mat image;
  data::ImageInfo info(3, 3, 3);
  REQUIRE_THROWS(GroupChannels(image, info));
}

/**
 * Test interleaving channels on empty image.
 */
TEST_CASE("InterleaveChannelsEmtpyImage", "[ImageTest]")
{
  data::ImageInfo info(3, 3, 3);
  arma::mat image;
  REQUIRE_THROWS(InterleaveChannels(image, info));
}

/**
 * Test grouping channels when there is only one channel.
 */
TEST_CASE("GroupChannelsOneChannel", "[ImageTest]")
{
  arma::mat image = arma::regspace(0, 8);
  data::ImageInfo info(3, 3, 1);
  arma::mat newLayout = GroupChannels(image, info);
  arma::mat expectedImage(image);

  CheckMatrices(newLayout, expectedImage);
}

/**
 * Test interleaving channels when there is only one channel.
 */
TEST_CASE("InterleaveChannelsOneChannel", "[ImageTest]")
{
  arma::mat image = arma::regspace(0, 8);
  data::ImageInfo info(3, 3, 1);
  arma::mat newLayout = InterleaveChannels(image, info);
  arma::mat expectedImage(image);

  CheckMatrices(newLayout, expectedImage);
}

/**
 * Test grouping channels when there is only one pixel.
 */
TEST_CASE("GroupChannelsOnePixel", "[ImageTest]")
{
  data::ImageInfo info(1, 1, 1);
  arma::mat image(1, 1);
  image.fill(5.0);

  arma::mat newLayout = GroupChannels(image, info);
  arma::mat expectedImage(image);

  CheckMatrices(newLayout, expectedImage);
}

/**
 * Test interleaving channels when there is only one pixel.
 */
TEST_CASE("InterleaveChannelsOnePixel", "[ImageTest]")
{
  data::ImageInfo info(1, 1, 1);
  arma::mat image(1, 1);
  image.fill(5.0);

  arma::mat newLayout = InterleaveChannels(image, info);
  arma::mat expectedImage(image);
  CheckMatrices(newLayout, expectedImage);
}
