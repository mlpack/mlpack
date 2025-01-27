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

#ifndef MLPACK_DISABLE_STB

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
 * Test if an image with a wrong dimesion throws an expected
 * exception while saving.
 */
TEST_CASE("SaveImageWrongInfo", "[ImageLoadTest]")
{
  data::ImageInfo info(5, 5, 3, 90);

  arma::Mat<unsigned char> im1;
  im1 = arma::randi<arma::Mat<unsigned char>>(24 * 25 * 7, 1);
  REQUIRE_THROWS_AS(data::Save("APITest.bmp", im1, info, false),
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
 * Test if the image is saved correctly using API for arma mat.
 */
TEST_CASE("SaveImageMatAPITest", "[ImageLoadTest]")
{
  data::ImageInfo info(5, 5, 3);

  arma::Mat<unsigned char> im1;
  size_t dimension = info.Width() * info.Height() * info.Channels();
  im1 = arma::randi<arma::Mat<unsigned char>>(dimension, 1);
  arma::mat input = ConvTo<arma::mat>::From(im1);
  REQUIRE(Save("APITest.bmp", input, info, false) == true);

  arma::mat output;
  REQUIRE(Load("APITest.bmp", output, info, false) == true);

  REQUIRE(input.n_cols == output.n_cols);
  REQUIRE(input.n_rows == output.n_rows);
  for (size_t i = 0; i < input.n_elem; ++i)
    REQUIRE(input[i] == Approx(output[i]).epsilon(1e-7));
  remove("APITest.bmp");
}

/**
 * Serialization test for the ImageInfo class.
 */
TEST_CASE("ImageInfoSerialization", "[ImageLoadTest]")
{
  data::ImageInfo info(5, 5, 3, 90);
  data::ImageInfo xmlInfo, jsonInfo, binaryInfo;

  SerializeObjectAll(info, xmlInfo, jsonInfo, binaryInfo);

  REQUIRE(info.Width() == xmlInfo.Width());
  REQUIRE(info.Height() == xmlInfo.Height());
  REQUIRE(info.Channels() == xmlInfo.Channels());
  REQUIRE(info.Quality() == xmlInfo.Quality());
  REQUIRE(info.Width() == jsonInfo.Width());
  REQUIRE(info.Height() == jsonInfo.Height());
  REQUIRE(info.Channels() == jsonInfo.Channels());
  REQUIRE(info.Quality() == jsonInfo.Quality());
  REQUIRE(info.Width() == binaryInfo.Width());
  REQUIRE(info.Height() == binaryInfo.Height());
  REQUIRE(info.Channels() == binaryInfo.Channels());
  REQUIRE(info.Quality() == binaryInfo.Quality());
}

/**
 * Test resize the image if this is done correctly.
 * For now we are not testing if the resize process is correct. We asssume
 * that this is done on STB side. However, it would be really nice to have
 * a KNN or other methods that test the similarity between the pixels and
 * give a score above a threshold. 
 */
TEST_CASE("ImagesResizeTest", "[ImageTest]")
{
  arma::Mat<unsigned char> image, images;
  data::ImageInfo info, resizedInfo, resizedInfo2;
  std::vector<std::string> files =
      {"sheep_1.jpg", "sheep_2.jpg", "sheep_3.jpg", "sheep_4.jpg",
       "sheep_5.jpg", "sheep_6.jpg"};
  std::vector<std::string> reSheeps =
      {"re_sheep_1.jpg", "re_sheep_2.jpg", "re_sheep_3.jpg", "re_sheep_4.jpg",
       "re_sheep_5.jpg", "re_sheep_6.jpg"};
  std::vector<std::string> smSheeps =
      {"sm_sheep_1.jpg", "sm_sheep_2.jpg", "sm_sheep_3.jpg", "sm_sheep_4.jpg",
       "sm_sheep_5.jpg", "sm_sheep_6.jpg"};

  // Load and Resize each one of them individually, because they do not have
  // the same sizes, and then the resized images, will be used in the next
  // test.
  for (size_t i = 0; i < files.size(); i++)
  {
    REQUIRE(data::Load(files.at(i), image, info, false) == true);
    Resize(image, info, 320, 320);
    REQUIRE(data::Save(reSheeps.at(i), image, info, false) == true);
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

#endif
