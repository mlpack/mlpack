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
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

#ifdef HAS_STB // Compile this only if stb is present.

BOOST_AUTO_TEST_SUITE(ImageLoadTest);

/**
 * Test if an image with an unsupported extension throws an expected
 * exception.
 */
BOOST_AUTO_TEST_CASE(LoadInvalidExtensionFile)
{
  arma::Mat<unsigned char> matrix;
  data::ImageInfo info;
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(data::Load("invalidExtendion.p4ng", matrix, info,
      true),  std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Test that the image is loaded correctly into the matrix using the API.
 */
BOOST_AUTO_TEST_CASE(LoadImageAPITest)
{
  arma::Mat<unsigned char> matrix;
  data::ImageInfo info;
  BOOST_REQUIRE(data::Load("test_image.png", matrix, info, false) == true);
  // width * height * channels.
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 50 * 50 * 3);
  BOOST_REQUIRE_EQUAL(info.Height(), 50);
  BOOST_REQUIRE_EQUAL(info.Width(), 50);
  BOOST_REQUIRE_EQUAL(info.Channels(), 3);
  BOOST_REQUIRE_EQUAL(matrix.n_cols, 1);
}

/**
 * Test if the image is saved correctly using API.
 */
BOOST_AUTO_TEST_CASE(SaveImageAPITest)
{
  data::ImageInfo info(5, 5, 3, 90);

  arma::Mat<unsigned char> im1;
  size_t dimension = info.Width() * info.Height() * info.Channels();
  im1 = arma::randi<arma::Mat<unsigned char>>(dimension, 1);
  BOOST_REQUIRE(data::Save("APITest.bmp", im1, info, false) == true);

  arma::Mat<unsigned char> im2;
  BOOST_REQUIRE(data::Load("APITest.bmp", im2, info, false) == true);

  BOOST_REQUIRE_EQUAL(im1.n_cols, im2.n_cols);
  BOOST_REQUIRE_EQUAL(im1.n_rows, im2.n_rows);
  for (size_t i = 0; i < im1.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(im1[i], im2[i]);
  remove("APITest.bmp");
}

/**
 * Test if an image with a wrong dimesion throws an expected
 * exception while saving.
 */
BOOST_AUTO_TEST_CASE(SaveImageWrongInfo)
{
  data::ImageInfo info(5, 5, 3, 90);

  arma::Mat<unsigned char> im1;
  size_t dimension = info.Width() * info.Height() * info.Channels();
  im1 = arma::randi<arma::Mat<unsigned char>>(24 * 25 * 7, 1);
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(data::Save("APITest.bmp", im1, info, false),
      std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Test that the image is loaded correctly into the matrix using the API
 * for vectors.
 */
BOOST_AUTO_TEST_CASE(LoadVectorImageAPITest)
{
  arma::Mat<unsigned char> matrix;
  data::ImageInfo info;
  std::vector<std::string> files = {"test_image.png", "test_image.png"};
  BOOST_REQUIRE(data::Load(files, matrix, info, false) == true);
  // width * height * channels.
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 50 * 50 * 3);
  BOOST_REQUIRE_EQUAL(info.Height(), 50);
  BOOST_REQUIRE_EQUAL(info.Width(), 50);
  BOOST_REQUIRE_EQUAL(info.Channels(), 3);
  BOOST_REQUIRE_EQUAL(matrix.n_cols, 2);
}

/**
 * Test if the image is saved correctly using API for arma mat.
 */
BOOST_AUTO_TEST_CASE(SaveImageMatAPITest)
{
  data::ImageInfo info(5, 5, 3);

  arma::Mat<unsigned char> im1;
  size_t dimension = info.Width() * info.Height() * info.Channels();
  im1 = arma::randi<arma::Mat<unsigned char>>(dimension, 1);
  arma::mat input = arma::conv_to<arma::mat>::from(im1);
  BOOST_REQUIRE(Save("APITest.bmp", input, info, false) == true);

  arma::mat output;
  BOOST_REQUIRE(Load("APITest.bmp", output, info, false) == true);

  BOOST_REQUIRE_EQUAL(input.n_cols, output.n_cols);
  BOOST_REQUIRE_EQUAL(input.n_rows, output.n_rows);
  for (size_t i = 0; i < input.n_elem; ++i)
    BOOST_REQUIRE_CLOSE(input[i], output[i], 1e-5);
  remove("APITest.bmp");
}

/**
 * Serialization test for the ImageInfo class.
 */
BOOST_AUTO_TEST_CASE(ImageInfoSerialization)
{
  data::ImageInfo info(5, 5, 3, 90);
  data::ImageInfo xmlInfo, textInfo, binaryInfo;

  SerializeObjectAll(info, xmlInfo, textInfo, binaryInfo);

  BOOST_REQUIRE_EQUAL(info.Width(), xmlInfo.Width());
  BOOST_REQUIRE_EQUAL(info.Height(), xmlInfo.Height());
  BOOST_REQUIRE_EQUAL(info.Channels(), xmlInfo.Channels());
  BOOST_REQUIRE_EQUAL(info.Quality(), xmlInfo.Quality());
  BOOST_REQUIRE_EQUAL(info.Width(), textInfo.Width());
  BOOST_REQUIRE_EQUAL(info.Height(), textInfo.Height());
  BOOST_REQUIRE_EQUAL(info.Channels(), textInfo.Channels());
  BOOST_REQUIRE_EQUAL(info.Quality(), textInfo.Quality());
  BOOST_REQUIRE_EQUAL(info.Width(), binaryInfo.Width());
  BOOST_REQUIRE_EQUAL(info.Height(), binaryInfo.Height());
  BOOST_REQUIRE_EQUAL(info.Channels(), binaryInfo.Channels());
  BOOST_REQUIRE_EQUAL(info.Quality(), binaryInfo.Quality());
}

BOOST_AUTO_TEST_SUITE_END();

#endif // HAS_STB.
