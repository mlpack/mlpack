/**
 * @file image_load_test.cpp
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
  BOOST_REQUIRE_THROW(data::Load("invalidExtendion.p4ng", matrix, info,
    false, true),  std::runtime_error);
}

/**
 * Test that the image is loaded correctly into the matrix using the API.
 */
BOOST_AUTO_TEST_CASE(LoadImageAPITest)
{
  arma::Mat<unsigned char> matrix;
  data::ImageInfo info;
  BOOST_REQUIRE(data::Load("test_image.png", matrix, info, false,
      true) == true);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 50 * 50 * 3); // width * height * channels.
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
  BOOST_REQUIRE(data::Save("APITest.bmp", im1, info, false, true) == true);

  arma::Mat<unsigned char> im2;
  BOOST_REQUIRE(data::Load("APITest.bmp", im2, info, false, true) == true);

  BOOST_REQUIRE_EQUAL(im1.n_cols, im2.n_cols);
  BOOST_REQUIRE_EQUAL(im1.n_rows, im2.n_rows);
  for (size_t i = 10; i < im1.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(im1[i], im2[i]);
}

BOOST_AUTO_TEST_SUITE_END();

#endif // HAS_STB.
