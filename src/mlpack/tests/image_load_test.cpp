/**
 * @file image_load_test.cpp
 * @author Mehul Kumar Nirala
 *
 * Tests for data::LoadImage().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <sstream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/load_image.hpp>
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace std;

BOOST_AUTO_TEST_SUITE(ImageLoadTest);

/**
 * Test the invalid extension.
 */
BOOST_AUTO_TEST_CASE(LoadInvalidExtensionFile)
{
  data::LoadImage loader;
  arma::Mat<unsigned char> img;
  BOOST_REQUIRE_THROW(loader.Load("invalidExtendion.p4ng", true,
      std::move(img)),  std::runtime_error);
}

/**
 * Test the images is loaded correctly into the matrix.
 */
BOOST_AUTO_TEST_CASE(LoadImageIntoMatrixFromFile)
{
  data::LoadImage loader;

  // Matrix to load contents of file into.
  arma::Mat<unsigned char> img;
  BOOST_REQUIRE(loader.Load("test_image.png", true,  std::move(img)) == true);
  BOOST_REQUIRE_EQUAL(img.n_cols, 50 * 50 * 3); // width * height * channels
  BOOST_REQUIRE_EQUAL(img.n_rows, 1);
}

#if defined(__cplusplus) && __cplusplus >= 201703L && defined(__has_include)
    && __has_include(<filesystem>)
/**
 * Test all the images in dir are loaded correctly.
 */
BOOST_AUTO_TEST_CASE(LoadImageIntoMatrixFromDir)
{
  data::LoadImage loader;

  // Matrix to load contents of dir into.
  arma::Mat<unsigned char> img;
  BOOST_REQUIRE(loader.LoadDir(".", true, std::move(img)) == true);
  BOOST_REQUIRE_EQUAL(img.n_cols, 50 * 50 * 3); // width * height * channels.
  BOOST_REQUIRE_EQUAL(img.n_rows, 1);
}
#endif

/**
 * Test height, width and channels are correctly loaded.
 */
BOOST_AUTO_TEST_CASE(GetImageHeightWidthChannels)
{
  data::LoadImage loader;

  // Matrix to load contents of dir into.
  arma::Mat<unsigned char> img;
  size_t height, width, channels;

  BOOST_REQUIRE(loader.Load("test_image.png", true, std::move(img),
      width, height, channels) == true);
  BOOST_REQUIRE_EQUAL(img.n_cols, 50 * 50 * 3); // width * height * channels.
  BOOST_REQUIRE_EQUAL(img.n_rows, 1);
  BOOST_REQUIRE_EQUAL(width, 50);
  BOOST_REQUIRE_EQUAL(height, 50);
  BOOST_REQUIRE_EQUAL(channels, 3);
}

/**
 * Test loading images from a vector containing image file names.
 */
BOOST_AUTO_TEST_CASE(LoadImagesInVector)
{
  data::LoadImage loader;

  // Matrix to load contents of dir into.
  arma::Mat<unsigned char> img;
  std::vector<std:: string> files{"test_image.png",
                                  "test_image.png",
                                  "test_image.png"};
  BOOST_REQUIRE(loader.Load(files, true, std::move(img)) == true);
  BOOST_REQUIRE_EQUAL(img.n_cols, 50 * 50 * 3); // width * height * channels.
  BOOST_REQUIRE_EQUAL(img.n_rows, files.size());
}

BOOST_AUTO_TEST_SUITE_END();
