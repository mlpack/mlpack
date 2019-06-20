/**
 * @file image_load_test.cpp
 * @author Mehul Kumar Nirala
 *
 * Tests for data::Image().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/data/load_image.hpp>
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace std;

#ifdef HAS_STB // Compile this only if stb is present.

BOOST_AUTO_TEST_SUITE(ImageLoadTest);

/**
 * Test the invalid extension.
 */
BOOST_AUTO_TEST_CASE(LoadInvalidExtensionFile)
{
  data::Image loader;
  arma::Mat<unsigned char> img;
  BOOST_REQUIRE_THROW(loader.Load("invalidExtendion.p4ng", true,
      std::move(img)),  std::runtime_error);
}

/**
 * Test the images is loaded correctly into the matrix.
 */
BOOST_AUTO_TEST_CASE(LoadImageIntoMatrixFromFile)
{
  data::Image loader;

  // Matrix to load contents of file into.
  arma::Mat<unsigned char> img;
  BOOST_REQUIRE(loader.Load("test_image.png", true,  std::move(img)) == true);
  BOOST_REQUIRE_EQUAL(img.n_rows, 50 * 50 * 3); // width * height * channels
  BOOST_REQUIRE_EQUAL(img.n_cols, 1);
}

#if defined(__cplusplus) && __cplusplus >= 201703L && defined(__has_include)
    && __has_include(<filesystem>)
/**
 * Test all the images in dir are loaded correctly.
 */
BOOST_AUTO_TEST_CASE(LoadImageIntoMatrixFromDir)
{
  data::Image loader;

  // Matrix to load contents of dir into.
  arma::Mat<unsigned char> img;
  BOOST_REQUIRE(loader.LoadDir(".", true, std::move(img)) == true);
  BOOST_REQUIRE_EQUAL(img.n_rows, 50 * 50 * 3); // width * height * channels.
  BOOST_REQUIRE_EQUAL(img.n_cols, 1);
}
#endif

/**
 * Test height, width and channels are correctly loaded.
 */
BOOST_AUTO_TEST_CASE(GetImageHeightWidthChannels)
{
  data::Image loader;

  // Matrix to load the image.
  arma::Mat<unsigned char> img;
  size_t height, width, channels;

  BOOST_REQUIRE(loader.Load("test_image.png", true, std::move(img),
      width, height, channels) == true);
  BOOST_REQUIRE_EQUAL(img.n_rows, 50 * 50 * 3); // width * height * channels.
  BOOST_REQUIRE_EQUAL(img.n_cols, 1);
  BOOST_REQUIRE_EQUAL(width, 50);
  BOOST_REQUIRE_EQUAL(height, 50);
  BOOST_REQUIRE_EQUAL(channels, 3);
}

/**
 * Test loading images from a vector containing image file names.
 */
BOOST_AUTO_TEST_CASE(LoadImagesInVector)
{
  data::Image loader;

  // Matrix to load contents of dir into.
  arma::Mat<unsigned char> img;
  std::vector<std:: string> files{"test_image.png",
                                  "test_image.png",
                                  "test_image.png"};
  BOOST_REQUIRE(loader.Load(files, true, std::move(img)) == true);
  BOOST_REQUIRE_EQUAL(img.n_rows, 50 * 50 * 3); // width * height * channels.
  BOOST_REQUIRE_EQUAL(img.n_cols, files.size());
}

/**
 * Test if the image is saved and loaded correctly.
 */
BOOST_AUTO_TEST_CASE(SaveAndLoadImageTest)
{
  arma::Mat<unsigned char> img1;
  img1 = arma::randi<arma::Mat<unsigned char>>(25*3, 1);
  data::Image loadAndSave;
  BOOST_REQUIRE(loadAndSave.Save("test.bmp", 5, 5,
      3, true, std::move(img1)) == true);

  arma::Mat<unsigned char> img2;
  size_t height, width, channels = 3;
  BOOST_REQUIRE(loadAndSave.Load("test.bmp", true, std::move(img2),
      width, height, channels) == true);

  BOOST_REQUIRE_EQUAL(img1.n_cols, img2.n_cols);
  BOOST_REQUIRE_EQUAL(img1.n_rows, img2.n_rows);

  for (size_t i = 0; i < img1.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(img1[i], img2[i]);
}

/**
 * Test if the image is saved correctly.
 */
BOOST_AUTO_TEST_CASE(SaveMultipleImageTest)
{
  data::Image loadAndSave;

  arma::Mat<unsigned char> img;
  size_t height, width, channels;
  std::vector<std:: string> files{"test_image.png",
                                  "test_image.png"};

  BOOST_REQUIRE(loadAndSave.Load(files, true, std::move(img)) == true);
  BOOST_REQUIRE_EQUAL(img.n_rows, 50 * 50 * 3); // width * height * channels.
  BOOST_REQUIRE_EQUAL(img.n_cols, files.size());

  std::vector<std:: string> sfiles{"saved_image1.png",
                                   "saved_image2.png"};

  BOOST_REQUIRE(loadAndSave.Save(sfiles, 50, 50, 3, true,
      std::move(img)) == true);

  // Retrieve the images saved and cross-check the dimensions.
  img.clear();
  BOOST_REQUIRE(loadAndSave.Load(sfiles, true, std::move(img)) == true);
  BOOST_REQUIRE_EQUAL(img.n_rows, 50 * 50 * 3); // width * height * channels.
  BOOST_REQUIRE_EQUAL(img.n_cols, sfiles.size());
}

BOOST_AUTO_TEST_SUITE_END();

#endif // HAS_STB.
