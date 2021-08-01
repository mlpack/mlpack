
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_image_load_test.cpp:

Program Listing for File image_load_test.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_image_load_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/image_load_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include "serialization.hpp"
   #include "test_catch_tools.hpp"
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::data;
   using namespace std;
   
   #ifdef HAS_STB // Compile this only if stb is present.
   
   TEST_CASE("LoadInvalidExtensionFile", "[ImageLoadTest]")
   {
     arma::Mat<unsigned char> matrix;
     data::ImageInfo info;
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(data::Load("invalidExtendion.p4ng", matrix, info,
         true),  std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
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
   
   TEST_CASE("SaveImageWrongInfo", "[ImageLoadTest]")
   {
     data::ImageInfo info(5, 5, 3, 90);
   
     arma::Mat<unsigned char> im1;
     im1 = arma::randi<arma::Mat<unsigned char>>(24 * 25 * 7, 1);
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(data::Save("APITest.bmp", im1, info, false),
         std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
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
   
   TEST_CASE("SaveImageMatAPITest", "[ImageLoadTest]")
   {
     data::ImageInfo info(5, 5, 3);
   
     arma::Mat<unsigned char> im1;
     size_t dimension = info.Width() * info.Height() * info.Channels();
     im1 = arma::randi<arma::Mat<unsigned char>>(dimension, 1);
     arma::mat input = arma::conv_to<arma::mat>::from(im1);
     REQUIRE(Save("APITest.bmp", input, info, false) == true);
   
     arma::mat output;
     REQUIRE(Load("APITest.bmp", output, info, false) == true);
   
     REQUIRE(input.n_cols == output.n_cols);
     REQUIRE(input.n_rows == output.n_rows);
     for (size_t i = 0; i < input.n_elem; ++i)
       REQUIRE(input[i] == Approx(output[i]).epsilon(1e-7));
     remove("APITest.bmp");
   }
   
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
   
   #endif // HAS_STB.
