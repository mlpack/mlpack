
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_serialization.cpp:

Program Listing for File serialization.cpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_serialization.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/serialization.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "serialization.hpp"
   #include "catch.hpp"
   
   namespace mlpack {
   
   // Utility function to check the equality of two Armadillo matrices.
   void CheckMatrices(const arma::mat& x,
                      const arma::mat& xmlX,
                      const arma::mat& jsonX,
                      const arma::mat& binaryX)
   {
     // First check dimensions.
     REQUIRE(x.n_rows == xmlX.n_rows);
     REQUIRE(x.n_rows == jsonX.n_rows);
     REQUIRE(x.n_rows == binaryX.n_rows);
   
     REQUIRE(x.n_cols == xmlX.n_cols);
     REQUIRE(x.n_cols == jsonX.n_cols);
     REQUIRE(x.n_cols == binaryX.n_cols);
   
     REQUIRE(x.n_elem == xmlX.n_elem);
     REQUIRE(x.n_elem == jsonX.n_elem);
     REQUIRE(x.n_elem == binaryX.n_elem);
   
     // Now check elements.
     for (size_t i = 0; i < x.n_elem; ++i)
     {
       const double val = x[i];
       if (val == 0.0)
       {
         REQUIRE(xmlX[i] == Approx(0.0).margin(1e-6 / 100));
         REQUIRE(jsonX[i] == Approx(0.0).margin(1e-6 / 100));
         REQUIRE(binaryX[i] == Approx(0.0).margin(1e-6 / 100));
       }
       else
       {
         REQUIRE(val == Approx(xmlX[i]).epsilon(1e-6 / 100));
         REQUIRE(val == Approx(jsonX[i]).epsilon(1e-6 / 100));
         REQUIRE(val == Approx(binaryX[i]).epsilon(1e-6 / 100));
       }
     }
   }
   
   void CheckMatrices(const arma::Mat<size_t>& x,
                      const arma::Mat<size_t>& xmlX,
                      const arma::Mat<size_t>& jsonX,
                      const arma::Mat<size_t>& binaryX)
   {
     // First check dimensions.
     REQUIRE(x.n_rows == xmlX.n_rows);
     REQUIRE(x.n_rows == jsonX.n_rows);
     REQUIRE(x.n_rows == binaryX.n_rows);
   
     REQUIRE(x.n_cols == xmlX.n_cols);
     REQUIRE(x.n_cols == jsonX.n_cols);
     REQUIRE(x.n_cols == binaryX.n_cols);
   
     REQUIRE(x.n_elem == xmlX.n_elem);
     REQUIRE(x.n_elem == jsonX.n_elem);
     REQUIRE(x.n_elem == binaryX.n_elem);
   
     // Now check elements.
     for (size_t i = 0; i < x.n_elem; ++i)
     {
       REQUIRE(x[i] == xmlX[i]);
       REQUIRE(x[i] == jsonX[i]);
       REQUIRE(x[i] == binaryX[i]);
     }
   }
   
   void CheckMatrices(const arma::cube& x,
                      const arma::cube& xmlX,
                      const arma::cube& jsonX,
                      const arma::cube& binaryX)
   {
     // First check dimensions.
     REQUIRE(x.n_rows == xmlX.n_rows);
     REQUIRE(x.n_rows == jsonX.n_rows);
     REQUIRE(x.n_rows == binaryX.n_rows);
   
     REQUIRE(x.n_cols == xmlX.n_cols);
     REQUIRE(x.n_cols == jsonX.n_cols);
     REQUIRE(x.n_cols == binaryX.n_cols);
   
     REQUIRE(x.n_slices == xmlX.n_slices);
     REQUIRE(x.n_slices == jsonX.n_slices);
     REQUIRE(x.n_slices == binaryX.n_slices);
   
     REQUIRE(x.n_elem == xmlX.n_elem);
     REQUIRE(x.n_elem == jsonX.n_elem);
     REQUIRE(x.n_elem == binaryX.n_elem);
   
     // Now check elements.
     for (size_t i = 0; i < x.n_elem; ++i)
     {
       REQUIRE(x[i] ==Approx(xmlX[i]));
       REQUIRE(x[i] ==Approx(jsonX[i]));
       REQUIRE(x[i] ==Approx(binaryX[i]));
     }
   }
   
   } // namespace mlpack
