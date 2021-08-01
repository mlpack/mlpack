
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_matrix_completion_test.cpp:

Program Listing for File matrix_completion_test.cpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_matrix_completion_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/matrix_completion_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/matrix_completion/matrix_completion.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::matrix_completion;
   
   TEST_CASE("UniformMatrixCompletionSDP", "[MatrixCompletionTest]")
   {
     arma::mat Xorig, values;
     arma::umat indices;
   
     if (!data::Load("completion_X.csv", Xorig, false, false))
       FAIL("Cannot load dataset completion_X.csv");
     if (!data::Load("completion_indices.csv", indices, false, false))
       FAIL("Cannot load dataset completion_indices.csv");
   
     values.set_size(indices.n_cols);
     for (size_t i = 0; i < indices.n_cols; ++i)
     {
       values(i) = Xorig(indices(0, i), indices(1, i));
     }
   
     arma::mat recovered;
     MatrixCompletion mc(Xorig.n_rows, Xorig.n_cols, indices, values);
     mc.Recover(recovered);
   
     const double err =
       arma::norm(Xorig - recovered, "fro") /
       arma::norm(Xorig, "fro");
     REQUIRE(err == Approx(0.0).margin(1e-5));
   
     for (size_t i = 0; i < indices.n_cols; ++i)
     {
       REQUIRE(recovered(indices(0, i), indices(1, i)) ==
          Approx(Xorig(indices(0, i), indices(1, i))).epsilon(1e-7));
     }
   }
