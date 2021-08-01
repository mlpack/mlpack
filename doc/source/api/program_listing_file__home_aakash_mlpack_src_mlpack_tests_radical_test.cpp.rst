
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_radical_test.cpp:

Program Listing for File radical_test.cpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_radical_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/radical_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/radical/radical.hpp>
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::radical;
   using namespace std;
   using namespace arma;
   
   TEST_CASE("Radical_Test_Radical3D", "[RadicalTest]")
   {
     mat matX;
     if (!data::Load("data_3d_mixed.txt", matX))
       FAIL("Cannot load dataset data_3d_mixed.txt");
   
     Radical rad(0.175, 5, 100, matX.n_rows - 1);
   
     mat matY;
     mat matW;
     rad.DoRadical(matX, matY, matW);
   
     mat matYT = trans(matY);
     double valEst = 0;
   
     for (uword i = 0; i < matYT.n_cols; ++i)
     {
       vec y = vec(matYT.col(i));
       valEst += rad.Vasicek(y);
     }
   
     mat matS;
     if (!data::Load("data_3d_ind.txt", matS))
       FAIL("Cannot load dataset data_3d_ind.txt");
     rad.DoRadical(matS, matY, matW);
   
     matYT = trans(matY);
     double valBest = 0;
   
     for (uword i = 0; i < matYT.n_cols; ++i)
     {
       vec y = vec(matYT.col(i));
       valBest += rad.Vasicek(y);
     }
   
     // Larger tolerance is sometimes needed.
     REQUIRE(valBest == Approx(valEst).epsilon(0.02));
   }
