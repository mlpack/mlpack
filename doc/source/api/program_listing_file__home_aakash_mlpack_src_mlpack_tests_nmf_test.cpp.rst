
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_nmf_test.cpp:

Program Listing for File nmf_test.cpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_nmf_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/nmf_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/amf/amf.hpp>
   #include <mlpack/methods/amf/init_rules/random_acol_init.hpp>
   #include <mlpack/methods/amf/init_rules/given_init.hpp>
   #include <mlpack/methods/amf/update_rules/nmf_mult_div.hpp>
   #include <mlpack/methods/amf/update_rules/nmf_als.hpp>
   #include <mlpack/methods/amf/update_rules/nmf_mult_dist.hpp>
   
   #include "catch.hpp"
   
   using namespace std;
   using namespace arma;
   using namespace mlpack;
   using namespace mlpack::amf;
   
   TEST_CASE("NMFDefaultTest", "[NMFTest]")
   {
     mat w = randu<mat>(20, 12);
     mat h = randu<mat>(12, 20);
     mat v = w * h;
     size_t r = 12;
   
     AMF<> nmf;
     nmf.Apply(v, r, w, h);
   
     mat wh = w * h;
   
     // Make sure reconstruction error is not too high.  5.0% tolerance.
     REQUIRE(arma::norm(v - wh, "fro") / arma::norm(v, "fro") ==
         Approx(0.0).margin(0.05));
   }
   
   TEST_CASE("NMFAcolDistTest", "[NMFTest]")
   {
     mat w = randu<mat>(20, 12);
     mat h = randu<mat>(12, 20);
     mat v = w * h;
     const size_t r = 12;
   
     SimpleResidueTermination srt(1e-7, 10000);
     AMF<SimpleResidueTermination, RandomAcolInitialization<> > nmf(srt);
     nmf.Apply(v, r, w, h);
   
     mat wh = w * h;
   
     REQUIRE(arma::norm(v - wh, "fro") / arma::norm(v, "fro") ==
         Approx(0.0).margin(0.15));
   }
   
   TEST_CASE("NMFRandomDivTest", "[NMFTest]")
   {
     mat w = randu<mat>(20, 12);
     mat h = randu<mat>(12, 20);
     mat v = w * h;
     size_t r = 12;
   
     const size_t trials = 3;
     bool success = false;
   
     for (size_t trial = 0; trial < trials; ++trial)
     {
       // Custom tighter tolerance.
       SimpleResidueTermination srt(1e-8, 10000);
       AMF<SimpleResidueTermination,
           RandomInitialization,
           NMFMultiplicativeDivergenceUpdate> nmf(srt);
       nmf.Apply(v, r, w, h);
   
       mat wh = w * h;
   
       // Make sure reconstruction error is not too high.  1.5% tolerance.
       if ((arma::norm(v - wh, "fro") / arma::norm(v, "fro")) < 0.015)
       {
         success = true;
         break;
       }
     }
   
     REQUIRE(success == true);
   }
   
   TEST_CASE("NMFALSTest", "[NMFTest]")
   {
     mat w = randu<mat>(20, 12);
     mat h = randu<mat>(12, 20);
     mat v = w * h;
     size_t r = 12;
   
     SimpleResidueTermination srt(1e-12, 50000);
     AMF<SimpleResidueTermination, RandomAcolInitialization<>, NMFALSUpdate>
           nmf(srt);
     nmf.Apply(v, r, w, h);
   
     const mat wh = w * h;
   
     // Make sure reconstruction error is not too high.  9% tolerance.  It seems
     // like ALS doesn't converge to results that are as good.  It also seems to be
     // particularly sensitive to initial conditions.
     REQUIRE(arma::norm(v - wh, "fro") / arma::norm(v, "fro") ==
         Approx(0.0).margin(0.09));
   }
   
   TEST_CASE("SparseNMFAcolDistTest", "[NMFTest]")
   {
     // We have to ensure that the residues aren't NaNs.  This can happen when a
     // matrix is created with all zeros in a column or row.
     double denseResidue = std::numeric_limits<double>::quiet_NaN();
     double sparseResidue = std::numeric_limits<double>::quiet_NaN();
   
     mat vp, dvp; // Resulting matrices.
   
     while (sparseResidue != sparseResidue && denseResidue != denseResidue)
     {
       mat w, h;
       sp_mat v;
       v.sprandu(20, 20, 0.3);
       // Ensure there is at least one nonzero element in every row and column.
       for (size_t i = 0; i < 20; ++i)
         v(i, i) += 1e-5;
       mat dv(v); // Make a dense copy.
       mat dw, dh;
       size_t r = 15;
   
       SimpleResidueTermination srt(1e-10, 10000);
   
       // Get an initialization.
       arma::mat iw, ih;
       RandomAcolInitialization<>::Initialize(v, r, iw, ih);
       GivenInitialization g(std::move(iw), std::move(ih));
   
       // The GivenInitialization will force the same initialization for both
       // Apply() calls.
       AMF<SimpleResidueTermination, GivenInitialization> nmf(srt, g);
       nmf.Apply(v, r, w, h);
       nmf.Apply(dv, r, dw, dh);
   
       // Reconstruct matrices.
       vp = w * h;
       dvp = dw * dh;
   
       denseResidue = arma::norm(v - vp, "fro");
       sparseResidue = arma::norm(dv - dvp, "fro");
     }
   
     // Make sure the results are about equal for the W and H matrices.
     REQUIRE(arma::norm(vp - dvp, "fro") / arma::norm(vp, "fro") ==
         Approx(0.0).margin(1e-5));
   }
   
   TEST_CASE("SparseNMFALSTest", "[NMFTest]")
   {
     // We have to ensure that the residues aren't NaNs.  This can happen when a
     // matrix is created with all zeros in a column or row.
     double denseResidue = std::numeric_limits<double>::quiet_NaN();
     double sparseResidue = std::numeric_limits<double>::quiet_NaN();
   
     mat vp, dvp; // Resulting matrices.
   
     bool success = false;
     for (size_t trial = 0; trial < 3; ++trial)
     {
       while (sparseResidue != sparseResidue && denseResidue != denseResidue)
       {
         mat w, h;
         sp_mat v;
         v.sprandu(10, 10, 0.3);
         // Ensure there is at least one nonzero element in every row and column.
         for (size_t i = 0; i < 10; ++i)
           v(i, i) += 1e-5;
         mat dv(v); // Make a dense copy.
         mat dw, dh;
         size_t r = 5;
   
         // Get an initialization.
         arma::mat iw, ih;
         RandomAcolInitialization<>::Initialize(v, r, iw, ih);
         GivenInitialization g(std::move(iw), std::move(ih));
   
         SimpleResidueTermination srt(1e-10, 10000);
         AMF<SimpleResidueTermination, GivenInitialization, NMFALSUpdate> nmf(srt,
             g);
         nmf.Apply(v, r, w, h);
         nmf.Apply(dv, r, dw, dh);
   
         // Reconstruct matrices.
         vp = w * h; // In general vp won't be sparse.
         dvp = dw * dh;
   
         denseResidue = arma::norm(v - vp, "fro");
         sparseResidue = arma::norm(dv - dvp, "fro");
       }
   
       // Make sure the results are about equal for the W and H matrices.
       const double relDiff = arma::norm(vp - dvp, "fro") / arma::norm(vp, "fro");
       if (relDiff < 1e-5)
       {
         success = true;
         break;
       }
     }
   
     REQUIRE(success == true);
   }
   
   TEST_CASE("NonNegNMFDefaultTest", "[NMFTest]")
   {
     mat w = randu<mat>(20, 12);
     mat h = randu<mat>(12, 20);
     mat v = w * h;
     const size_t r = 12;
   
     AMF<> nmf;
     nmf.Apply(v, r, w, h);
   
     REQUIRE((arma::all(arma::vectorise(w) >= 0)
         && arma::all(arma::vectorise(h) >= 0)));
   }
   
   TEST_CASE("NonNegNMFRandomDivTest", "[NMFTest]")
   {
     mat w = randu<mat>(20, 12);
     mat h = randu<mat>(12, 20);
     mat v = w * h;
     const size_t r = 12;
   
     // Custom tighter tolerance.
     SimpleResidueTermination srt(1e-8, 10000);
     AMF<SimpleResidueTermination,
         RandomInitialization,
         NMFMultiplicativeDivergenceUpdate> nmf(srt);
     nmf.Apply(v, r, w, h);
   
     REQUIRE((arma::all(arma::vectorise(w) >= 0)
         && arma::all(arma::vectorise(h) >= 0)));
   }
   
   TEST_CASE("NonNegNMFALSTest", "[NMFTest]")
   {
     mat w = randu<mat>(20, 12);
     mat h = randu<mat>(12, 20);
     mat v = w * h;
     const size_t r = 12;
   
     SimpleResidueTermination srt(1e-12, 50000);
     AMF<SimpleResidueTermination,
         RandomAcolInitialization<>,
         NMFALSUpdate> nmf(srt);
     nmf.Apply(v, r, w, h);
   
     REQUIRE((arma::all(arma::vectorise(w) >= 0)
         && arma::all(arma::vectorise(h) >= 0)));
   }
