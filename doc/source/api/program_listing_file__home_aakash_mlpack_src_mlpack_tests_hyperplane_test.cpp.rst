
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_hyperplane_test.cpp:

Program Listing for File hyperplane_test.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_hyperplane_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/hyperplane_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/core/tree/space_split/hyperplane.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::math;
   using namespace mlpack::tree;
   using namespace mlpack::metric;
   using namespace mlpack::bound;
   
   TEST_CASE("HyperplaneEmptyConstructor", "[HyperplaneTest]")
   {
     Hyperplane<EuclideanDistance> h1;
     AxisOrthogonalHyperplane<EuclideanDistance> h2;
   
     arma::mat dataset;
     dataset.randu(3, 20); // 20 points in 3 dimensions.
   
     for (size_t i = 0; i < dataset.n_cols; ++i)
     {
       REQUIRE(h1.Left(dataset.col(i)));
       REQUIRE(h2.Left(dataset.col(i)));
       REQUIRE(!h1.Right(dataset.col(i)));
       REQUIRE(!h2.Right(dataset.col(i)));
     }
   }
   
   TEST_CASE("ProjectionTest", "[HyperplaneTest]")
   {
     // General hyperplane.
     ProjVector projVect1(arma::vec("1 1"));
     Hyperplane<EuclideanDistance> h1(projVect1, 0);
   
     REQUIRE(h1.Project(arma::vec("1 -1")) == 0);
     REQUIRE(h1.Left(arma::vec("1 -1")));
     REQUIRE(!h1.Right(arma::vec("1 -1")));
   
     REQUIRE(h1.Project(arma::vec("-1 1")) == 0);
     REQUIRE(h1.Left(arma::vec("-1 1")));
     REQUIRE(!h1.Right(arma::vec("-1 1")));
   
     REQUIRE(h1.Project(arma::vec("1 0")) == h1.Project(arma::vec("0 1")));
     REQUIRE(h1.Right(arma::vec("1 0")));
     REQUIRE(!h1.Left(arma::vec("1 0")));
   
     REQUIRE(h1.Project(arma::vec("-1 -1")) == h1.Project(arma::vec("-2 0")));
     REQUIRE(h1.Left(arma::vec("-1 -1")));
     REQUIRE(!h1.Right(arma::vec("-1 -1")));
   
     // A simple 2-dimensional bound.
     BallBound<EuclideanDistance> b1(2);
   
     b1.Center() = arma::vec("-1 -1");
     b1.Radius() = 1.41;
     REQUIRE(h1.Left(b1));
     REQUIRE(!h1.Right(b1));
   
     b1.Center() = arma::vec("1 1");
     b1.Radius() = 1.41;
     REQUIRE(h1.Right(b1));
     REQUIRE(!h1.Left(b1));
   
     b1.Center() = arma::vec("0 0");
     b1.Radius() = 1.41;
     REQUIRE(!h1.Right(b1));
     REQUIRE(!h1.Left(b1));
   }
   
   TEST_CASE("AxisOrthogonalProjectionTest", "[HyperplaneTest]")
   {
     // AxisParallel hyperplane.
     AxisParallelProjVector projVect2(1);
     AxisOrthogonalHyperplane<EuclideanDistance> h2(projVect2, 1);
   
     REQUIRE(h2.Project(arma::vec("0 0")) == -1);
     REQUIRE(h2.Left(arma::vec("0 0")));
     REQUIRE(!h2.Right(arma::vec("0 0")));
   
     REQUIRE(h2.Project(arma::vec("0 1")) == 0);
     REQUIRE(h2.Left(arma::vec("0 1")));
     REQUIRE(!h2.Right(arma::vec("0 1")));
   
     REQUIRE(h2.Project(arma::vec("0 2")) == 1);
     REQUIRE(h2.Right(arma::vec("0 2")));
     REQUIRE(!h2.Left(arma::vec("0 2")));
   
     REQUIRE(h2.Project(arma::vec("1 2")) == 1);
     REQUIRE(h2.Right(arma::vec("1 2")));
     REQUIRE(!h2.Left(arma::vec("1 2")));
   
     REQUIRE(h2.Project(arma::vec("1 0")) == -1);
     REQUIRE(h2.Left(arma::vec("1 0")));
     REQUIRE(!h2.Right(arma::vec("1 0")));
   
     // A simple 2-dimensional bound.
     HRectBound<EuclideanDistance> b2(2);
   
     b2[0] = Range(-1.0, 1.0);
     b2[1] = Range(-1.0, 1.0);
     REQUIRE(h2.Left(b2));
     REQUIRE(!h2.Right(b2));
   
     b2[0] = Range(-1.0, 1.0);
     b2[1] = Range(1.001, 2.0);
     REQUIRE(h2.Right(b2));
     REQUIRE(!h2.Left(b2));
   
     b2[0] = Range(-1.0, 1.0);
     b2[1] = Range(0, 2.0);
     REQUIRE(!h2.Right(b2));
     REQUIRE(!h2.Left(b2));
   }
