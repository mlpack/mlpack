
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_akfn_test.cpp:

Program Listing for File akfn_test.cpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_akfn_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/akfn_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/neighbor_search/neighbor_search.hpp>
   #include <mlpack/core/tree/cover_tree.hpp>
   #include "test_catch_tools.hpp"
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::neighbor;
   using namespace mlpack::tree;
   using namespace mlpack::metric;
   using namespace mlpack::bound;
   
   TEST_CASE("AKFNApproxVsExact1", "[AKFNTest]")
   {
     arma::mat dataset;
   
     if (!data::Load("test_data_3_1000.csv", dataset))
       FAIL("Cannot load test dataset test_data_3_1000.csv!");
   
     KFN exact(dataset);
     arma::Mat<size_t> neighborsExact;
     arma::mat distancesExact;
     exact.Search(dataset, 15, neighborsExact, distancesExact);
   
     for (size_t c = 0; c < 4; c++)
     {
       KFN* akfn;
       double epsilon;
   
       switch (c)
       {
         case 0: // Use the dual-tree method with e=0.02.
           epsilon = 0.02;
           break;
         case 1: // Use the dual-tree method with e=0.05.
           epsilon = 0.05;
           break;
         case 2: // Use the dual-tree method with e=0.10.
           epsilon = 0.10;
           break;
         case 3: // Use the dual-tree method with e=0.20.
           epsilon = 0.20;
           break;
       }
   
       // Now perform the actual calculation.
       akfn = new KFN(dataset, DUAL_TREE_MODE, epsilon);
       arma::Mat<size_t> neighborsApprox;
       arma::mat distancesApprox;
       akfn->Search(dataset, 15, neighborsApprox, distancesApprox);
   
       for (size_t i = 0; i < neighborsApprox.n_elem; ++i)
         REQUIRE_RELATIVE_ERR(distancesApprox(i), distancesExact(i), epsilon);
   
       // Clean the memory.
       delete akfn;
     }
   }
   
   TEST_CASE("AKFNApproxVsExact2", "[AKFNTest]")
   {
     arma::mat dataset;
   
     if (!data::Load("test_data_3_1000.csv", dataset))
       FAIL("Cannot load test dataset test_data_3_1000.csv!");
   
     KFN exact(dataset);
     arma::Mat<size_t> neighborsExact;
     arma::mat distancesExact;
     exact.Search(15, neighborsExact, distancesExact);
   
     KFN akfn(dataset, DUAL_TREE_MODE, 0.05);
     arma::Mat<size_t> neighborsApprox;
     arma::mat distancesApprox;
     akfn.Search(15, neighborsApprox, distancesApprox);
   
     for (size_t i = 0; i < neighborsApprox.n_elem; ++i)
       REQUIRE_RELATIVE_ERR(distancesApprox[i], distancesExact[i], 0.05);
   }
   
   TEST_CASE("AKFNSingleTreeVsExact", "[AKFNTest]")
   {
     arma::mat dataset;
   
     if (!data::Load("test_data_3_1000.csv", dataset))
       FAIL("Cannot load test dataset test_data_3_1000.csv!");
   
     KFN exact(dataset);
     arma::Mat<size_t> neighborsExact;
     arma::mat distancesExact;
     exact.Search(15, neighborsExact, distancesExact);
   
     KFN akfn(dataset, SINGLE_TREE_MODE, 0.05);
     arma::Mat<size_t> neighborsApprox;
     arma::mat distancesApprox;
     akfn.Search(15, neighborsApprox, distancesApprox);
   
     for (size_t i = 0; i < neighborsApprox.n_elem; ++i)
       REQUIRE_RELATIVE_ERR(distancesApprox[i], distancesExact[i], 0.05);
   }
   
   TEST_CASE("AKFNSingleCoverTreeTest", "[AKFNTest]")
   {
     arma::mat dataset;
     dataset.randu(75, 1000); // 75 dimensional, 1000 points.
   
     KFN exact(dataset);
     arma::Mat<size_t> neighborsExact;
     arma::mat distancesExact;
     exact.Search(dataset, 15, neighborsExact, distancesExact);
   
     StandardCoverTree<EuclideanDistance, NeighborSearchStat<FurthestNeighborSort>,
         arma::mat> tree(dataset);
   
     NeighborSearch<FurthestNeighborSort, LMetric<2>, arma::mat, StandardCoverTree>
         coverTreeSearch(std::move(tree), SINGLE_TREE_MODE, 0.05);
   
     arma::Mat<size_t> neighborsCoverTree;
     arma::mat distancesCoverTree;
     coverTreeSearch.Search(dataset, 15, neighborsCoverTree, distancesCoverTree);
   
     for (size_t i = 0; i < neighborsCoverTree.n_elem; ++i)
       REQUIRE_RELATIVE_ERR(distancesCoverTree[i], distancesExact[i], 0.05);
   }
   
   TEST_CASE("AKFNDualCoverTreeTest", "[AKFNTest]")
   {
     arma::mat dataset;
     if (!data::Load("test_data_3_1000.csv", dataset))
       FAIL("Cannot load test dataset test_data_3_1000.csv!");
   
     KFN exact(dataset);
     arma::Mat<size_t> neighborsExact;
     arma::mat distancesExact;
     exact.Search(dataset, 15, neighborsExact, distancesExact);
   
     StandardCoverTree<EuclideanDistance, NeighborSearchStat<FurthestNeighborSort>,
         arma::mat> referenceTree(dataset);
   
     NeighborSearch<FurthestNeighborSort, LMetric<2>, arma::mat, StandardCoverTree>
         coverTreeSearch(std::move(referenceTree), DUAL_TREE_MODE, 0.05);
   
     arma::Mat<size_t> neighborsCoverTree;
     arma::mat distancesCoverTree;
     coverTreeSearch.Search(dataset, 15, neighborsCoverTree, distancesCoverTree);
   
     for (size_t i = 0; i < neighborsCoverTree.n_elem; ++i)
       REQUIRE_RELATIVE_ERR(distancesCoverTree[i], distancesExact[i], 0.05);
   }
   
   TEST_CASE("AKFNSingleBallTreeTest", "[AKFNTest]")
   {
     arma::mat dataset;
     dataset.randu(75, 1000); // 75 dimensional, 1000 points.
   
     KFN exact(dataset);
     arma::Mat<size_t> neighborsExact;
     arma::mat distancesExact;
     exact.Search(dataset, 15, neighborsExact, distancesExact);
   
     NeighborSearch<FurthestNeighborSort, EuclideanDistance, arma::mat, BallTree>
         ballTreeSearch(dataset, SINGLE_TREE_MODE, 0.05);
   
     arma::Mat<size_t> neighborsBallTree;
     arma::mat distancesBallTree;
     ballTreeSearch.Search(dataset, 15, neighborsBallTree, distancesBallTree);
   
     for (size_t i = 0; i < neighborsBallTree.n_elem; ++i)
       REQUIRE_RELATIVE_ERR(distancesBallTree(i), distancesExact(i), 0.05);
   }
   
   TEST_CASE("AKFNDualBallTreeTest", "[AKFNTest]")
   {
     arma::mat dataset;
     if (!data::Load("test_data_3_1000.csv", dataset))
       FAIL("Cannot load test dataset test_data_3_1000.csv!");
   
     KFN exact(dataset);
     arma::Mat<size_t> neighborsExact;
     arma::mat distancesExact;
     exact.Search(15, neighborsExact, distancesExact);
   
     NeighborSearch<FurthestNeighborSort, EuclideanDistance, arma::mat, BallTree>
         ballTreeSearch(dataset, DUAL_TREE_MODE, 0.05);
     arma::Mat<size_t> neighborsBallTree;
     arma::mat distancesBallTree;
     ballTreeSearch.Search(15, neighborsBallTree, distancesBallTree);
   
     for (size_t i = 0; i < neighborsBallTree.n_elem; ++i)
       REQUIRE_RELATIVE_ERR(distancesBallTree(i), distancesExact(i), 0.05);
   }
