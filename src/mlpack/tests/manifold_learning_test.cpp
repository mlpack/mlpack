/**
 * @file manifold_learning_test.cpp
 * @author Shangtong Zhang
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/manifold_learning/manifold_learning.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace manifold;

BOOST_AUTO_TEST_SUITE(ManifoldLearningTest);

// Test rules for generate similarity mat
BOOST_AUTO_TEST_CASE(SimilarityRuleTest)
{
  // Generate test data randomly
  arma::mat data;
  size_t nData = 6;
  size_t nDim = 5;
  data.randu(nDim, nData);
  
  // Generate correct answer by simplest approach
  arma::mat kAnswer;
  size_t nNeighbors = 4;
  kAnswer.zeros(nData, nData);
  for (size_t i = 0; i < nData; ++i)
  {
    arma::colvec dist(nData);
    for (size_t j = 0; j < nData; ++j)
    {
      dist(j) = metric::EuclideanDistance::Evaluate(
          data.unsafe_col(i), data.unsafe_col(j));
    }
    arma::uvec ind = arma::sort_index(dist, "ascend");
    for (size_t j = 1; j < nNeighbors + 1; ++j)
        kAnswer(i, ind(j)) = dist(ind(j));
  }
  
  // Generate result using similarity rule class
  manifold::KSimilarity<false, metric::EuclideanDistance, arma::mat>
      kBuilder(nNeighbors);
  arma::mat kResult = kBuilder.BuildSimilarityMat(data);
  BOOST_REQUIRE_EQUAL(0, arma::accu(arma::abs(kAnswer - kResult)));

  // Generate correct answer by simplest approach
  arma::mat epsilonAnswer;
  double epsilon = 1.5;
  epsilonAnswer.zeros(nData, nData);
  for (size_t i = 0; i < nData; ++i)
  {
    arma::colvec dist(nData);
    for (size_t j = 0; j < nData; ++j)
    {
      dist(j) = metric::EuclideanDistance::Evaluate(
          data.unsafe_col(i), data.unsafe_col(j));
    }
    arma::uvec ind = arma::find(dist < epsilon);
    for (size_t j = 0; j < ind.n_elem; ++j)
      epsilonAnswer(i, ind(j)) = dist(ind(j));
  }
  
  // Generate result using similarity rule class
  manifold::EpsilonSimilarity<false, metric::EuclideanDistance, arma::mat>
      epsilonBuilder(epsilon);
  arma::mat epsilonResult = epsilonBuilder.BuildSimilarityMat(data);
  BOOST_REQUIRE_EQUAL(0, arma::accu(arma::abs(epsilonAnswer - epsilonResult)));
  
  /**
   * Test local weight for reconstruction
   * First generate the local weight matrix using the rule class,
   * then disturb the weight matrix randomly.
   * The reconstruction error using original weight matrix should always be
   * less than error using disturbed weight matrix.
   */
  
  size_t nEpoch = 5;
  manifold::KSimilarity<true, metric::EuclideanDistance, arma::mat>
  kWeightBuilder(nNeighbors);
  arma::mat kWeight = kWeightBuilder.BuildSimilarityMat(data);
  for (size_t epoch = 0; epoch < nEpoch; ++epoch)
  {
    arma::mat disturbedKWeight =
        kWeight + arma::randn<arma::mat>(nData, nData);
    for (size_t i = 0; i < nData; ++i)
    {
      arma::colvec v1(nDim);
      v1.zeros();
      arma::colvec v2(nDim);
      v2.zeros();
      for (size_t j = 0; j < nData; ++j)
      {
        if (j != i)
        {
          v1 += kWeight(i, j) * data.unsafe_col(j);
          v2 += disturbedKWeight(i, j) * data.unsafe_col(j);
        }
      }
      double err1 = metric::EuclideanDistance::Evaluate(v1, data.unsafe_col(i));
      double err2 = metric::EuclideanDistance::Evaluate(v2, data.unsafe_col(i));
      BOOST_REQUIRE_LE(err1, err2);
    }
  }
  
  manifold::EpsilonSimilarity<true, metric::EuclideanDistance, arma::mat>
      epsilonWeightBuilder(epsilon);
  arma::mat epsilonWeight = epsilonWeightBuilder.BuildSimilarityMat(data);
  for (size_t epoch = 0; epoch < nEpoch; ++epoch)
  {
    arma::mat disturbedEpsilonWeight =
        epsilonWeight + arma::randn<arma::mat>(nData, nData);
    for (size_t i = 0; i < nData; ++i)
    {
      arma::colvec v1(nDim);
      v1.zeros();
      arma::colvec v2(nDim);
      v2.zeros();
      for (size_t j = 0; j < nData; ++j)
      {
        if (j != i)
        {
          v1 += epsilonWeight(i, j) * data.unsafe_col(j);
          v2 += disturbedEpsilonWeight(i, j) * data.unsafe_col(j);
        }
      }
      double err1 = metric::EuclideanDistance::Evaluate(v1, data.unsafe_col(i));
      double err2 = metric::EuclideanDistance::Evaluate(v2, data.unsafe_col(i));
      BOOST_REQUIRE_LE(err1, err2);
    }
  }
}

// Test MDS
BOOST_AUTO_TEST_CASE(MDSTest)
{
  
  arma::mat tdata("-4 34 7 10;"
                 "23 35 13 6;"
                 "14 -44 18 7;"
                  "2 5 6 7;"
                  "15 25 12 19;"
                  "56 90 23 4;"
                  "12 14 19 80;");
  arma::mat data = tdata.rows(0, tdata.n_rows - 2).t();
  arma::colvec newVec = tdata.row(tdata.n_rows - 1).t();
  MDS<> mds(data, 2);
  arma::mat embedding;
  mds.Transform(embedding);
  
  arma::colvec newEmbedding;
  mds.Transform(newVec, newEmbedding);
  
  // calculate the kruskal stress
  MDSSimilarity<> similarity;
  arma::mat simMatRaw = similarity.BuildSimilarityMat(data);
  arma::mat simMatEmbedding = similarity.BuildSimilarityMat(embedding);
  double kruskal = arma::accu(simMatRaw - simMatEmbedding) / arma::accu(simMatRaw);
  BOOST_REQUIRE_LE(kruskal, 0.025);
  
}

// Test shortest path solver
BOOST_AUTO_TEST_CASE(ShortestPathTest)
{
  // a small symmetrical graph
  arma::mat graph("0 4 5 6 0 0;"
                  "0 0 6 0 3 5;"
                  "0 0 0 9 0 0;"
                  "0 0 0 0 4 1;"
                  "0 0 0 0 0 5;"
                  "0 0 0 0 0 0;");
  graph = graph + graph.t();
  arma::mat dist("0 4 5 6 7 7;"
                 "4 0 6 6 3 5;"
                 "5 6 0 9 9 10;"
                 "6 6 9 0 4 1;"
                 "7 3 9 4 0 5;"
                 "7 5 10 1 5 0;");
  
  arma::mat floydDist;
  FloydWarshall<arma::mat>::Solve(graph, floydDist);
  arma::mat dijkstraDist;
  Dijkstra<arma::mat>::Solve(graph, dijkstraDist);
  
  BOOST_REQUIRE_EQUAL(0, arma::accu(arma::abs(dist - floydDist)));
  BOOST_REQUIRE_EQUAL(0, arma::accu(arma::abs(dist - dijkstraDist)));
  
  // randomly generate a big asymmetrical graph
  arma::mat bigGraph;
  bigGraph.randn(400, 400);
  bigGraph = arma::abs(bigGraph);
  bigGraph.diag().zeros();
  FloydWarshall<arma::mat>::Solve(bigGraph, floydDist);
  Dijkstra<arma::mat>::Solve(bigGraph, dijkstraDist);
  BOOST_REQUIRE_LE(arma::accu(arma::abs(dijkstraDist - floydDist)), 1e-5);
}

/**
 * Test Isomap, LLE, LE on swissroll dataset.
 * The swissroll dataset is generated according to
 * http://people.cs.uchicago.edu/~dinoj/manifold/swissroll.html
 * Difference is that swissroll.dat contains 400 points totally
 * rather than 1600 points.
 * Each algorithm will generate a xxx.dat file containing corresponding 
 * embedding vectors.
 * These embedding vectors can be visualized in matlab using the following code:
 * @code
 * color = ['r' 'b' 'g' 'y'];
 * load('lle.dat');
 * lle = lle';
 * X = lle(:,1);
 * Y = lle(:,2);
 * step = 100;
 * sp = 1;
 * ep = step;
 * figure
 * for i = 1 : 4
 *    scatter(X(sp:ep,:), Y(sp:ep,:), color(i));
 *    hold on
 *    sp = sp + step;
 *    ep = ep + step;
 * end
 * hold off
 * @endcode
 */
BOOST_AUTO_TEST_CASE(SwissrollTest)
{
  arma::mat rawData;
  arma::mat data;
  arma::colvec newVec;
  rawData.load("swissroll.dat");
//  data = rawData.rows(0, rawData.n_rows - 2).t();
  data = rawData.t();
  newVec = rawData.row(rawData.n_rows - 1).t();
  size_t dim = 2;
  
  arma::mat lleEmbedding;
  LLE<> lle(data, dim);
  lle.Transform(lleEmbedding);
  lleEmbedding.save("lle.dat", arma::raw_ascii);
  arma::colvec newLLEEmbedding;
  lle.Transform(newVec, newLLEEmbedding);
  
  arma::mat leEmbedding;
  LE<> le(data, dim);
  le.Transform(leEmbedding);
  leEmbedding.save("le.dat", arma::raw_ascii);
  arma::colvec newLEEmbedding;
  le.Transform(newVec, newLEEmbedding);

  arma::mat isomapEmbedding;
  Isomap<> isomap(data, dim);
  isomap.Transform(isomapEmbedding);
  isomapEmbedding.save("isomap.dat", arma::raw_ascii);
  arma::colvec newIsomapEmbedding;
  isomap.Transform(newVec, newIsomapEmbedding);
  
}


BOOST_AUTO_TEST_SUITE_END();