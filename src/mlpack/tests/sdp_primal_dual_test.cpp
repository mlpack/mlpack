/**
 * @file sdp_primal_dual_test.cpp
 * @author Stephen Tu
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sdp/sdp.hpp>
#include <mlpack/core/optimizers/sdp/primal_dual.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::distribution;
using namespace mlpack::neighbor;

class UndirectedGraph
{
 public:

  UndirectedGraph() {}

  size_t NumVertices() const { return numVertices; }
  size_t NumEdges() const { return edges.n_cols; }

  const arma::umat& Edges() const { return edges; }
  const arma::vec& Weights() const { return weights; }

  void Laplacian(arma::sp_mat& laplacian) const
  {
    laplacian.zeros(numVertices, numVertices);

    for (size_t i = 0; i < edges.n_cols; ++i)
    {
      laplacian(edges(0, i), edges(1, i)) = -weights(i);
      laplacian(edges(1, i), edges(0, i)) = -weights(i);
    }

    for (size_t i = 0; i < numVertices; ++i)
    {
      laplacian(i, i) = -arma::accu(laplacian.row(i));
    }
  }

  static void LoadFromEdges(UndirectedGraph& g,
                            const std::string& edgesFilename,
                            bool transposeEdges)
  {
    data::Load(edgesFilename, g.edges, true, transposeEdges);
    if (g.edges.n_rows != 2)
      Log::Fatal << "Invalid datafile" << std::endl;
    g.weights.ones(g.edges.n_cols);
    g.ComputeVertices();
  }

  static void LoadFromEdgesAndWeights(UndirectedGraph& g,
                                      const std::string& edgesFilename,
                                      bool transposeEdges,
                                      const std::string& weightsFilename,
                                      bool transposeWeights)
  {
    data::Load(edgesFilename, g.edges, true, transposeEdges);
    if (g.edges.n_rows != 2)
      Log::Fatal << "Invalid datafile" << std::endl;
    data::Load(weightsFilename, g.weights, true, transposeWeights);
    if (g.weights.n_elem != g.edges.n_cols)
      Log::Fatal << "Size mismatch" << std::endl;
    g.ComputeVertices();
  }

  static void ErdosRenyiRandomGraph(UndirectedGraph& g,
                                    size_t numVertices,
                                    double edgeProbability,
                                    bool weighted,
                                    bool selfLoops = false)
  {
    if (edgeProbability < 0. || edgeProbability > 1.)
      Log::Fatal << "edgeProbability not in [0, 1]" << std::endl;

    std::vector<std::pair<size_t, size_t>> edges;
    std::vector<double> weights;

    for (size_t i = 0; i < numVertices; i ++)
    {
      for (size_t j = (selfLoops ? i : i + 1); j < numVertices; j++)
      {
        if (math::Random() > edgeProbability)
          continue;
        edges.emplace_back(i, j);
        weights.push_back(weighted ? math::Random() : 1.);
      }
    }

    g.edges.set_size(2, edges.size());
    for (size_t i = 0; i < edges.size(); i++)
    {
      g.edges(0, i) = edges[i].first;
      g.edges(1, i) = edges[i].second;
    }
    g.weights = arma::vec(weights);

    g.numVertices = numVertices;
  }

 private:

  void ComputeVertices()
  {
    numVertices = max(max(edges)) + 1;
  }

  arma::umat edges;
  arma::vec weights;
  size_t numVertices;
};

static inline SDP<arma::sp_mat>
ConstructMaxCutSDPFromGraph(const UndirectedGraph& g)
{
  SDP<arma::sp_mat> sdp(g.NumVertices(), g.NumVertices(), 0);
  g.Laplacian(sdp.C());
  sdp.C() *= -1;
  for (size_t i = 0; i < g.NumVertices(); i++)
  {
    sdp.SparseA()[i].zeros(g.NumVertices(), g.NumVertices());
    sdp.SparseA()[i](i, i) = 1.;
  }
  sdp.SparseB().ones();
  return sdp;
}

static inline SDP<arma::mat>
ConstructLovaszThetaSDPFromGraph(const UndirectedGraph& g)
{
  SDP<arma::mat> sdp(g.NumVertices(), g.NumEdges() + 1, 0);
  sdp.C().ones();
  sdp.C() *= -1.;
  sdp.SparseA()[0].eye(g.NumVertices(), g.NumVertices());
  for (size_t i = 0; i < g.NumEdges(); i++)
  {
    sdp.SparseA()[i + 1].zeros(g.NumVertices(), g.NumVertices());
    sdp.SparseA()[i + 1](g.Edges()(0, i), g.Edges()(1, i)) = 1.;
    sdp.SparseA()[i + 1](g.Edges()(1, i), g.Edges()(0, i)) = 1.;
  }
  sdp.SparseB().zeros();
  sdp.SparseB()[0] = 1.;
  return sdp;
}

static inline SDP<arma::sp_mat>
ConstructMaxCutSDPFromLaplacian(const std::string& laplacianFilename)
{
  arma::mat laplacian;
  data::Load(laplacianFilename, laplacian, true, false);
  if (laplacian.n_rows != laplacian.n_cols)
    Log::Fatal << "laplacian not square" << std::endl;
  SDP<arma::sp_mat> sdp(laplacian.n_rows, laplacian.n_rows, 0);
  sdp.C() = -arma::sp_mat(laplacian);
  for (size_t i = 0; i < laplacian.n_rows; i++)
  {
    sdp.SparseA()[i].zeros(laplacian.n_rows, laplacian.n_rows);
    sdp.SparseA()[i](i, i) = 1.;
  }
  sdp.SparseB().ones();
  return sdp;
}

static void CheckPositiveSemiDefinite(const arma::mat& X)
{
  const auto evals = arma::eig_sym(X);
  BOOST_REQUIRE_GE(evals(0), 1e-20);
}

template <typename SDPType>
static void CheckKKT(const SDPType& sdp,
                     const arma::mat& X,
                     const arma::vec& ysparse,
                     const arma::vec& ydense,
                     const arma::mat& Z)
{
  // require that the KKT optimality conditions for sdp are satisfied
  // by the primal-dual pair (X, y, Z)

  CheckPositiveSemiDefinite(X);
  CheckPositiveSemiDefinite(Z);

  const double normXz = arma::norm(X * Z, "fro");
  BOOST_REQUIRE_SMALL(normXz, 1e-5);

  for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
  {
    BOOST_REQUIRE_SMALL(
      fabs(arma::dot(sdp.SparseA()[i], X) - sdp.SparseB()[i]),
      1e-5);
  }

  for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
  {
    BOOST_REQUIRE_SMALL(
      fabs(arma::dot(sdp.DenseA()[i], X) - sdp.DenseB()[i]),
      1e-5);
  }

  arma::mat dualCheck = Z - sdp.C();
  for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
    dualCheck += ysparse(i) * sdp.SparseA()[i];
  for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
    dualCheck += ydense(i) * sdp.DenseA()[i];
  const double dualInfeas = arma::norm(dualCheck, "fro");
  BOOST_REQUIRE_SMALL(dualInfeas, 1e-5);
}

BOOST_AUTO_TEST_SUITE(SdpPrimalDualTest);

static void SolveMaxCutFeasibleSDP(const SDP<arma::sp_mat>& sdp)
{
  arma::mat X0, Z0;
  arma::vec ysparse0, ydense0;
  ydense0.set_size(0);

  // strictly feasible starting point
  X0.eye(sdp.N(), sdp.N());
  ysparse0 = -1.1 * arma::vec(arma::sum(arma::abs(sdp.C()), 0).t());
  Z0 = -arma::diagmat(ysparse0) + sdp.C();

  PrimalDualSolver<SDP<arma::sp_mat>> solver(sdp, X0, ysparse0, ydense0, Z0);

  arma::mat X, Z;
  arma::vec ysparse, ydense;
  solver.Optimize(X, ysparse, ydense, Z);
  CheckKKT(sdp, X, ysparse, ydense, Z);
}

static void SolveMaxCutPositiveSDP(const SDP<arma::sp_mat>& sdp)
{
  arma::mat X0, Z0;
  arma::vec ysparse0, ydense0;
  ydense0.set_size(0);

  // infeasible, but positive starting point
  X0 = arma::eye<arma::mat>(sdp.N(), sdp.N());
  ysparse0 = arma::randu<arma::vec>(sdp.NumSparseConstraints());
  Z0.eye(sdp.N(), sdp.N());

  PrimalDualSolver<SDP<arma::sp_mat>> solver(sdp, X0, ysparse0, ydense0, Z0);

  arma::mat X, Z;
  arma::vec ysparse, ydense;
  solver.Optimize(X, ysparse, ydense, Z);
  CheckKKT(sdp, X, ysparse, ydense, Z);
}

BOOST_AUTO_TEST_CASE(SmallMaxCutSdp)
{
  auto sdp = ConstructMaxCutSDPFromLaplacian("r10.txt");
  SolveMaxCutFeasibleSDP(sdp);
  SolveMaxCutPositiveSDP(sdp);

  UndirectedGraph g;
  UndirectedGraph::ErdosRenyiRandomGraph(g, 10, 0.3, true);
  sdp = ConstructMaxCutSDPFromGraph(g);

  // the following was resulting in non-positive Z0 matrices on some
  // random instances.
  //SolveMaxCutFeasibleSDP(sdp);

  SolveMaxCutPositiveSDP(sdp);
}

BOOST_AUTO_TEST_CASE(SmallLovaszThetaSdp)
{
  UndirectedGraph g;
  UndirectedGraph::LoadFromEdges(g, "johnson8-4-4.csv", true);
  auto sdp = ConstructLovaszThetaSDPFromGraph(g);

  PrimalDualSolver<SDP<arma::mat>> solver(sdp);

  arma::mat X, Z;
  arma::vec ysparse, ydense;
  solver.Optimize(X, ysparse, ydense, Z);
  CheckKKT(sdp, X, ysparse, ydense, Z);
}

static inline arma::sp_mat
RepeatBlockDiag(const arma::sp_mat& block, size_t repeat)
{
  assert(block.n_rows == block.n_cols);
  arma::sp_mat ret(block.n_rows * repeat, block.n_rows * repeat);
  ret.zeros();
  for (size_t i = 0; i < repeat; i++)
    ret(arma::span(i * block.n_rows, (i + 1) * block.n_rows - 1),
        arma::span(i * block.n_rows, (i + 1) * block.n_rows - 1)) = block;
  return ret;
}

static inline arma::sp_mat
BlockDiag(const std::vector<arma::sp_mat>& blocks)
{
  // assumes all blocks are the same size
  const size_t n = blocks.front().n_rows;
  assert(blocks.front().n_cols == n);
  arma::sp_mat ret(n * blocks.size(), n * blocks.size());
  ret.zeros();
  for (size_t i = 0; i < blocks.size(); i++)
    ret(arma::span(i * n, (i + 1) * n - 1),
        arma::span(i * n, (i + 1) * n - 1)) = blocks[i];
  return ret;
}

static inline SDP<arma::sp_mat>
ConstructLogChebychevApproxSdp(const arma::mat& A, const arma::vec& b)
{
  if (A.n_rows != b.n_elem)
    Log::Fatal << "A.n_rows != len(b)" << std::endl;
  const size_t p = A.n_rows;
  const size_t k = A.n_cols;

  // [0, 0, 0]
  // [0, 0, 1]
  // [0, 1, 0]
  arma::sp_mat cblock(3, 3);
  cblock(1, 2) = cblock(2, 1) = 1.;
  const arma::sp_mat C = RepeatBlockDiag(cblock, p);

  SDP<arma::sp_mat> sdp(C.n_rows, k + 1, 0);
  sdp.C() = C;
  sdp.SparseB().zeros();
  sdp.SparseB()[0] = -1;

  // [1, 0, 0]
  // [0, 0, 0]
  // [0, 0, 1]
  arma::sp_mat a0block(3, 3);
  a0block(0, 0) = a0block(2, 2) = 1.;
  sdp.SparseA()[0] = RepeatBlockDiag(a0block, p);
  sdp.SparseA()[0] *= -1.;

  for (size_t i = 0; i < k; i++)
  {
    std::vector<arma::sp_mat> blocks;
    for (size_t j = 0; j < p; j++)
    {
      arma::sp_mat block(3, 3);
      const double f = A(j, i) / b(j);
      // [ -a_j(i)/b_j     0        0 ]
      // [      0       a_j(i)/b_j  0 ]
      // [      0          0        0 ]
      block(0, 0) = -f;
      block(1, 1) = f;
      blocks.emplace_back(block);
    }
    sdp.SparseA()[i + 1] = BlockDiag(blocks);
    sdp.SparseA()[i + 1] *= -1;
  }

  return sdp;
}

static inline arma::mat
RandomOrthogonalMatrix(size_t rows, size_t cols)
{
  arma::mat Q, R;
  if (!arma::qr(Q, R, arma::randu<arma::mat>(rows, cols)))
    Log::Fatal << "could not compute QR decomposition" << std::endl;
  return Q;
}

static inline arma::mat
RandomFullRowRankMatrix(size_t rows, size_t cols)
{
  const arma::mat U = RandomOrthogonalMatrix(rows, rows);
  const arma::mat V = RandomOrthogonalMatrix(cols, cols);
  arma::mat S;
  S.zeros(rows, cols);
  for (size_t i = 0; i < std::min(rows, cols); i++)
  {
    S(i, i) = math::Random() + 1e-3;
  }
  return U * S * V;
}

/**
 * See the examples section, Eq. 9, of
 *
 *   Semidefinite Programming.
 *   Lieven Vandenberghe and Stephen Boyd.
 *   SIAM Review. 1996.
 *
 * The logarithmic Chebychev approximation to Ax = b, A is p x k and b is
 * length p is given by the SDP:
 *
 *   min    t
 *   s.t.
 *          [ t - dot(a_i, x)          0             0 ]
 *          [       0           dot(a_i, x) / b_i    1 ]  >= 0, i=1,...,p
 *          [       0                  1             t ]
 *
 */
BOOST_AUTO_TEST_CASE(LogChebychevApproxSdp)
{
  const size_t p0 = 5;
  const size_t k0 = 10;
  const arma::mat A0 = RandomFullRowRankMatrix(p0, k0);
  const arma::vec b0 = arma::randu<arma::vec>(p0);
  const auto sdp0 = ConstructLogChebychevApproxSdp(A0, b0);
  PrimalDualSolver<SDP<arma::sp_mat>> solver0(sdp0);
  arma::mat X0, Z0;
  arma::vec ysparse0, ydense0;
  solver0.Optimize(X0, ysparse0, ydense0, Z0);
  CheckKKT(sdp0, X0, ysparse0, ydense0, Z0);

  const size_t p1 = 10;
  const size_t k1 = 5;
  const arma::mat A1 = RandomFullRowRankMatrix(p1, k1);
  const arma::vec b1 = arma::randu<arma::vec>(p1);
  const auto sdp1 = ConstructLogChebychevApproxSdp(A1, b1);
  PrimalDualSolver<SDP<arma::sp_mat>> solver1(sdp1);
  arma::mat X1, Z1;
  arma::vec ysparse1, ydense1;
  solver1.Optimize(X1, ysparse1, ydense1, Z1);
  CheckKKT(sdp1, X1, ysparse1, ydense1, Z1);
}

/**
 * Example 1 on the SDP wiki
 *
 *   min   x_13
 *   s.t.
 *         -0.2 <= x_12 <= -0.1
 *          0.4 <= x_23 <=  0.5
 *          x_11 = x_22 = x_33 = 1
 *          X >= 0
 *
 */
BOOST_AUTO_TEST_CASE(CorrelationCoeffToySdp)
{
  // The semi-definite constraint looks like:
  //
  // [ 1  x_12  x_13  0  0  0  0 ]
  // [     1    x_23  0  0  0  0 ]
  // [            1   0  0  0  0 ]
  // [               s1  0  0  0 ]  >= 0
  // [                  s2  0  0 ]
  // [                     s3  0 ]
  // [                        s4 ]


  // x_11 == 0
  arma::sp_mat A0(7, 7); A0.zeros();
  A0(0, 0) = 1.;

  // x_22 == 0
  arma::sp_mat A1(7, 7); A1.zeros();
  A1(1, 1) = 1.;

  // x_33 == 0
  arma::sp_mat A2(7, 7); A2.zeros();
  A2(2, 2) = 1.;

  // x_12 <= -0.1  <==>  x_12 + s1 == -0.1, s1 >= 0
  arma::sp_mat A3(7, 7); A3.zeros();
  A3(1, 0) = A3(0, 1) = 1.; A3(3, 3) = 2.;

  // -0.2 <= x_12  <==>  x_12 - s2 == -0.2, s2 >= 0
  arma::sp_mat A4(7, 7); A4.zeros();
  A4(1, 0) = A4(0, 1) = 1.; A4(4, 4) = -2.;

  // x_23 <= 0.5  <==>  x_23 + s3 == 0.5, s3 >= 0
  arma::sp_mat A5(7, 7); A5.zeros();
  A5(2, 1) = A5(1, 2) = 1.; A5(5, 5) = 2.;

  // 0.4 <= x_23  <==>  x_23 - s4 == 0.4, s4 >= 0
  arma::sp_mat A6(7, 7); A6.zeros();
  A6(2, 1) = A6(1, 2) = 1.; A6(6, 6) = -2.;

  std::vector<arma::sp_mat> ais({A0, A1, A2, A3, A4, A5, A6});

  SDP<arma::sp_mat> sdp(7, 7 + 4 + 4 + 4 + 3 + 2 + 1, 0);

  for (size_t j = 0; j < 3; j++)
  {
    // x_j4 == x_j5 == x_j6 == x_j7 == 0
    for (size_t i = 0; i < 4; i++)
    {
      arma::sp_mat A(7, 7); A.zeros();
      A(i + 3, j) = A(j, i + 3) = 1;
      ais.emplace_back(A);
    }
  }

  // x_45 == x_46 == x_47 == 0
  for (size_t i = 0; i < 3; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 4, 3) = A(3, i + 4) = 1;
    ais.emplace_back(A);
  }

  // x_56 == x_57 == 0
  for (size_t i = 0; i < 2; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 5, 4) = A(4, i + 5) = 1;
    ais.emplace_back(A);
  }

  // x_67 == 0
  arma::sp_mat A(7, 7); A.zeros();
  A(6, 5) = A(5, 6) = 1;
  ais.emplace_back(A);

  std::swap(sdp.SparseA(), ais);

  sdp.SparseB().zeros();

  sdp.SparseB()[0] = sdp.SparseB()[1] = sdp.SparseB()[2] = 1.;

  sdp.SparseB()[3] = -0.2; sdp.SparseB()[4] = -0.4;

  sdp.SparseB()[5] = 1.; sdp.SparseB()[6] = 0.8;

  sdp.C().zeros();
  sdp.C()(0, 2) = sdp.C()(2, 0) = 1.;

  PrimalDualSolver<SDP<arma::sp_mat>> solver(sdp);
  arma::mat X, Z;
  arma::vec ysparse, ydense;
  const double obj = solver.Optimize(X, ysparse, ydense, Z);
  CheckKKT(sdp, X, ysparse, ydense, Z);
  BOOST_REQUIRE_CLOSE(obj, 2 * (-0.978), 1e-3);
}

///**
// * Maximum variance unfolding (MVU) SDP to learn the unrolled gram matrix. For
// * the SDP formulation, see:
// *
// *   Unsupervised learning of image manifolds by semidefinite programming.
// *   Kilian Weinberger and Lawrence Saul. CVPR 04.
// *   http://repository.upenn.edu/cgi/viewcontent.cgi?article=1000&context=cis_papers
// *
// * @param origData origDim x numPoints
// * @param numNeighbors
// */
//static inline SDP<arma::sp_mat> ConstructMvuSDP(const arma::mat& origData,
//                                                size_t numNeighbors)
//{
//  const size_t numPoints = origData.n_cols;
//
//  assert(numNeighbors <= numPoints);
//
//  arma::Mat<size_t> neighbors;
//  arma::mat distances;
//  KNN knn(origData);
//  knn.Search(numNeighbors, neighbors, distances);
//
//  SDP<arma::sp_mat> sdp(numPoints, numNeighbors * numPoints, 1);
//  sdp.C().eye(numPoints, numPoints);
//  sdp.C() *= -1;
//  sdp.DenseA()[0].ones(numPoints, numPoints);
//  sdp.DenseB()[0] = 0;
//
//  for (size_t i = 0; i < neighbors.n_cols; ++i)
//  {
//    for (size_t j = 0; j < numNeighbors; ++j)
//    {
//      // This is the index of the constraint.
//      const size_t index = (i * numNeighbors) + j;
//
//      arma::sp_mat& aRef = sdp.SparseA()[index];
//      aRef.zeros(numPoints, numPoints);
//
//      // A_ij(i, i) = 1.
//      aRef(i, i) = 1;
//
//      // A_ij(i, j) = -1.
//      aRef(i, neighbors(j, i)) = -1;
//
//      // A_ij(j, i) = -1.
//      aRef(neighbors(j, i), i) = -1;
//
//      // A_ij(j, j) = 1.
//      aRef(neighbors(j, i), neighbors(j, i)) = 1;
//
//      // The constraint b_ij is the distance between these two points.
//      sdp.SparseB()[index] = distances(j, i);
//    }
//  }
//
//  return sdp;
//}
//
///**
// * Maximum variance unfolding
// *
// * Test doesn't work, because the constraint matrices are not linearly
// * independent.
// */
//BOOST_AUTO_TEST_CASE(SmallMvuSdp)
//{
//  const size_t n = 20;
//
//  arma::mat origData(3, n);
//
//  // sample n random points on 3-dim unit sphere
//  GaussianDistribution gauss(3);
//  for (size_t i = 0; i < n; i++)
//  {
//    // how european of them
//    origData.col(i) = arma::normalise(gauss.Random());
//  }
//
//  auto sdp = ConstructMvuSDP(origData, 5);
//
//  PrimalDualSolver<SDP<arma::sp_mat>> solver(sdp);
//  arma::mat X, Z;
//  arma::vec ysparse, ydense;
//  const auto p = solver.Optimize(X, ysparse, ydense, Z);
//  BOOST_REQUIRE(p.first);
//}

BOOST_AUTO_TEST_SUITE_END();
