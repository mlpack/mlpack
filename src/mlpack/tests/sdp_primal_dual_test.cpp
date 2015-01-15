/**
 * @file sdp_primal_dual_test.cpp
 * @author Stephen Tu
 *
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sdp/sdp.hpp>
#include <mlpack/core/optimizers/sdp/primal_dual.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::optimization;

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
                            const std::string& edgesFilename)
  {
    data::Load(edgesFilename, g.edges, true);
    if (g.edges.n_rows != 2)
      Log::Fatal << "Invalid datafile" << std::endl;
    g.weights.ones(g.edges.n_cols);
    g.ComputeVertices();
  }

  static void LoadFromEdgesAndWeights(UndirectedGraph& g,
                                      const std::string& edgesFilename,
                                      const std::string& weightsFilename)
  {
    data::Load(edgesFilename, g.edges, true);
    if (g.edges.n_rows != 2)
      Log::Fatal << "Invalid datafile" << std::endl;
    data::Load(weightsFilename, g.weights);
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

static inline SDP
ConstructMaxCutSDP(const UndirectedGraph& g)
{
  SDP sdp(g.NumVertices(), g.NumVertices(), 0);
  g.Laplacian(sdp.SparseC());
  sdp.SparseC() *= -1;
  for (size_t i = 0; i < g.NumVertices(); i++)
  {
    sdp.SparseA()[i].zeros(g.NumVertices(), g.NumVertices());
    sdp.SparseA()[i](i, i) = 1.;
  }
  sdp.SparseB().ones();
  return sdp;
}

// TODO: does arma have a builtin way to do this?
static inline arma::mat
Diag(const arma::vec& diag)
{
  arma::mat ret;
  ret.zeros(diag.n_elem, diag.n_elem);
  for (size_t i = 0; i < diag.n_elem; i++)
  {
    ret(i, i) = diag(i);
  }
  return ret;
}

BOOST_AUTO_TEST_SUITE(SdpPrimalDualTest);

/**
 * Start from a strictly feasible point
 */
BOOST_AUTO_TEST_CASE(SmallMaxCutFeasibleSdp)
{
  UndirectedGraph g;
  UndirectedGraph::ErdosRenyiRandomGraph(g, 10, 0.3, true);
  SDP sdp = ConstructMaxCutSDP(g);

  arma::mat X0, Z0;
  arma::vec ysparse0, ydense0;
  ydense0.set_size(0);

  X0.eye(g.NumVertices(), g.NumVertices());
  ysparse0 = -1.1 * arma::vec(arma::sum(arma::abs(sdp.SparseC()), 0));
  Z0 = -Diag(ysparse0) + sdp.SparseC();

  PrimalDualSolver solver(sdp, X0, ysparse0, ydense0, Z0);

  arma::mat X, Z;
  arma::vec ysparse, ydense;
  solver.Optimize(X, ysparse, ydense, Z);

}

BOOST_AUTO_TEST_SUITE_END();
