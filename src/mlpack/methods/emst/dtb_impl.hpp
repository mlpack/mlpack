/**
 * @file dtb_impl.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Implementation of DTB.
 */
#ifndef __MLPACK_METHODS_EMST_DTB_IMPL_HPP
#define __MLPACK_METHODS_EMST_DTB_IMPL_HPP

#include "dtb_rules.hpp"

namespace mlpack {
namespace emst {

//! Call the tree constructor that does mapping.
template<typename TreeType>
TreeType* BuildTree(
    typename TreeType::Mat& dataset,
    std::vector<size_t>& oldFromNew,
    typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == true, TreeType*
    >::type = 0)
{
  return new TreeType(dataset, oldFromNew);
}

//! Call the tree constructor that does not do mapping.
template<typename TreeType>
TreeType* BuildTree(
    const typename TreeType::Mat& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == false, TreeType*
    >::type = 0)
{
  return new TreeType(dataset);
}

/**
 * Takes in a reference to the data set.  Copies the data, builds the tree,
 * and initializes all of the member variables.
 */
template<typename MetricType, typename TreeType>
DualTreeBoruvka<MetricType, TreeType>::DualTreeBoruvka(
    const typename TreeType::Mat& dataset,
    const bool naive,
    const MetricType metric) :
    data((tree::TreeTraits<TreeType>::RearrangesDataset && !naive) ? dataCopy : dataset),
    ownTree(!naive),
    naive(naive),
    connections(dataset.n_cols),
    totalDist(0.0),
    metric(metric)
{
  Timer::Start("emst/tree_building");

  if (!naive)
  {
    // Copy the dataset, if it will be modified during tree construction.
    if (tree::TreeTraits<TreeType>::RearrangesDataset)
      dataCopy = dataset;

    tree = BuildTree<TreeType>(const_cast<typename TreeType::Mat&>(data),
        oldFromNew);
  }

  Timer::Stop("emst/tree_building");

  edges.reserve(data.n_cols - 1); // Set size.

  neighborsInComponent.set_size(data.n_cols);
  neighborsOutComponent.set_size(data.n_cols);
  neighborsDistances.set_size(data.n_cols);
  neighborsDistances.fill(DBL_MAX);
} // Constructor

template<typename MetricType, typename TreeType>
DualTreeBoruvka<MetricType, TreeType>::DualTreeBoruvka(
    TreeType* tree,
    const typename TreeType::Mat& dataset,
    const MetricType metric) :
    data(dataset),
    tree(tree),
    ownTree(false),
    naive(false),
    connections(data.n_cols),
    totalDist(0.0),
    metric(metric)
{
  edges.reserve(data.n_cols - 1); // Fill with EdgePairs.

  neighborsInComponent.set_size(data.n_cols);
  neighborsOutComponent.set_size(data.n_cols);
  neighborsDistances.set_size(data.n_cols);
  neighborsDistances.fill(DBL_MAX);
}

template<typename MetricType, typename TreeType>
DualTreeBoruvka<MetricType, TreeType>::~DualTreeBoruvka()
{
  if (ownTree)
    delete tree;
}

/**
 * Iteratively find the nearest neighbor of each component until the MST is
 * complete.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::ComputeMST(arma::mat& results)
{
  Timer::Start("emst/mst_computation");

  totalDist = 0; // Reset distance.

  typedef DTBRules<MetricType, TreeType> RuleType;
  RuleType rules(data, connections, neighborsDistances, neighborsInComponent,
                 neighborsOutComponent, metric);
  while (edges.size() < (data.n_cols - 1))
  {
    if (naive)
    {
      // Full O(N^2) traversal.
      for (size_t i = 0; i < data.n_cols; ++i)
        for (size_t j = 0; j < data.n_cols; ++j)
          rules.BaseCase(i, j);
    }
    else
    {
      typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);
      traverser.Traverse(*tree, *tree);
    }

    AddAllEdges();

    Cleanup();

    Log::Info << edges.size() << " edges found so far." << std::endl;
    if (!naive)
    {
      Log::Info << rules.BaseCases() << " cumulative base cases." << std::endl;
      Log::Info << rules.Scores() << " cumulative node combinations scored."
          << std::endl;
    }
  }

  Timer::Stop("emst/mst_computation");

  EmitResults(results);

  Log::Info << "Total spanning tree length: " << totalDist << std::endl;
}

/**
 * Adds a single edge to the edge list
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::AddEdge(const size_t e1,
                                        const size_t e2,
                                        const double distance)
{
  Log::Assert((distance >= 0.0),
      "DualTreeBoruvka::AddEdge(): distance cannot be negative.");

  if (e1 < e2)
    edges.push_back(EdgePair(e1, e2, distance));
  else
    edges.push_back(EdgePair(e2, e1, distance));
} // AddEdge

/**
 * Adds all the edges found in one iteration to the list of neighbors.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::AddAllEdges()
{
  for (size_t i = 0; i < data.n_cols; i++)
  {
    size_t component = connections.Find(i);
    size_t inEdge = neighborsInComponent[component];
    size_t outEdge = neighborsOutComponent[component];
    if (connections.Find(inEdge) != connections.Find(outEdge))
    {
      //totalDist = totalDist + dist;
      // changed to make this agree with the cover tree code
      totalDist += neighborsDistances[component];
      AddEdge(inEdge, outEdge, neighborsDistances[component]);
      connections.Union(inEdge, outEdge);
    }
  }
} // AddAllEdges

/**
 * Unpermute the edge list (if necessary) and output it to results.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::EmitResults(arma::mat& results)
{
  // Sort the edges.
  std::sort(edges.begin(), edges.end(), SortFun);

  Log::Assert(edges.size() == data.n_cols - 1);
  results.set_size(3, edges.size());

  // Need to unpermute the point labels.
  if (!naive && ownTree && tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    for (size_t i = 0; i < (data.n_cols - 1); i++)
    {
      // Make sure the edge list stores the smaller index first to
      // make checking correctness easier
      size_t ind1 = oldFromNew[edges[i].Lesser()];
      size_t ind2 = oldFromNew[edges[i].Greater()];

      if (ind1 < ind2)
      {
        edges[i].Lesser() = ind1;
        edges[i].Greater() = ind2;
      }
      else
      {
        edges[i].Lesser() = ind2;
        edges[i].Greater() = ind1;
      }

      results(0, i) = edges[i].Lesser();
      results(1, i) = edges[i].Greater();
      results(2, i) = edges[i].Distance();
    }
  }
  else
  {
    for (size_t i = 0; i < edges.size(); i++)
    {
      results(0, i) = edges[i].Lesser();
      results(1, i) = edges[i].Greater();
      results(2, i) = edges[i].Distance();
    }
  }
} // EmitResults

/**
 * This function resets the values in the nodes of the tree nearest neighbor
 * distance and checks for fully connected nodes.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::CleanupHelper(TreeType* tree)
{
  // Reset the statistic information.
  tree->Stat().MaxNeighborDistance() = DBL_MAX;
  tree->Stat().MinNeighborDistance() = DBL_MAX;
  tree->Stat().Bound() = DBL_MAX;

  // Recurse into all children.
  for (size_t i = 0; i < tree->NumChildren(); ++i)
    CleanupHelper(&tree->Child(i));

  // Get the component of the first child or point.  Then we will check to see
  // if all other components of children and points are the same.
  const int component = (tree->NumChildren() != 0) ?
      tree->Child(0).Stat().ComponentMembership() :
      connections.Find(tree->Point(0));

  // Check components of children.
  for (size_t i = 0; i < tree->NumChildren(); ++i)
    if (tree->Child(i).Stat().ComponentMembership() != component)
      return;

  // Check components of points.
  for (size_t i = 0; i < tree->NumPoints(); ++i)
    if (connections.Find(tree->Point(i)) != int(component))
      return;

  // If we made it this far, all components are the same.
  tree->Stat().ComponentMembership() = component;
}

/**
 * The values stored in the tree must be reset on each iteration.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::Cleanup()
{
  for (size_t i = 0; i < data.n_cols; i++)
    neighborsDistances[i] = DBL_MAX;

  if (!naive)
    CleanupHelper(tree);
}

// convert the object to a string
template<typename MetricType, typename TreeType>
std::string DualTreeBoruvka<MetricType, TreeType>::ToString() const
{
  std::ostringstream convert;
  convert << "DualTreeBoruvka [" << this << "]" << std::endl;
  convert << "  Data: " << data.n_rows << "x" << data.n_cols <<std::endl;
  convert << "  Total Distance: " << totalDist <<std::endl;
  convert << "  Naive: " << naive << std::endl;
  convert << "  Metric: " << std::endl;
  convert << util::Indent(metric.ToString(), 2);
  convert << std::endl;
  return convert.str();
}

}; // namespace emst
}; // namespace mlpack

#endif
