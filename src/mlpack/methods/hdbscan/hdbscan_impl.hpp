  /**
 * @file hdbscan_impl.hpp
 * @author Sudhanshu Ranjan
 *
 * Implementation of HDBSCAN.
 */
#ifndef __MLPACK_METHODS_HDBSCAN_HDBSCAN_IMPL_HPP
#define __MLPACK_METHODS_HDBSCAN_HDBSCAN_IMPL_HPP

#include "hdbscan.hpp"

namespace mlpack {
namespace hdbscan {

/**
 * Construct the HDBSCAN object with the given parameters.
 */
template<typename NeighborSearch,
         typename MetricType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType >
HDBSCAN<NeighborSearch,
        MetricType,
        TreeType>::
HDBSCAN(const size_t minPoints,
        bool allowSingleCluster) :
        minPoints(minPoints),
        allowSingleCluster(allowSingleCluster)
{
  // Nothing to do.
}

/**
 * Performs HDBSCAN clustering on the data,returning the list of cluster 
 * assignments. The assigned clusters have values from 0 onwards. The noise 
 * points are labelled as (total nuber of points + 1)
 *
 * Dcore -> distance of k-th nearest neighbor
 *          here k is minPoints
 * 1. Find the dcore of all the points.
 * 2. Then append dcore to end of the input data (dataWithDcore)
 * 3. Compute MST using the Dual Tree Borvuka Method.
 * 4. Sort the obtained MST on basis of length of edges
 *    in ascending order.
 * 5. Covert MST to single likage tree .
 *    Single linkage tree is a clustering method in which
 *    we go on clustering the points until a single cluster
 *    is formed.
 * 6. Condense the obtained single linkage tree
 *    to form condensed tree. Condensed tree contains only those
 *    clusters as parent which have more then minPoints number of
 *    points.
 * 7. From this condense tree obtain labels of all the points.
 */
template<typename NeighborSearch,
         typename MetricType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType >
template<typename MatType>
void HDBSCAN<NeighborSearch,
    MetricType,
    TreeType>::
Cluster(const MatType& data,
        arma::Row<size_t>& assignments)
{
  // genearting a matrix with the values of dcore
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighborSearch.Train(data);
  neighborSearch.Search(minPoints, neighbors, distances);
  arma::mat dataWithDcore = arma::conv_to<arma::mat>::from(data);
  dataWithDcore.resize(dataWithDcore.n_rows+1, dataWithDcore.n_cols);
  dataWithDcore.row(dataWithDcore.n_rows-1) = distances.row(distances.n_rows-1);

  // computing the mst of the given data and sort the obtained result
  emst::DualTreeBoruvka<MetricType, arma::mat, TreeType>
                       dtb(dataWithDcore, false, hMetric);
  arma::mat resultsMST;
  dtb.ComputeMST(resultsMST);

  // create single linkage tree from the MST obtained.
  arma::mat singleLinkageTree;
  SingleLinkageTreeClustering(resultsMST, singleLinkageTree);

  // Condense the obtained tree
  arma::mat condensedTree;
  CondenseTree(singleLinkageTree, condensedTree, minPoints);

  // get clusters
  arma::Mat<size_t> result;
  GetClusters(condensedTree, result);

  // save results obtained
  assignments = result;
}

//  A class to help merge two clusters to form a
// single linkage tree.
class UnionForSingleLinkageClustering
{
  arma::Col<size_t> parentVector;
  arma::Col<size_t> sizeVector;
  size_t nextLabel;

 public:
  UnionForSingleLinkageClustering(size_t n)
  {
    // Initialize the elments
    // There would be a max of 2*n-1 clusters
    // formed after completion
    // Initally all n points form a cluster
    // After first merging there are
    // (n-2) old clusters and 1 new cluster
    // This goes on until at last there is one cluster
    // (n-1) + n = (2n-1)
    parentVector.set_size(2*n-1);
    parentVector.fill(SIZE_MAX);
    sizeVector.set_size(2*n-1);
    for (size_t i = 0; i < n; i++)
      sizeVector(i) = 1;
    for (size_t i = n; i < 2*n-1; i++)
      sizeVector(i) = 0;
    nextLabel = n;
  }
  void UnionOfTwoClusters(size_t m, size_t n)
  {
    // Each union will form a new cluster
    // with size as sum of its children
    sizeVector(nextLabel) = sizeVector(m) + sizeVector(n);
    parentVector(m) = nextLabel;
    parentVector(n) = nextLabel;
    nextLabel++;
  }
  size_t Find(size_t n)
  {
    // finds the parent
    size_t root = n;
    while (parentVector(root) != SIZE_MAX)
      root = parentVector(root);
    while (parentVector(n) != root)
    {
      size_t temp = parentVector(n);
      parentVector(n) = root;
      n = temp;
      if (n == SIZE_MAX)
        break;
    }
      return root;
  }
  size_t GetSize(size_t i)
  {
    return sizeVector(i);
  }
};

/**
 * Converts the input (sorted MST) to a single linkage tree.
 * Each row of the output contaisn 3 values
 * (cluster1 cluster2 d size)
 *  d = distance between cluster1 and cluster2
 *  size = size of cluster formed by merging cluster1 and cluster2
 *  cluster1 and cluster2 may contain more than one point
 *  Initially points are labelled as 0,1,2,...,(n-1)
 *  and are single cluster(contain only one point)
 *  Then we go on merging 2 clusters at a time until we
 *  reach a point when all points are in one cluster
 * 
 *  This is done with help of unionForSingleLinkageTree
 *  For every edge in MST, a merge operation is performed
 *  which merges two clusters.
 */

template<typename NeighborSearch,
         typename MetricType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType >
template<typename MatType>
void HDBSCAN<NeighborSearch,
             MetricType,
             TreeType>::
SingleLinkageTreeClustering(const MatType& inputMST,
                          MatType& singleLinkageTree)
{
  singleLinkageTree.set_size(4, inputMST.n_cols);
  size_t no = inputMST.n_cols + 1;
  UnionForSingleLinkageClustering u(no);
  size_t parentOfv1, parentOfv2;
  double delta;
  for (size_t i = 0; i < inputMST.n_cols; i++)
  {
    delta = inputMST(2, i);
    parentOfv1 = u.Find(inputMST(0, i));
    parentOfv2 = u.Find(inputMST(1, i));
    singleLinkageTree.col(i) = arma::vec({(double)parentOfv1,
      (double)parentOfv2, delta,
      (double)(u.GetSize(parentOfv1) + u.GetSize(parentOfv2))});
    u.UnionOfTwoClusters(parentOfv1, parentOfv2);
  }
}


/**
 * Performs a modified bfs on single linkage tree,
 * helper function to condense tree
 * Only those clusters of single linkage tree
 * are present in BFS which have at least 2 points. 
 *
 */
template<typename NeighborSearch,
         typename MetricType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType >
template<typename MatType>
void HDBSCAN<NeighborSearch,
             MetricType,
             TreeType>::
SingleLinkageTreeToModifiedBFS(const MatType& singleLinkageTree,
                 std::vector<size_t>& bfs,
                 size_t rootOfBFS)
{
  // size_t numPoints = singleLinkageTree.n_rows + 1;
  size_t numPoints = singleLinkageTree.n_cols + 1;
  std::queue<size_t> q;
  q.push(rootOfBFS);

  while (!q.empty())
  {
    size_t index, topElement = q.front();
    q.pop();
    bfs.push_back(topElement);
    if (topElement >= numPoints)
    {
      index = topElement -numPoints;
      q.push(singleLinkageTree(0, index));
      q.push(singleLinkageTree(1, index));
    }
  }
}

/**
 * This function condenses the single linkage tree .
 * Each row of output of condensed tree contains 4 values.
 * (parent child lambda size)
 * parent -> this cluster is above child in the condensed tree,
 *           does not need to be immediate parent
 * child ->  this cluster is child,
 *           the parent belongs to a bigger cluster of which the
 *           child cluster is a part
 * lambda -> lambda = (1/d)
 *           where d is distance between cluster 
 *           containing parent and cluster containing child
 * size -> size of cluster to which child belongs
 *
 * 1. Consider that cluster of single linkage tree.
 *    which has all the points in it. Perform BFS on
 *    single linkage tree with this cluster as root.
 * 2. Now consider all the points inside the obtained BFS
 *    which have at least two child clusters.
 *    2.1 Find the left and right child cluster.
 *    2.2 Find the number of point in both of them.
 *    2.3 For each child cluster,
 *        if the size of child cluster  > minPoints
 *            assign new label to child cluster
 *            add it as the edge of condensed tree
 *        else
 *            assign it and all its children
 *            in bfs of single linkage tree
 *            the label of parent cluster,
 *            set ignore as true for this
 *            child cluster and all its children in
 *            the single linkage tree                              
 *       
 */
template<typename NeighborSearch,
         typename MetricType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType >
template<typename MatType>
void HDBSCAN<NeighborSearch,
             MetricType,
             TreeType>::
CondenseTree(const MatType& singleLinkageTree,
           MatType& result,
           size_t minClusterSize )
{
  std::vector<size_t> bfs;
  // Keeps track of number of elments in result matrix
  size_t resultCounter = 0;
  size_t rootOfBFS = 2 * singleLinkageTree.n_cols;
  SingleLinkageTreeToModifiedBFS(singleLinkageTree, bfs, rootOfBFS);
  result.set_size(4, bfs.size() * 2);
  double lambda;
  size_t noOfPoints = singleLinkageTree.n_cols + 1;
  size_t nextLabel = noOfPoints + 1;

  // stores label of all the clusters
  std::vector<size_t> relabel(2 * singleLinkageTree.n_cols + 1);
  relabel[rootOfBFS] = noOfPoints;
  // if cluster i can be a parent or not
  std::vector<bool> ignore(bfs.size(), false);
  double delta;
  size_t leftCount, rightCount, leftChild, rightChild, currNode;
  std::vector<size_t> bfsOfChild;

  for (size_t i = 0; i < bfs.size(); i++)
  {
    currNode = bfs[i];
    // currNode cant be a parent
    if ( ignore[currNode] || currNode < noOfPoints) continue;
    // find left right child clusters, their size and lambda
    leftChild = singleLinkageTree(0, (currNode - noOfPoints));
    rightChild = singleLinkageTree(1, (currNode - noOfPoints));
    leftCount = ((leftChild >= noOfPoints) ?
                  singleLinkageTree(3, (leftChild - noOfPoints)) :
                  1);
    rightCount = ((rightChild >= noOfPoints) ?
                   singleLinkageTree(3, (rightChild - noOfPoints)) :
                   1);
    delta = singleLinkageTree(2, (currNode - noOfPoints));
    if (delta > 0.0)
      lambda = 1.0 / delta;
    else
      lambda = std::numeric_limits<double>::infinity();

    if (leftCount >= minClusterSize && rightCount >= minClusterSize)
    {
      relabel[leftChild] = nextLabel;
      nextLabel++;
      result.col(resultCounter) = arma::vec({(double)relabel[currNode],
        (double)relabel[leftChild], (double)lambda, (double)leftCount });
      resultCounter++;

      relabel[rightChild] = nextLabel;
      nextLabel++;
      result.col(resultCounter) = arma::vec({(double)relabel[currNode],
        (double)relabel[rightChild], (double)lambda, (double)rightCount });
      resultCounter++;
    }
    else if (leftCount < minClusterSize && rightCount < minClusterSize)
    {
      bfsOfChild.resize(0);
      SingleLinkageTreeToModifiedBFS(singleLinkageTree,
                                   bfsOfChild,
                                   leftChild);
      for (size_t j = 0; j < bfsOfChild.size(); j++)
      {
        size_t subNode = bfsOfChild[j];
        if (subNode < noOfPoints)
        {
          result.col(resultCounter) = arma::vec({(double)relabel[currNode],
            (double)subNode, (double)lambda, 1.0});
          resultCounter++;
        }
        ignore[subNode] = true;
      }
      bfsOfChild.resize(0);
      SingleLinkageTreeToModifiedBFS(singleLinkageTree,
                                   bfsOfChild,
                                   rightChild);
      for (size_t j = 0; j < bfsOfChild.size(); j++)
      {
        size_t subNode = bfsOfChild[j];
        if (subNode < noOfPoints)
        {
          result.col(resultCounter) = arma::vec({(double)relabel[currNode],
            (double)subNode, (double)lambda, 1.0});
          resultCounter++;
        }
        ignore[subNode] = true;
      }
    }
    else if (leftCount < minClusterSize)
    {
      relabel[rightChild] = relabel[currNode];
      bfsOfChild.resize(0);
      SingleLinkageTreeToModifiedBFS(singleLinkageTree,
                                   bfsOfChild,
                                   leftChild);
      for (size_t j = 0; j < bfsOfChild.size(); j++)
      {
        size_t subNode = bfsOfChild[j];
        if (subNode < noOfPoints)
        {
          result.col(resultCounter) = arma::vec({(double)relabel[currNode],
            (double)subNode, (double)lambda, 1.0});
          resultCounter++;
        }
        ignore[subNode] = true;
      }
    }
    else
    {
      relabel[leftChild] = relabel[currNode];
      bfsOfChild.resize(0);
      SingleLinkageTreeToModifiedBFS(singleLinkageTree,
                                   bfsOfChild,
                                   rightChild);
      for (size_t j = 0; j < bfsOfChild.size(); j++)
      {
        size_t subNode = bfsOfChild[j];
        if (subNode < noOfPoints)
        {
          result.col(resultCounter) = arma::vec({(double)relabel[currNode],
            (double)subNode, (double)lambda, 1.0});
          resultCounter++;
        }
        ignore[subNode] = true;
      }
    }
  }
  result.resize(4, resultCounter);
}

/**
 * This function helps in computing stabilities
 * helper function of getClusters
 *
 * For a cluster lambda_birth = lambda when cluster split off from larger
 *                              cluster and became a new cluster
 * For a smaller cluster p inside a bigger cluster C,
 * define lambda_p = lambda when cluster p falls off from
 *                              the parent cluster C
 * Define stability of cluster C as :
 * stability = sum( (lambda_p - lamba_birth) * (size of child cluster) )
 *                              for all smaller clusters p in
 *                              bigger cluster C
 *
 *
 * 1. Find lambda at time of birth of each cluster.
 * 2. Find stability of each cluster using above mentioned formula.
 */
template<typename NeighborSearch,
         typename MetricType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType >
template<typename MatType>
void HDBSCAN<NeighborSearch,
             MetricType,
             TreeType>::
GetStabilities(const MatType& condensedTree,
             std::map<size_t, double>& result)
{
  size_t largestChild = arma::max(condensedTree.row(1));
  size_t smallestCluster = arma::min(condensedTree.row(0));
  size_t noOfClusters = arma::max(condensedTree.row(0)) -
              arma::min(condensedTree.row(0)) + 1;
  for (size_t i = 0; i < noOfClusters; i++)
    result[smallestCluster + i] = 0;
  if (largestChild < smallestCluster)
    largestChild = smallestCluster;

  MatType sortedChildData(2, condensedTree.n_cols);
  sortedChildData.row(0) = condensedTree.row(1);
  sortedChildData.row(1) = condensedTree.row(2);

  // SortMatrix(sortedChildData, 0);
  arma::uvec indices = arma::sort_index(sortedChildData.row(0));
  size_t i;

  std::vector<double> births(largestChild+1);
  size_t currentChild = SIZE_MAX;
  size_t child;
  double lambda;
  double minLambda = 0;

  // Finding lambda at birth of each cluster
  // This for loop helps when there are multiple instances
  // of same cluster as child.
  // Then lambda will be minimum of all those.
  for (size_t j = 0; j < sortedChildData.n_cols; j++)
  {
    i = indices[j];
    child = sortedChildData(0, i);
    lambda = sortedChildData(1, i);

    if (child == currentChild)
    {
      minLambda = std::min(lambda, minLambda);
    }
    else if (currentChild != SIZE_MAX)
    {
      births[currentChild] = minLambda;
      currentChild = child;
      minLambda = lambda;
    }
    else
    {
      currentChild = child;
      minLambda = lambda;
    }
  }
  if (currentChild != SIZE_MAX)
    births[currentChild] = minLambda;
  births[smallestCluster] = 0.0;

  // calculate stability for each cluster now
  size_t parent, childSize;
  for (size_t i = 0; i < condensedTree.n_cols; i++)
  {
    parent = condensedTree(0, i);
    lambda = condensedTree(2, i);
    childSize = condensedTree(3, i);

    result[parent] += (lambda - births[parent]) * childSize;
  }
} // function getStabilities ends

/**
 * This function coverts a clustered tree to BFS and chooses root 
 * as the specified node.
 * Helper function of getClusters
 */
template<typename NeighborSearch,
         typename MetricType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType >
template<typename MatType>
void HDBSCAN<NeighborSearch,
             MetricType,
             TreeType>::
GetBfsFromClusteredTree(MatType& clusteredTree,
                      size_t rootNode,
                      std::vector<size_t>& resultBFS)
{
  std::queue<size_t> q;

  // Clustered tree is empty
  if (clusteredTree.n_cols == 0 || clusteredTree.n_rows == 0)
    return;
  std::vector<bool> visited(std::max(arma::max(clusteredTree.row(0)),
                                     arma::max(clusteredTree.row(1))),
                                     false);
  visited[rootNode] = true;
  q.push(rootNode);
  while (!q.empty())
  {
    size_t currentNode = q.front();
    resultBFS.push_back(currentNode);
    visited[currentNode] = true;
    q.pop();
    for (size_t i = 0; i < clusteredTree.n_cols; i++)
      if (clusteredTree(0, i) == currentNode && !visited[clusteredTree(1, i)])
        q.push(clusteredTree(1, i));
  }
} // function getBfsFromClusteredTree ends

/**
 * This function helps in assigning labels to all the points
 * provided a condensedTree and clusters which can be root of other clusters.
 * Root of cluster -> Parent of a cluster in condensed tree.
 * Helper function of getClusters
 * Roots of cluster are labeled and all the children of a root are
 * labeled same as root. 
 *
 * 1. Labels the clusters which can be root.
 * 2. Iterate through the condensed tree,
 *    check if child cluster can be root,
 *    no -> merge them 
 * 3. Iterate through all the points in the
 *    dataset, if the parent of the point
 *    can be root of any cluster,
 *    yes: assign parent's label to the point,
 *    else assign noOfPoints+1 
 *   
 */
template<typename NeighborSearch,
         typename MetricType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType >
template<typename MatType>
void HDBSCAN<NeighborSearch,
             MetricType,
             TreeType>::
GetLabels(const MatType& condensedTree,
        std::vector<size_t> clusters,
        arma::Mat<size_t>& result)
{
  // Label all the clusters which can be root
  sort(clusters.begin(), clusters.end());
  std::map<size_t, size_t> labelOfClusters;
  for (size_t i = 0; i < clusters.size(); i++)
    labelOfClusters[clusters[i]] = i;

  // Find the smallest cluster which can be the root
  // Condensed tree contan all those clusters
  // whose child cluster has size >= 1
  // Minimum will be that cluster which has one point
  // And will be equal to size of input original data set
  size_t rootCluster = arma::min(condensedTree.row(0));
  result.set_size(1, rootCluster);
  emst::UnionFind unionTree(arma::max(condensedTree.row(0))+1);

  // Go through all edges in the condensed tree
  // If the child cluster of edge cannot be root of any cluster
  // Merge those clusters
  for (size_t i = 0; i < condensedTree.n_cols; i++)
    if (!std::binary_search(clusters.begin(),
                           clusters.end(),
                           condensedTree(1, i)))
      unionTree.Union(condensedTree(0, i), condensedTree(1, i));

  // Now assign labels to all the points of the dataset
  size_t currentPoint, parentOfCurrentPoint;

  // If only one cluster is possible
  // Check lambda value of all the children
  // If lambda_p >= lambda_rootCluster
  // p is included in cluster
  // Otherwise it is a noise
  if (clusters.size() == 1 && allowSingleCluster)
  {
    // find lambda of root cluster
    double lambdaRootCluster = 0;
    size_t temp = 0;
    for (size_t i = 0; i < condensedTree.n_cols; i++)
    {
      if (condensedTree(0, i) == rootCluster)
        temp++, lambdaRootCluster = std::max(lambdaRootCluster, condensedTree(2, i));
    }

    // Mark all the points as noises
    for (size_t i = 0; i < rootCluster; i++)
      result[i] = SIZE_MAX;

    double eps = std::numeric_limits<double>::epsilon();
    double eps_error = pow(condensedTree.n_cols, 2) * eps;
    // consider all edges in condensed tree
    // whose child is a single point (not cluster)
    // check if their lambda has value
    // less then that of the root cluster
    for (size_t i = 0; i < condensedTree.n_cols; i++)
    {      currentPoint = condensedTree(1, i);
      if (currentPoint >= rootCluster)  continue;
      parentOfCurrentPoint = unionTree.Find(currentPoint);
      if (parentOfCurrentPoint == rootCluster &&
        condensedTree(2, i) + eps_error >= lambdaRootCluster)
        result[currentPoint] = 0;
    }

    return;
  }

  for (currentPoint = 0; currentPoint < rootCluster; currentPoint++)
  {
    parentOfCurrentPoint = unionTree.Find(currentPoint);
    if (parentOfCurrentPoint <= rootCluster)
      // label these points as noise
      result[currentPoint] = SIZE_MAX;
    else
      // assign the correct cluster
      result[currentPoint] = labelOfClusters[parentOfCurrentPoint];
  }
} // function getLabels ends

/**
 * This functin provides label to each and every point
 * Noise points are labelled as ( total number of points + 1)
 *
 * Find all the clusters which can be root and then assign labels
 * to the points in the input dataset.
 */
template<typename NeighborSearch,
         typename MetricType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType >
template<typename MatType>
void HDBSCAN<NeighborSearch,
             MetricType,
             TreeType>::
GetClusters(MatType& condensedTree,
          arma::Mat<size_t>& result)
{
  std::map<size_t, double> stabilities;
  GetStabilities(condensedTree, stabilities);
  // nodes contain all clusters in condensedTree which can be parent
  std::vector<size_t> nodes;
  // std::map<size_t, double>::iterator it;
  for (auto it = stabilities.begin(); it != stabilities.end(); ++it)
    nodes.push_back(it->first);
  sort(nodes.begin(), nodes.end(), std::greater<size_t>());

  // Stops from making a single cluster for entire dataset
  if (!allowSingleCluster)
  {
    if (nodes.size() > 0)
      nodes.pop_back();
  }

  // Clustered tree contains all those edges of condensed tree
  // whose child cluster has size more than 1
  MatType clusteredTree(condensedTree.n_rows, condensedTree.n_cols);
  size_t clusterTreeIndex = 0;

  for (size_t i = 0; i < condensedTree.n_cols; i++)
    if (condensedTree(3, i)  > 1)
      {
        clusteredTree.col(clusterTreeIndex) = condensedTree.col(i);
        clusterTreeIndex++;
      }
  clusteredTree.resize(condensedTree.n_rows, clusterTreeIndex);

  // Keeps track, if a point is in cluster or not
  std::vector<bool> isCluster(std::max(arma::max(condensedTree.row(0)),
                              arma::max(condensedTree.row(1))),
                              false);

  for (size_t i = 0; i < nodes.size(); i++)
    isCluster[nodes[i]] = true;

  size_t currentNode;
  double subtreeStability = 0;
  // Finds all stable clusters
  for (size_t i = 0; i < nodes.size(); i++)
  {
    currentNode = nodes[i];
    subtreeStability = 0;

    for (size_t j = 0; j < clusteredTree.n_cols; j++)
        if (clusteredTree(0, j) == currentNode)
        {
          subtreeStability += stabilities[(size_t)clusteredTree(1, j)];
        }

    if (subtreeStability > stabilities[currentNode])
    {
      isCluster[currentNode] = false;
      stabilities[currentNode] = subtreeStability;
    }
    else
    {
      std::vector<size_t> bfsFromClusteredTree;
      GetBfsFromClusteredTree(clusteredTree, currentNode, bfsFromClusteredTree);
      for (size_t j = 0; j < bfsFromClusteredTree.size(); j++)
      {
        if (bfsFromClusteredTree[j] != currentNode)
          isCluster[bfsFromClusteredTree[j]] = false;
      }
    }
  }

  // Send all cluster which can be root to
  // the function getLabels
  std::vector<size_t> clusters;
  for (size_t i = 0; i < nodes.size(); i++)
    if (isCluster[nodes[i]])
      clusters.push_back(nodes[i]);

  GetLabels(condensedTree, clusters, result);
} // function getCluster ends

} // namespace hdbscan
} // namespace mlpack

#endif

