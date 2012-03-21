/**
 * @file kmeans_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * Implementation for the K-means method for getting an initial point.
 */
#include "kmeans.hpp"

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/hrectbound.hpp>
#include <mlpack/core/tree/mrkd_statistic.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include <stack>

namespace mlpack {
namespace kmeans {

/**
 * Construct the K-Means object.
 */
template<typename DistanceMetric,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy>
KMeans<
    DistanceMetric,
    InitialPartitionPolicy,
    EmptyClusterPolicy>::
KMeans(const size_t maxIterations,
       const double overclusteringFactor,
       const DistanceMetric metric,
       const InitialPartitionPolicy partitioner,
       const EmptyClusterPolicy emptyClusterAction) :
    maxIterations(maxIterations),
    metric(metric),
    partitioner(partitioner),
    emptyClusterAction(emptyClusterAction)
{
  // Validate overclustering factor.
  if (overclusteringFactor < 1.0)
  {
    Log::Warn << "KMeans::KMeans(): overclustering factor must be >= 1.0 ("
        << overclusteringFactor << " given). Setting factor to 1.0.\n";
    this->overclusteringFactor = 1.0;
  }
  else
  {
    this->overclusteringFactor = overclusteringFactor;
  }
}

template<typename DistanceMetric,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy>
template<typename MatType>
void KMeans<
    DistanceMetric,
    InitialPartitionPolicy,
    EmptyClusterPolicy>::
FastCluster(MatType& data,
            const size_t clusters,
            arma::Col<size_t>& assignments) const
{
  size_t actualClusters = size_t(overclusteringFactor * clusters);
  if (actualClusters > data.n_cols)
  {
    Log::Warn << "KMeans::Cluster(): overclustering factor is too large.  No "
        << "overclustering will be done." << std::endl;
    actualClusters = clusters;
  }

  // TODO: remove
  // Scale the data to [0,1]
  if(0){
    arma::rowvec min = arma::min(data, 0);
    data = (data - arma::ones<arma::colvec>(data.n_rows) * min) / (arma::ones<arma::colvec>(data.n_rows) * (arma::max(data,0) - min));
    for(size_t i = 0; i < data.n_cols; ++i)
      for(size_t j = 0; j < data.n_rows; ++j)
        assert(data(j,i) >= 0 && data(j,i) <= 1);
  }

  if (assignments.n_rows != data.n_cols)
    assignments.resize(data.n_cols);

  // Centroids of each cluster.  Each column corresponds to a centroid.
  MatType centroids(data.n_rows, actualClusters);

  // Counts of points in each cluster.
  arma::Col<size_t> counts(actualClusters);
  counts.zeros();

  // Build the mrkd-tree on this dataset
  tree::BinarySpaceTree<typename bound::HRectBound<2>, tree::MRKDStatistic> tree(data, 1);
  // A pointer for traversing the mrkd-tree
  tree::BinarySpaceTree<typename bound::HRectBound<2>, tree::MRKDStatistic>* node;

  // We use this to store the furtherst point in a hyperrectangle from a given
  // vector.
  arma::colvec p(data.n_rows);

  // Make random centroids and fit them into the root hyperrectangle.
  {
    centroids.randu();
    bound::HRectBound<2>& bound = tree.Bound();
    size_t dim = bound.Dim();
    for (size_t i = 0; i < dim; ++i) {
      double min = bound[i].Lo();
      double max = bound[i].Hi();
      for (size_t j = 0; j < centroids.n_cols; ++j)
      {
        if (centroids(i,j) < min)
          centroids(i,j) = min;
        else if (centroids(i,j) > max)
          centroids(i,j) = max;
      }
    }
  }

  // Instead of retraversing the tree after an iteration, we will update centroid
  // positions in this matrix, which also prevents clobbering our centroids from
  // the previous iteration.
  MatType newCentroids(centroids.n_rows, centroids.n_cols);

  std::cout << data.n_cols << std::endl;
  size_t iteration = 0;
  size_t changedAssignments = 0;
  do 
  {
    // Keep track of what iteration we are on.
    ++iteration;
    changedAssignments = 0;
    newCentroids.zeros();
    counts.zeros();

    // Create a stack for traversing the mrkd-tree
    std::stack<typename tree::BinarySpaceTree<typename bound::HRectBound<2>, 
                                              tree::MRKDStatistic>* > stack;
    // Add the root node of the tree to the stack
    stack.push(&tree);

    while (!stack.empty())
    {
      node = stack.top();
      stack.pop();

      tree::MRKDStatistic& mrkd = node->Stat();

      size_t minIndex = 0;

      // If this node is a leaf, then we calculate the distance from
      // the centroids to every point the node contains.
      if (node->IsLeaf())
      {
        //std::cout << "Leaf\t";
        for (size_t i = mrkd.begin; i < mrkd.count + mrkd.begin; ++i)
        {
          // Initialize minDistance to be nonzero.
          double minDistance = metric::SquaredEuclideanDistance::Evaluate(
              data.col(i), centroids.col(0));
          // Find the minimal distance centroid for this point.
          for (size_t j = 1; j < centroids.n_cols; ++j)
          {
            double distance = metric::SquaredEuclideanDistance::Evaluate(
                data.col(i), centroids.col(j));
            if ( minDistance > distance )
            {
              minIndex = j;
              minDistance = distance;
            }
          }

          newCentroids.col(minIndex) += data.col(i);
          ++counts(minIndex);
          //std::cout << counts(minIndex) << "\t";
          if (assignments(i) != minIndex)
          {
            ++changedAssignments;
            // TODO: this if should be removed
            //if(counts(assignments(i)))
              //--counts(assignments(i));
            assignments(i) = minIndex;
          }
        }
        //std::cout << std::endl;
      }
      // If this node is not a leaf, then we continue trying to find dominant
      // centroids
      else
      {
        //std::cout << "Parent\t";
        bound::HRectBound<2>& bound = node->Bound();

        bool noDomination = false;

        // There was no centroid inside this hyperrectangle.
        // We must determine if an external centroid dominates it.
        for (size_t i = 0; i < centroids.n_cols; ++i) 
        {
          noDomination = false;
          for (size_t j = 0; j < centroids.n_cols; ++j)
          {
            if (j == i)
              continue;

            for (size_t k = 0; k < p.n_rows; ++k)
            {
              p(k) = (centroids(k,j) > centroids(k,i)) ?
                bound[k].Hi() : bound[k].Lo();
            }

            double distancei = metric::SquaredEuclideanDistance::Evaluate(
                p.col(0), centroids.col(i));
            double distancej = metric::SquaredEuclideanDistance::Evaluate(
                p.col(0), centroids.col(j));

            if (distancei >= distancej)
            {
              noDomination = true;
              break;
            }

          }

          // We identified a centroid that dominates this hyperrectangle.
          if (!noDomination)
          {
            //std::cout << "Domination\t";
            newCentroids.col(minIndex) += mrkd.centerOfMass;
            counts(i) += mrkd.count;
            //std::cout << counts(i) << std::endl;
            // Update all assignments for this node
            const size_t begin = node->Begin();
            const size_t end = node->End();
            for (size_t j = begin; j < end; ++j)
            {
              if (assignments(j) != i)
              {
                ++changedAssignments;
                //if(counts(assignments(j)))
                  //--counts(assignments(j));
                //++counts(i);
                assignments(j) = i;
              }
            }
            mrkd.dominatingCentroid = i;
            break;
          }
        }

        // If we did not find a dominating centroid then we fall through to the
        // default case, where we add the children of this node to the stack.
        if (noDomination)
        {
          //std::cout << "No Domination" << std::endl;
          stack.push(node->Left());
          stack.push(node->Right());
        }
      }

    }

    for (size_t i = 0; i < centroids.n_cols; ++i)
    {
      if (counts(i)) {
        // Divide by the number of points assigned to this centroid so that we
        // have the actual center of mass and update centroids' positions.
        centroids.col(i) = newCentroids.col(i) / counts(i);
      }
    }
    size_t count = 0;
    for(size_t k = 0; k < counts.n_rows; ++k)
    {
      std::cout << counts(k) << '\t';
      count += counts(k);
    }
    std::cout << '\n' << count <<'\t'<< data.n_cols<< std::endl;
    assert(count <= data.n_cols);
  } while(0);
  //} while (changedAssignments > 0 && iteration != maxIterations);

}

/**
 * Perform K-Means clustering on the data, returning a list of cluster
 * assignments.
 */
template<typename DistanceMetric,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy>
template<typename MatType>
void KMeans<
    DistanceMetric,
    InitialPartitionPolicy,
    EmptyClusterPolicy>::
Cluster(const MatType& data,
        const size_t clusters,
        arma::Col<size_t>& assignments) const
{
  // Make sure we have more points than clusters.
  if (clusters > data.n_cols)
    Log::Warn << "KMeans::Cluster(): more clusters requested than points given."
        << std::endl;

  // Make sure our overclustering factor is valid.
  size_t actualClusters = size_t(overclusteringFactor * clusters);
  if (actualClusters > data.n_cols)
  {
    Log::Warn << "KMeans::Cluster(): overclustering factor is too large.  No "
        << "overclustering will be done." << std::endl;
    actualClusters = clusters;
  }

  // Now, the initial assignments.  First determine if they are necessary.
  if (assignments.n_elem != data.n_cols)
  {
    // Use the partitioner to come up with the partition assignments.
    partitioner.Cluster(data, actualClusters, assignments);
  }

  // Centroids of each cluster.  Each column corresponds to a centroid.
  MatType centroids(data.n_rows, actualClusters);
  // Counts of points in each cluster.
  arma::Col<size_t> counts(actualClusters);
  counts.zeros();

  // Set counts correctly.
  for (size_t i = 0; i < assignments.n_elem; i++)
    counts[assignments[i]]++;

  size_t changedAssignments = 0;
  size_t iteration = 0;
  do
  {
    // Update step.
    // Calculate centroids based on given assignments.
    centroids.zeros();

    for (size_t i = 0; i < data.n_cols; i++)
      centroids.col(assignments[i]) += data.col(i);

    for (size_t i = 0; i < actualClusters; i++)
      centroids.col(i) /= counts[i];

    // Assignment step.
    // Find the closest centroid to each point.  We will keep track of how many
    // assignments change.  When no assignments change, we are done.
    changedAssignments = 0;
    for (size_t i = 0; i < data.n_cols; i++)
    {
      // Find the closest centroid to this point.
      double minDistance = std::numeric_limits<double>::infinity();
      size_t closestCluster = actualClusters; // Invalid value.

      for (size_t j = 0; j < actualClusters; j++)
      {
        double distance = metric::SquaredEuclideanDistance::Evaluate(
            data.col(i), centroids.col(j));

        if (distance < minDistance)
        {
          minDistance = distance;
          closestCluster = j;
        }
      }

      // Reassign this point to the closest cluster.
      if (assignments[i] != closestCluster)
      {
        // Update counts.
        counts[assignments[i]]--;
        counts[closestCluster]++;
        // Update assignment.
        assignments[i] = closestCluster;
        changedAssignments++;
      }
    }

    // If we are not allowing empty clusters, then check that all of our
    // clusters have points.
    for (size_t i = 0; i < actualClusters; i++)
      if (counts[i] == 0)
        changedAssignments += emptyClusterAction.EmptyCluster(data, i,
            centroids, counts, assignments);

    iteration++;

  } while (changedAssignments > 0 && iteration != maxIterations);

  // If we have overclustered, we need to merge the nearest clusters.
  if (actualClusters != clusters)
  {
    // Generate a list of all the clusters' distances from each other.  This
    // list will become mangled and unused as the number of clusters decreases.
    size_t numDistances = ((actualClusters - 1) * actualClusters) / 2;
    size_t clustersLeft = actualClusters;
    arma::vec distances(numDistances);
    arma::Col<size_t> firstCluster(numDistances);
    arma::Col<size_t> secondCluster(numDistances);

    // Keep the mappings of clusters that we are changing.
    arma::Col<size_t> mappings = arma::linspace<arma::Col<size_t> >(0,
        actualClusters - 1, actualClusters);

    size_t i = 0;
    for (size_t first = 0; first < actualClusters; first++)
    {
      for (size_t second = first + 1; second < actualClusters; second++)
      {
        distances(i) = metric::SquaredEuclideanDistance::Evaluate(
            centroids.col(first), centroids.col(second));
        firstCluster(i) = first;
        secondCluster(i) = second;
        i++;
      }
    }

    while (clustersLeft != clusters)
    {
      arma::uword minIndex;
      distances.min(minIndex);

      // Now we merge the clusters which that distance belongs to.
      size_t first = firstCluster(minIndex);
      size_t second = secondCluster(minIndex);
      for (size_t j = 0; j < assignments.n_elem; j++)
        if (assignments(j) == second)
          assignments(j) = first;

      // Now merge the centroids.
      centroids.col(first) *= counts[first];
      centroids.col(first) += (counts[second] * centroids.col(second));
      centroids.col(first) /= (counts[first] + counts[second]);

      // Update the counts.
      counts[first] += counts[second];
      counts[second] = 0;

      // Now update all the relevant distances.
      // First the distances where either cluster is the second cluster.
      for (size_t cluster = 0; cluster < second; cluster++)
      {
        // The offset is sum^n i - sum^(n - m) i, where n is actualClusters and
        // m is the cluster we are trying to offset to.
        size_t offset = (size_t) (((actualClusters - 1) * cluster)
            + (cluster - pow(cluster, 2.0)) / 2) - 1;

        // See if we need to update the distance from this cluster to the first
        // cluster.
        if (cluster < first)
        {
          // Make sure it isn't already DBL_MAX.
          if (distances(offset + (first - cluster)) != DBL_MAX)
            distances(offset + (first - cluster)) =
                metric::SquaredEuclideanDistance::Evaluate(
                centroids.col(first), centroids.col(cluster));
        }

        distances(offset + (second - cluster)) = DBL_MAX;
      }

      // Now the distances where the first cluster is the first cluster.
      size_t offset = (size_t) (((actualClusters - 1) * first)
          + (first - pow(first, 2.0)) / 2) - 1;
      for (size_t cluster = first + 1; cluster < actualClusters; cluster++)
      {
        // Make sure it isn't already DBL_MAX.
        if (distances(offset + (cluster - first)) != DBL_MAX)
        {
          distances(offset + (cluster - first)) =
              metric::SquaredEuclideanDistance::Evaluate(
              centroids.col(first), centroids.col(cluster));
        }
      }

      // Max the distance between the first and second clusters.
      distances(offset + (second - first)) = DBL_MAX;

      // Now max the distances for the second cluster (which no longer has
      // anything in it).
      offset = (size_t) (((actualClusters - 1) * second) 
			 + (second - pow(second, 2.0)) / 2) - 1;
      for (size_t cluster = second + 1; cluster < actualClusters; cluster++)
        distances(offset + (cluster - second)) = DBL_MAX;

      clustersLeft--;

      // Update the cluster mappings.
      mappings(second) = first;
      // Also update any mappings that were pointed at the previous cluster.
      for (size_t cluster = 0; cluster < actualClusters; cluster++)
        if (mappings(cluster) == second)
          mappings(cluster) = first;
    }

    // Now remap the mappings down to the smallest possible numbers.
    // Could this process be sped up?
    arma::Col<size_t> remappings(actualClusters);
    remappings.fill(actualClusters);
    size_t remap = 0; // Counter variable.
    for (size_t cluster = 0; cluster < actualClusters; cluster++)
    {
      // If the mapping of the current cluster has not been assigned a value
      // yet, we will assign it a cluster number.
      if (remappings(mappings(cluster)) == actualClusters)
      {
        remappings(mappings(cluster)) = remap;
        remap++;
      }
    }

    // Fix the assignments using the mappings we created.
    for (size_t j = 0; j < assignments.n_elem; j++)
      assignments(j) = remappings(mappings(assignments(j)));
  }
}

}; // namespace gmm
}; // namespace mlpack
