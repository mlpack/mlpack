/**
 * @file kmeans_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * Implementation for the K-means method for getting an initial point.
 */
#include "kmeans.hpp"

#include <mlpack/core/tree/mrkd_statistic.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include <stack>
#include <limits>

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

  size_t dimensionality = data.n_rows;

  // Centroids of each cluster.  Each column corresponds to a centroid.
  MatType centroids(dimensionality, actualClusters);
  centroids.zeros();

  // Counts of points in each cluster.
  arma::Col<size_t> counts(actualClusters);
  counts.zeros();

  // Build the mrkd-tree on this dataset.
  tree::BinarySpaceTree<typename bound::HRectBound<2>, tree::MRKDStatistic>
      tree(data, 1);
  Log::Debug << "Tree Built." << std::endl;
  // A pointer for traversing the mrkd-tree.
  tree::BinarySpaceTree<typename bound::HRectBound<2>, tree::MRKDStatistic>*
      node;

  // Now, the initial assignments.  First determine if they are necessary.
  if (assignments.n_elem != data.n_cols)
  {
    // Use the partitioner to come up with the partition assignments.
    partitioner.Cluster(data, actualClusters, assignments);
  }

  // Set counts correctly.
  for (size_t i = 0; i < assignments.n_elem; i++)
    counts[assignments[i]]++;

  // Sum the points for each centroid
  for (size_t i = 0; i < data.n_cols; i++)
    centroids.col(assignments[i]) += data.col(i);

  // Then divide the sums by the count to get the center of mass for this
  // centroids assigned points
  for (size_t i = 0; i < actualClusters; i++)
    centroids.col(i) /= counts[i];

  // Instead of retraversing the tree after an iteration, we will update
  // centroid positions in this matrix, which also prevents clobbering our
  // centroids from the previous iteration.
  MatType newCentroids(dimensionality, centroids.n_cols);

  // Create a stack for traversing the mrkd-tree.
  std::stack<typename tree::BinarySpaceTree<typename bound::HRectBound<2>,
                                            tree::MRKDStatistic>* > stack;

  // A variable to keep track of how many kmeans iterations we have made.
  size_t iteration = 0;

  // A variable to keep track of how many nodes assignments have changed in
  // each kmeans iteration.
  size_t changedAssignments = 0;

  // A variable to keep track of the number of times something is skipped due
  // to the blacklist.
  size_t skip = 0;

  // A variable to keep track of the number of distances calculated.
  size_t comps = 0;

  // A variable to keep track of how often we stop at a parent node.
  size_t dominations = 0;
  do
  {
    // Keep track of what iteration we are on.
    ++iteration;
    changedAssignments = 0;

    // Reset the newCentroids so that we can store the newly calculated ones
    // here.
    newCentroids.zeros();

    // Reset the counts.
    counts.zeros();

    // Add the root node of the tree to the stack.
    stack.push(&tree);
    // Set the top level whitelist.
    tree.Stat().Whitelist().resize(centroids.n_cols, true);

    // Traverse the tree.
    while (!stack.empty())
    {
      // Get the next node in the tree.
      node = stack.top();
      // Remove the node from the stack.
      stack.pop();

      // Get a reference to the mrkd statistic for this hyperrectangle.
      tree::MRKDStatistic& mrkd = node->Stat();

      // We use this to store the index of the centroid with the minimum
      // distance from this hyperrectangle or point.
      size_t minIndex = 0;

      // If this node is a leaf, then we calculate the distance from
      // the centroids to every point the node contains.
      if (node->IsLeaf())
      {
        for (size_t i = mrkd.Begin(); i < mrkd.Count() + mrkd.Begin(); ++i)
        {
          // Initialize minDistance to be nonzero.
          double minDistance = metric::SquaredEuclideanDistance::Evaluate(
              data.col(i), centroids.col(0));

          // Find the minimal distance centroid for this point.
          for (size_t j = 1; j < centroids.n_cols; ++j)
          {
            // If this centroid is not in the whitelist, skip it.
            if (!mrkd.Whitelist()[j])
            {
              ++skip;
              continue;
            }

            ++comps;
            double distance = metric::SquaredEuclideanDistance::Evaluate(
                data.col(i), centroids.col(j));
            if (minDistance > distance)
            {
              minIndex = j;
              minDistance = distance;
            }
          }

          // Add this point to the undivided center of mass summation for its
          // assigned centroid.
          newCentroids.col(minIndex) += data.col(i);

          // Increment the count for the minimum distance centroid.
          ++counts(minIndex);

          // If we actually changed assignments, increment changedAssignments
          // and modify the assignment vector for this point.
          if (assignments(i) != minIndex)
          {
            ++changedAssignments;
            assignments(i) = minIndex;
          }
        }
      }
      // If this node is not a leaf, then we continue trying to find dominant
      // centroids.
      else
      {
        bound::HRectBound<2>& bound = node->Bound();

        // A flag to keep track of if we find a single centroid that is closer
        // to all points in this hyperrectangle than any other centroid.
        bool noDomination = false;

        // Calculate the center of mass of this hyperrectangle.
        arma::vec center = mrkd.CenterOfMass() / mrkd.Count();

        // Set the minDistance to the maximum value of a double so any value
        // must be smaller than this.
        double minDistance = std::numeric_limits<double>::max();

        // The candidate distance we calculate for each centroid.
        double distance = 0.0;

        // How many points are inside this hyperrectangle, we stop if we
        // see more than 1.
        size_t contains = 0;

        // Find the "owner" of this hyperrectangle, if one exists.
        for (size_t i = 0; i < centroids.n_cols; ++i)
        {
          // If this centroid is not in the whitelist, skip it.
          if (!mrkd.Whitelist()[i])
          {
            ++skip;
            continue;
          }

          // Incrememnt the number of distance calculations for what we are
          // about to do.
          comps += 2;

          // Reinitialize the distance so += works right.
          distance = 0.0;

          // We keep track of how many dimensions have nonzero distance,
          // if this is 0 then the distance is 0.
          size_t nonZero = 0;

          /*
            Compute the distance to the hyperrectangle for this centroid.
            We do this by finding the furthest point from the centroid inside
            the hyperrectangle. This is a corner of the hyperrectangle.

            In order to do this faster, we calculate both the distance and the
            furthest point simultaneously.

            This following code is equivalent to, but faster than:

            arma::vec p;
            p.zeros(dimensionality);

            for (size_t j = 0; j < dimensionality; ++j)
            {
              if (centroids(j,i) < bound[j].Lo())
                p(j) = bound[j].Lo();
              else
                p(j) = bound[j].Hi();
            }

            distance = metric::SquaredEuclideanDistance::Evaluate(
                p.col(0), centroids.col(i));
          */
          for (size_t j = 0; j < dimensionality; ++j)
          {
            double ij = centroids(j,i);
            double lo = bound[j].Lo();

            if (ij < lo)
            {
              // (ij - lo)^2
              ij -= lo;
              ij *= ij;

              distance += ij;
              ++nonZero;
            }
            else
            {
              double hi = bound[j].Hi();
              if (ij > hi)
              {
                // (ij - hi)^2
                ij -= hi;
                ij *= ij;

                distance += ij;
                ++nonZero;
              }
            }
          }

          // The centroid is inside the hyperrectangle.
          if (nonZero == 0)
          {
            ++contains;
            minDistance = 0.0;
            minIndex = i;

            // If more than two points are within this hyperrectangle, then
            // there can be no dominating centroid, so we should continue
            // to the children nodes.
            if (contains > 1)
            {
              noDomination = true;
              break;
            }
          }

          if (fabs(distance - minDistance) <= 1e-10)
          {
            noDomination = true;
            break;
          }
          else if (distance < minDistance)
          {
            minIndex = i;
            minDistance = distance;
          }
        }

        distance = minDistance;
        // Determine if the owner dominates this centroid only if there was
        // exactly one owner.
        if (!noDomination)
        {
          for (size_t i = 0; i < centroids.n_cols; ++i)
          {
            if (i == minIndex)
              continue;
            // If this centroid is blacklisted for this hyperrectangle, then
            // we skip it.
            if (!mrkd.Whitelist()[i])
            {
              ++skip;
              continue;
            }
            /*
              Compute the dominating centroid for this hyperrectangle, if one
              exists. We do this by calculating the point which is furthest
              from the min'th centroid in the direction of c_k - c_min. We do
              this as outlined in the Pelleg and Moore paper.

              This following code is equivalent to, but faster than:

              arma::vec p;
              p.zeros(dimensionality);

              for (size_t k = 0; k < dimensionality; ++k)
              {
                p(k) = (centroids(k,i) > centroids(k,minIndex)) ?
                  bound[k].Hi() : bound[k].Lo();
              }

              double distancei = metric::SquaredEuclideanDistance::Evaluate(
                  p.col(0), centroids.col(i));
              double distanceMin = metric::SquaredEuclideanDistance::Evaluate(
                  p.col(0), centroids.col(minIndex));
            */

            comps += 1;
            double distancei = 0.0;
            double distanceMin = 0.0;
            for (size_t k = 0; k < dimensionality; ++k)
            {
              double ci = centroids(k, i);
              double cm = centroids(k, minIndex);
              if (ci > cm)
              {
                double hi = bound[k].Hi();

                ci -= hi;
                cm -= hi;

                ci *= ci;
                cm *= cm;

                distancei += ci;
                distanceMin += cm;
              }
              else
              {
                double lo = bound[k].Lo();

                ci -= lo;
                cm -= lo;

                ci *= ci;
                cm *= cm;

                distancei += ci;
                distanceMin += cm;
              }
            }

            if (distanceMin >= distancei)
            {
              noDomination = true;
              break;
            }
            else
            {
              mrkd.Whitelist()[i] = false;
            }
          }
        }

        // If did found a centroid that was closer to every point in the
        // hyperrectangle than every other centroid, then update that centroid.
        if (!noDomination)
        {
          // Adjust the new centroid sum for the min distance point to this
          // hyperrectangle by the center of mass of this hyperrectangle.
          newCentroids.col(minIndex) += mrkd.CenterOfMass();

          // Increment the counts for this centroid.
          counts(minIndex) += mrkd.Count();

          // Update all assignments for this node.
          const size_t begin = node->Begin();
          const size_t end = node->End();

          // TODO: Do this outside of the kmeans iterations.
          for (size_t j = begin; j < end; ++j)
          {
            if (assignments(j) != minIndex)
            {
              ++changedAssignments;
              assignments(j) = minIndex;
            }
          }
          mrkd.DominatingCentroid() = minIndex;

          // Keep track of the number of times we found a dominating centroid.
          ++dominations;
        }

        // If we did not find a dominating centroid then we fall through to the
        // default case, where we add the children of this node to the stack.
        else
        {
          // Add this hyperrectangle's children to our stack.
          stack.push(node->Left());
          stack.push(node->Right());

          // (Re)Initialize the whiteList for the children.
          node->Left()->Stat().Whitelist() = mrkd.Whitelist();
          node->Right()->Stat().Whitelist() = mrkd.Whitelist();
        }
      }

    }

    // Divide by the number of points assigned to the centroids so that we
    // have the actual center of mass and update centroids' positions.
    for (size_t i = 0; i < centroids.n_cols; ++i)
      if (counts(i))
        centroids.col(i) = newCentroids.col(i) / counts(i);

    // Stop when we reach max iterations or we changed no assignments
    // assignments.
  } while (changedAssignments > 0 && iteration != maxIterations);

  Log::Info << "Iterations: " << iteration << std::endl
      << "Skips: " << skip << std::endl
      << "Comparisons: " << comps << std::endl
      << "Dominations: " << dominations << std::endl;
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

  Log::Debug << "Iterations: " << iteration << std::endl;

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

}; // namespace kmeans
}; // namespace mlpack
