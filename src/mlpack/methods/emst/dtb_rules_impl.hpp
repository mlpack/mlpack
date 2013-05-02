/**
 * @file dtb_impl.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Tree traverser rules for the DualTreeBoruvka algorithm.
 *
 * This file is part of MLPACK 1.0.5.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef __MLPACK_METHODS_EMST_DTB_RULES_IMPL_HPP
#define __MLPACK_METHODS_EMST_DTB_RULES_IMPL_HPP

namespace mlpack {
namespace emst {

template<typename MetricType, typename TreeType>
DTBRules<MetricType, TreeType>::
DTBRules(const arma::mat& dataSet,
         UnionFind& connections,
         arma::vec& neighborsDistances,
         arma::Col<size_t>& neighborsInComponent,
         arma::Col<size_t>& neighborsOutComponent,
         MetricType& metric)
:
  dataSet(dataSet),
  connections(connections),
  neighborsDistances(neighborsDistances),
  neighborsInComponent(neighborsInComponent),
  neighborsOutComponent(neighborsOutComponent),
  metric(metric)
{
  // nothing else to do
} // constructor

template<typename MetricType, typename TreeType>
double DTBRules<MetricType, TreeType>::BaseCase(const size_t queryIndex,
                                                const size_t referenceIndex)
{
  
  // Check if the points are in the same component at this iteration.
  // If not, return the distance between them.
  // Also responsible for storing this as the current neighbor
  
  double newUpperBound = -1.0;
  
  // Find the index of the component the query is in.
  size_t queryComponentIndex = connections.Find(queryIndex);
  
  size_t referenceComponentIndex = connections.Find(referenceIndex);
  
  if (queryComponentIndex != referenceComponentIndex)
  {
    double distance = metric.Evaluate(dataSet.col(queryIndex),
                                      dataSet.col(referenceIndex));
    
    if (distance < neighborsDistances[queryComponentIndex])
    {
      Log::Assert(queryIndex != referenceIndex);
      
      neighborsDistances[queryComponentIndex] = distance;
      neighborsInComponent[queryComponentIndex] = queryIndex;
      neighborsOutComponent[queryComponentIndex] = referenceIndex;
      
    } // if distance
  } // if indices not equal

  if (newUpperBound < neighborsDistances[queryComponentIndex])
    newUpperBound = neighborsDistances[queryComponentIndex];
  
  Log::Assert(newUpperBound >= 0.0);
  
  return newUpperBound;
  
} // BaseCase()

template<typename MetricType, typename TreeType>
void DTBRules<MetricType, TreeType>::UpdateAfterRecursion(
    TreeType& queryNode,
    TreeType& /* referenceNode */)
{

  // Find the worst distance that the children found (including any points), and
  // update the bound accordingly.
  double newUpperBound = 0.0;
  
  // First look through children nodes.
  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    if (newUpperBound < queryNode.Child(i).Stat().MaxNeighborDistance())
      newUpperBound = queryNode.Child(i).Stat().MaxNeighborDistance();
  }
  
  // Now look through children points.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    size_t pointComponent = connections.Find(queryNode.Point(i));
    if (newUpperBound < neighborsDistances[pointComponent])
      newUpperBound = neighborsDistances[pointComponent];
  }
  
  // Update the bound in the query's stat
  queryNode.Stat().MaxNeighborDistance() = newUpperBound;
  
} // UpdateAfterRecursion

template<typename MetricType, typename TreeType>
double DTBRules<MetricType, TreeType>::Score(const size_t queryIndex,
                                             TreeType& referenceNode)
{
  
  size_t queryComponentIndex = connections.Find(queryIndex);
  
  // If the query belongs to the same component as all of the references,
  // then prune.
  // Casting this to stop a warning about comparing unsigned to signed
  // values.
  if (queryComponentIndex == (size_t)referenceNode.Stat().ComponentMembership())
    return DBL_MAX;
  
  const arma::vec queryPoint = dataSet.unsafe_col(queryIndex);
  
  const double distance = referenceNode.MinDistance(queryPoint);
  
  // If all the points in the reference node are farther than the candidate
  // nearest neighbor for the query's component, we prune.
  return neighborsDistances[queryComponentIndex] < distance
      ? DBL_MAX : distance;
  
} // Score()

template<typename MetricType, typename TreeType>
double DTBRules<MetricType, TreeType>::Score(const size_t queryIndex,
                                             TreeType& referenceNode,
                                             const double baseCaseResult) 
{
  // I don't really understand the last argument here
  // It just gets passed in the distance call, otherwise this function
  // is the same as the one above

  size_t queryComponentIndex = connections.Find(queryIndex);
  
  // if the query belongs to the same component as all of the references,
  // then prune
  if (queryComponentIndex == referenceNode.Stat().ComponentMembership())
    return DBL_MAX;
  
  const arma::vec queryPoint = dataSet.unsafe_col(queryIndex);
  
  const double distance = referenceNode.MinDistance(queryPoint,
                                                    baseCaseResult);
  
  // If all the points in the reference node are farther than the candidate
  // nearest neighbor for the query's component, we prune.
  return neighborsDistances[queryComponentIndex] < distance
      ? DBL_MAX : distance;
  
} // Score()

template<typename MetricType, typename TreeType>
double DTBRules<MetricType, TreeType>::Rescore(const size_t queryIndex,
                                               TreeType& referenceNode,
                                               const double oldScore) 
{
  // We don't need to check component membership again, because it can't
  // change inside a single iteration.
  
  // If we are already pruning, still prune.
  if (oldScore == DBL_MAX)
    return oldScore;

  if (oldScore > neighborsDistances[connections.Find(queryIndex)])
    return DBL_MAX;
  else
    return oldScore;
  
} // Rescore

template<typename MetricType, typename TreeType>
double DTBRules<MetricType, TreeType>::Score(TreeType& queryNode,
                                             TreeType& referenceNode) const
{
  // If all the queries belong to the same component as all the references
  // then we prune.
  if ((queryNode.Stat().ComponentMembership() >= 0)
      && (queryNode.Stat().ComponentMembership() ==
          referenceNode.Stat().ComponentMembership()))
    return DBL_MAX;

  double distance = queryNode.MinDistance(&referenceNode);
  
  // If all the points in the reference node are farther than the candidate
  // nearest neighbor for all queries in the node, we prune.
  return queryNode.Stat().MaxNeighborDistance() < distance
      ? DBL_MAX : distance;
  
} // Score()

template<typename MetricType, typename TreeType>
double DTBRules<MetricType, TreeType>::Score(TreeType& queryNode,
                                             TreeType& referenceNode,
                                             const double baseCaseResult) const
{
  
  // If all the queries belong to the same component as all the references
  // then we prune.
  if ((queryNode.Stat().ComponentMembership() >= 0)
      && (queryNode.Stat().ComponentMembership() ==
          referenceNode.Stat().ComponentMembership()))
    return DBL_MAX;
  
  const double distance = queryNode.MinDistance(referenceNode,
                                                baseCaseResult);
  
  // If all the points in the reference node are farther than the candidate
  // nearest neighbor for all queries in the node, we prune.
  return queryNode.Stat().MaxNeighborDistance() < distance
      ? DBL_MAX : distance;
  
} // Score()

template<typename MetricType, typename TreeType>
double DTBRules<MetricType, TreeType>::Rescore(TreeType& queryNode,
                                               TreeType& /* referenceNode */,
                                               const double oldScore) const
{
  
  // Same as above, but for nodes,
  
  // If we are already pruning, still prune.
  if (oldScore == DBL_MAX)
    return oldScore;
  
  if (oldScore > queryNode.Stat().MaxNeighborDistance())
    return DBL_MAX;
  else
    return oldScore;
  
} // Rescore

} // namespace emst
} // namespace mlpack



#endif 

