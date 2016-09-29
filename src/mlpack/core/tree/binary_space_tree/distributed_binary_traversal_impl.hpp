/**
 * @file distributed_binary_traversal_impl.hpp
 * @author Ryan Curtin
 *
 * Use MPI to perform a distributed traversal.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_BINARY_TRAVERSAL_IMPL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_BINARY_TRAVERSAL_IMPL_HPP

#include "distributed_binary_traversal.hpp"
#include "../binary_space_tree.hpp"
#include "dual_tree_traverser.hpp"
#include <boost/mpi.hpp>

namespace mlpack {
namespace tree {

template<typename RuleType>
DistributedBinaryTraversal<RuleType>::DistributedBinaryTraversal(
    RuleType& rule) :
    rule(&rule),
    world()
{
  // Nothing to do.
}

template<typename RuleType>
DistributedBinaryTraversal<RuleType>::DistributedBinaryTraversal() :
    rule(NULL),
    world()
{
  // We are an MPI child.  We must receive and construct our own RuleType
  // object, query tree, and reference tree.  Once we have done that, we kick
  // off the usual recursion, and when we're done, we send the results back.
  typename RuleType::MPIWrapper wrapper;
  Log::Info << "Process " << world.rank() << " is waiting for a message.\n";
  Timer::Start("child_receive");
  world.recv(0, 0, wrapper);
  Timer::Stop("child_receive");
  Log::Info << "Process " << world.rank() << " has received a message.\n";


  // We've now received our information.  Start the recursion.
  this->rule = wrapper.Rules();
  Timer::Start("child_traversal");
  Traverse(*wrapper.QueryTree(), *wrapper.ReferenceTree());
  Timer::Stop("child_traversal");

  // Now, we have to ship the neighbors and distances back to the master.
  typename RuleType::MPIResultsWrapper resultsWrapper(rule->Neighbors(),
                                                      rule->Distances());
  Log::Info << "Process " << world.rank() << " is sending results.\n";
  Timer::Start("send_results");
  world.send(0, 0, resultsWrapper);
  Timer::Stop("send_results");
  Log::Info << "Process " << world.rank() << " is finished.\n";
}

template<typename RuleType>
template<typename TreeType>
void DistributedBinaryTraversal<RuleType>::Traverse(const size_t queryIndex,
                                                    TreeType& referenceNode)
{

}

template<typename RuleType>
template<typename TreeType>
void DistributedBinaryTraversal<RuleType>::Traverse(TreeType& queryNode,
                                                    TreeType& referenceNode)
{
  // If we are the master, call the master traversal.  Otherwise, call the child
  // traversal.
  if (world.rank() == 0)
  {
    // Start the traversal, and pass the work to the children.
    MasterTraverse(queryNode, referenceNode);
  }
  else
  {
    ChildTraverse(queryNode, referenceNode);
  }
}

template<typename RuleType>
template<typename TreeType>
void DistributedBinaryTraversal<RuleType>::MasterTraverse(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  // A list of jobs to be done.
  std::queue<std::pair<TreeType*, TreeType*>> jobs;
  jobs.push(&queryNode, &referenceNode);

  // A list of which nodes are busy and which aren't.
  std::vector<bool> busy(world.size() - 1, false);

  while (!jobs.empty())
  {
    // Find an unused worker (wait for a response).
    RuleType::MPIResultType result;
    boost::mpi::status status;
    status = communicator.recv(boost::mpi::any_source, boost::mpi::any_tag,
        result);

    // Immediately put that worker back to work on a new job.
    RuleType::MPIWorkType work(queue.front().first, queue.front().second);
    communicator.send(status.source(), 0 /* zero tag */, work);

    if (result.tag() == 1) // Initialization tag; no data.
    {
      // Now, look through the results to add new jobs.
      const RuleType::MPIWorkType& job = jobs[status.source()];
      for (size_t i = 0; i < result.NumNewTasks(); ++i)
        jobs.push_back(result.NewTask(job.QueryNode(), job.ReferenceNode()));

      // And merge the results into the tree that we have.
      for (size_t i = 0; i < result.NumTreeUpdates(); ++i)
        result.MergeResult(job.QueryNode(), job.ReferenceNode());
    }
  }
}

template<typename RuleType>
template<typename TreeType>
void DistributedBinaryTraversal<RuleType>::ChildTraverse(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  // We'll just call out to the standard dual-tree traversal for a single node.
  typename TreeType::template DualTreeTraverser<RuleType> traverser(*rule);

  traverser.Traverse(queryNode, referenceNode);
}

template<typename RuleType>
template<typename TreeType>
size_t DistributedBinaryTraversal<RuleType>::GetTarget(
    TreeType& queryNode,
    TreeType& referenceNode) const
{
  // We assemble the ID of the target process in a bitwise manner.  The leftmost
  // combination maps to process 0.  At any level of recursion, because this is
  // a binary recursion, the query node may be either the left (L) child or the
  // right (R) child, and the same applies to the reference node.  Thus the
  // direction we have gone at a recursion can have four possibilities: LL, LR,
  // RL, and RR.  Take L = 0 and R = 1; now a single recursion can be
  // represented as two bits.  The highest-level recursion will be the two most
  // significant bits and the most recent recursion will be the two least
  // significant bits.  Thus, if the most recent recursion was RL and the
  // higher-level recursion was LR, and there were no higher recursions than
  // that, the index will be LRRL -> 0110 -> 6.  If any recursion was not a dual
  // recursion, undefined behavior will happen.  It probably won't crash.
  size_t index = 0;

  TreeType* currentQuery = &queryNode;
  TreeType* currentRef = &referenceNode;
  size_t level = 0;
  while (currentQuery->Parent() != NULL && currentRef->Parent() != NULL)
  {
    // Assemble this index.
    size_t currentIndex = 0; // Assume LL, change if otherwise.
    if (currentQuery->Parent()->Right() == currentQuery)
      currentIndex += 2; // Now it's RL.
    if (currentRef->Parent()->Right() == currentRef)
      currentIndex++; // Now it's LR or RR.

    // Append this index.
    index += (currentIndex << (level * 2));
    ++level;

    currentQuery = currentQuery->Parent();
    currentRef = currentRef->Parent();
  }

  return index + 1; // Index 0 is the root.
}

} // namespace tree
} // namespace mlpack

#endif
