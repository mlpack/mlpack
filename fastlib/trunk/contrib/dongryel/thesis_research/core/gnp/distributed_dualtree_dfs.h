/** @file distributed_dualtree_dfs.h
 *
 *  The prototype header for performing a distributed pairwise GNPs.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DISTRIBUTED_DUALTREE_DFS_H
#define CORE_GNP_DISTRIBUTED_DUALTREE_DFS_H

#include <boost/mpi/communicator.hpp>
#include <boost/tuple/tuple.hpp>
#include "core/math/range.h"

namespace core {
namespace gnp {
template<typename DistributedProblemType>
class DistributedDualtreeDfs {

  public:

    typedef typename DistributedProblemType::TableType TableType;
    typedef typename DistributedProblemType::ProblemType ProblemType;
    typedef typename DistributedProblemType::DistributedTableType
    DistributedTableType;
    typedef typename TableType::TreeType TreeType;
    typedef typename DistributedTableType::TreeType DistributedTreeType;
    typedef typename DistributedProblemType::GlobalType GlobalType;
    typedef typename DistributedProblemType::ResultType ResultType;
    typedef typename DistributedProblemType::ArgumentType ArgumentType;

    typedef boost::tuple <
    TreeType *, std::pair<int, int>, double > FrontierObjectType;

  private:

    boost::mpi::communicator *world_;

    DistributedProblemType *problem_;

    DistributedTableType *query_table_;

    DistributedTableType *reference_table_;

  private:

    template<typename MetricType>
    void AllToAllReduce_(
      const MetricType &metric,
      typename DistributedProblemType::ResultType *query_results);

    void ResetStatisticRecursion_(
      DistributedTreeType *node, DistributedTableType * table);

    template<typename TemplateTreeType>
    void PreProcessReferenceTree_(TemplateTreeType *rnode);

    template<typename TemplateTreeType>
    void PreProcess_(TemplateTreeType *qnode);

    template<typename MetricType>
    void PostProcess_(
      const MetricType &metric,
      TreeType *qnode, ResultType *query_results);

    class PrioritizeTasks_:
      public std::binary_function <
        FrontierObjectType &, FrontierObjectType &, bool > {
      public:
        bool operator()(
          const FrontierObjectType &a, const FrontierObjectType &b) const {
          return a.get<2>() > b.get<2>();
        }
    };

  public:

    DistributedProblemType *problem();

    DistributedTableType *query_table();

    DistributedTableType *reference_table();

    void ResetStatistic();

    void Init(
      boost::mpi::communicator *world, DistributedProblemType &problem_in);

    template<typename MetricType>
    void Compute(
      const MetricType &metric,
      typename DistributedProblemType::ResultType *query_results);
};
}
}

#endif
