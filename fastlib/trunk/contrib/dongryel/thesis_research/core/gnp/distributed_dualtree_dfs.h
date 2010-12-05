/** @file distributed_dualtree_dfs.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DISTRIBUTED_DUALTREE_DFS_H
#define CORE_GNP_DISTRIBUTED_DUALTREE_DFS_H

#include <boost/mpi/communicator.hpp>
#include "core/metric_kernels/abstract_metric.h"
#include "core/math/range.h"

namespace core {
namespace gnp {
template<typename ProblemType>
class DistributedDualtreeDfs {

  public:

    typedef typename ProblemType::TableType TableType;
    typedef typename ProblemType::DistributedTableType DistributedTableType;
    typedef typename TableType::TreeType TreeType;
    typedef typename DistributedTableType::TreeType DistributedTreeType;
    typedef typename ProblemType::GlobalType GlobalType;
    typedef typename ProblemType::ResultType ResultType;

  private:

    boost::mpi::communicator *world_;

    ProblemType *problem_;

    DistributedTableType *query_table_;

    DistributedTableType *reference_table_;

  private:

    void AllReduce_();

    void ResetStatisticRecursion_(
      DistributedTreeType *node, DistributedTableType * table);

    template<typename TemplateTreeType>
    void PreProcessReferenceTree_(TemplateTreeType *rnode);

    template<typename TemplateTreeType>
    void PreProcess_(TemplateTreeType *qnode);

    void PostProcess_(
      const core::metric_kernels::AbstractMetric &metric,
      TreeType *qnode, ResultType *query_results);

  public:

    ProblemType *problem();

    DistributedTableType *query_table();

    DistributedTableType *reference_table();

    void ResetStatistic();

    void Init(boost::mpi::communicator *world, ProblemType &problem_in);

    void Compute(
      const core::metric_kernels::AbstractMetric &metric,
      typename ProblemType::ResultType *query_results);
};
};
};

#endif
