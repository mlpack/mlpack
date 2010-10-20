#ifndef FL_LITE_MLPACK_KCDE_MULTITREE_MONTE_CARLO_H
#define FL_LITE_MLPACK_KCDE_MULTITREE_MONTE_CARLO_H

#include <vector>
#include "boost/math/distributions/normal.hpp"

namespace fl {
namespace ml {

template<typename TableType>
class MultitreeMonteCarlo {

  public:

    typedef typename TableType::Tree_t Tree_t;

    typedef typename TableType::Dataset_t::Point_t Point_t;

  private:

    /** @brief The list of child problems that need to be solved.
     */
    std::vector< MultitreeMonteCarlo<TableType> *> subproblems_;

    /* @brief The function that needs to be applied at the
     *        current problem level.
     */

    /** @brief The list of constant points and the corresponding
    *         indices.
    */
    std::vector< std::pair<TableType *, int> > constant_arguments_;

    /** @brief The list of tables for variable arguments for which the
     *         stratification has to be performed. Each table is
     *         paired with a list of indices which should not be
     *         sampled from the table.
     */
    std::vector< std::pair<TableType *, std::vector<int> > >
    tables_for_variable_arguments_;

    double relative_error_;

    double probability_;

    boost::math::normal normal_dist_;

    double num_standard_deviations_;

  private:

    bool Converged_(
      const std::pair<double, double> &summary_mean_variance_pair) const;

    void Stratify_();

  public:

    void AddSubProblem(MultitreeMonteCarlo<TableType> &subproblem_in);

    std::vector< MultitreeMonteCarlo<TableType> *> &subproblems();

    const std::vector< std::pair<TableType *, int> > &constant_arguments() const;

    std::vector< std::pair<TableType *, int> > &constant_arguments();

    const std::vector< std::pair<TableType *, std::vector<int> > >
    &tables_for_variable_arguments() const;

    std::vector< std::pair<TableType *, std::vector<int> > >
    &tables_for_variable_arguments();

    template<typename PointType>
    void get(int variable_argument_index, int point_index,
             PointType *point_out);

    void clear_constant_arguments();

    void set_constant_argument(int constant_argument_index, int point_index);

    void add_constant_argument(TableType *constant_argument_in);

    void add_variable_argument(
      TableType &variable_argument,
      const std::vector<int> &leave_one_out_dataset_indices);

    void set_error(double relative_error_in, double probability_in);

    template<typename FunctionType>
    void Compute(
      const FunctionType &function_in,
      std::vector< std::pair<double, double> > *mean_variance_pair_out);
};
};
};

#endif
