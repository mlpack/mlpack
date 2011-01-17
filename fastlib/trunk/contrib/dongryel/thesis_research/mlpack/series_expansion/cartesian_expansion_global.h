/** @file cartesian_expansion_global.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_CARTESIAN_EXPANSION_GLOBAL_H
#define MLPACK_SERIES_EXPANSION_CARTESIAN_EXPANSION_GLOBAL_H

#include <vector>
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"
#include "mlpack/series_expansion/cartesian_expansion_type.h"

namespace mlpack {
namespace series_expansion {

/** @brief The set of global mappings and constants necessary for
 *         performing a Cartesian series expansion of a pairwise
 *         kernel.
 */
template<enum mlpack::series_expansion::CartesianExpansionType>
class CartesianExpansionGlobal {

  private:

    /** @brief The dimensionality of the Cartesian expansion.
     */
    int dim_;

    /** @brief The maximum allowable truncation order.
     */
    int max_order_;

    /** @brief The list of precomputed factorial values.
     */
    core::table::DensePoint factorials_;

    /** @brief The list of precomputed total number of coefficients
     *         per each order.
     */
    std::vector<int> list_total_num_coeffs_;

    /** @brief The list of precomputed inverse multiindex factorial
     *         values.
     */
    core::table::DensePoint inv_multiindex_factorials_;

    /** @brief The list of precomputed inverse multiindex factorial
     *         values with the parity equal to the multiindex
     *         sum. $(-1)^{\alpha} / \alpha !$.
     */
    core::table::DensePoint neg_inv_multiindex_factorials_;

    /** @brief The list of precomputed multiindex combination $\alpha
     *         choose \beta$'s.
     */
    core::table::DenseMatrix multiindex_combination_;

    /** @brief The list of multiindices per each position in the
     *         coefficient layout.
    */
    std::vector< std::vector<short int> > multiindex_mapping_;

    /** @brief For each i-th multiindex m_i, store the positions of
     *         the j-th multiindex mapping such that m_i - m_j >= 0
     *         (the difference in all coordinates is nonnegative).
     */
    std::vector< std::vector<short int> > lower_mapping_index_;

    /** @brief For each i-th multiindex m_i, store the positions of
     *         the j-th multiindex mapping such that m_i - m_j <= 0
     *         (the difference in all coordinates is nonpositive).
     */
    std::vector< std::vector<short int> > upper_mapping_index_;

    /** @brief Stores the $N \choose K$'s. The row index is for n,
     *         column index is for k.
     */
    core::table::DenseMatrix n_choose_k_;

    /** @brief For each i-th multiindex m_i, store the positions of
     *         the j-th multiindex mapping such that m_i - m_j <= 0
     *         (the difference in all coordinates is
     *         nonpositive). This is only for $O(p^D)$ expansion.
     */
    std::vector< std::vector<short int> > traversal_mapping_;

  private:

    /** @brief Compute the list of factorials.
     */
    void ComputeFactorials_();

    void ComputeLowerMappingIndex_();

    void ComputeMultiindexCombination_();

    void ComputeUpperMappingIndex_();

    /** @brief Computes the traversal mapping. This is only for
     *         $O(p^D)$ expansions.
     */
    void ComputeTraversalMapping_();

  public:

    double factorial(int k) const;

    int get_dimension() const;

    int get_total_num_coeffs(int order) const;

    int get_max_total_num_coeffs() const;

    const core::table::DensePoint& get_inv_multiindex_factorials() const;

    const std::vector< short int > * get_lower_mapping_index() const;

    int get_max_order() const;

    const std::vector< short int > & get_multiindex(int pos) const;

    const std::vector< short int > * get_multiindex_mapping() const;

    const core::table::DensePoint& get_neg_inv_multiindex_factorials() const;

    double get_n_choose_k(int n, int k) const;

    double get_n_multichoose_k_by_pos(int n, int k) const;

    const std::vector< short int > * get_upper_mapping_index() const;

    /** @brief Computes the position of the given multiindex.
     */
    int ComputeMultiindexPosition(
      const std::vector<short int> &multiindex) const;

    /** @brief Computes the computational cost of evaluating a far-field
     *         expansion of order p at a single query point.
     */
    double FarFieldEvaluationCost(int order) const;

    /** @brief Computes the compuational cost of translating a far-field
     *         moment of order p into a local moment of the same order.
     */
    double FarFieldToLocalTranslationCost(int order) const;

    /** @brief Computes the computational cost of directly accumulating
     *         a single reference point into a local moment of order p.
     */
    double DirectLocalAccumulationCost(int order) const;

    /** @brief Initialize the auxiliary object with precomputed
     *         quantities for order up to max_order for the given
     *         dimensionality.
     */
    void Init(int max_order, int dim);

    /** @brief Print useful information about this object.
     */
    void Print(const char *name = "", FILE *stream = stderr) const;
};
}
}

#endif
