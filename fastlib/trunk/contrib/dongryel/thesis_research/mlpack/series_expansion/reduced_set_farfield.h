/** @file reduced_set_farfield.h
 *
 *  Reduced set expansion based on Smola/Bengio's method.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_REDUCED_SET_FARFIELD_H
#define MLPACK_SERIES_EXPANSION_REDUCED_SET_FARFIELD_H

#include <boost/serialization/serialization.hpp>
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"

namespace mlpack {
namespace series_expansion {

/** @brief Reduced set expansion based on Smola/Bengio's method.
 */
template<typename TreeIteratorType>
class ReducedSetFarField {

  public:
    typedef std::vector <
    std::pair<core::table::DensePoint *, int> > DictionaryType;

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

    /** @brief The current kernel matrix.
     */
    core::table::DenseMatrix *current_kernel_matrix_;

    /** @brief The inverse of the current kernel matrix.
     */
    core::table::DenseMatrix *current_kernel_matrix_inverse_;

    /** @brief The projection matrix that projects each point to the
     *         span of the dictionary points.
     */
    core::table::DenseMatrix projection_matrix_;

    /** @brief Tells whether each row of the projection matrix belongs
     *         to the dictionary or not.
     */
    std::vector<bool> in_dictionary_;

    /** @brief The dictionary points and their DFS indices.
     */
    DictionaryType dictionary_;

    /** @brief The number of compressed points.
     */
    int num_compressed_points_;

    /** @brief The list of child expansions (for implementing the
     *         far-to-far translation operators).
     */
    std::vector <
    const mlpack::series_expansion::ReducedSetFarField <
    TreeIteratorType > * > child_expansions_;

  private:

    /** @brief Fills out a vector of kernel values for a given point
     *         against the pre-existing dictionary points.
     */
    template <
    typename MetricType, typename KernelAuxType >
    void FillKernelValues_(
      const MetricType &metric_in,
      const KernelAuxType &kernel_aux_in,
      const core::table::DensePoint &candidate,
      arma::vec *kernel_values_out,
      double *self_value) const;

    void UpdateDictionary_(
      const core::table::DensePoint &new_point,
      int new_point_index,
      const TreeIteratorType &it,
      const arma::vec &new_column_vector,
      double self_value,
      double projection_error,
      const arma::vec &inverse_times_column_vector);

    void AddBasis_(
      const core::table::DensePoint &new_point,
      int new_point_index,
      const TreeIteratorType &it,
      const arma::vec &new_column_vector_in,
      double self_value);

    /** @brief Finalize the compression by computing the final
     *         mapping.
     */
    template<typename MetricType, typename KernelAuxType>
    void FinalizeCompression_(
      const MetricType &metric_in,
      const KernelAuxType &kernel_aux_in,
      TreeIteratorType &it);

  public:

    const DictionaryType &dictionary() const;

    void Init(const TreeIteratorType &it);

    const core::table::DenseMatrix *current_kernel_matrix() const;

    core::table::DenseMatrix *current_kernel_matrix();

    const core::table::DenseMatrix *current_kernel_matrix_inverse() const;

    core::table::DenseMatrix *current_kernel_matrix_inverse();

  public:

    /** @brief Returns the number of compressed points.
     */
    int num_compressed_points() const;

    /** @brief Serializes the far field object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }

    /** @brief The constructor.
     */
    ReducedSetFarField();

    /** @brief The destructor.
     */
    ~ReducedSetFarField();

    /** @brief Accumulates the far field moment represented by the given
     *         reference data into the coefficients.
     */
    template<typename MetricType, typename KernelAuxType>
    void AccumulateCoeffs(
      const MetricType &metric_in,
      const KernelAuxType &kernel_aux_in,
      TreeIteratorType &it);

    /** @brief Evaluates the far-field coefficients at the given point.
     */
    template <
    typename MetricType, typename KernelAuxType >
    double EvaluateField(
      const MetricType &metric_in,
      const KernelAuxType &kernel_aux_in,
      const core::table::DensePoint &point,
      TreeIteratorType &reference_it) const;

    /** @brief Prints out the series expansion represented by this object.
     */
    template<typename KernelAuxType>
    void Print(
      const KernelAuxType &kernel_aux_in,
      const char *name = "", FILE *stream = stderr) const;

    /** @brief Translate from a far field expansion to the expansion
     *         here. The translated coefficients are added up to the
     *         ones here.
     */
    template <
    typename MetricType, typename KernelAuxType >
    void TranslateFromFarField(
      const MetricType &metric_in,
      const KernelAuxType &kernel_aux_in,
      const ReducedSetFarField &se,
      TreeIteratorType &it);

    /** @brief Translate to the given local expansion. The translated
     *         coefficients are added up to the passed-in local
     *         expansion coefficients.
     */
    template<typename KernelAuxType, typename ReducedSetLocalType>
    void TranslateToLocal(
      const KernelAuxType &kernel_aux_in,
      int truncation_order, ReducedSetLocalType *se) const;
};
}
}

#endif
