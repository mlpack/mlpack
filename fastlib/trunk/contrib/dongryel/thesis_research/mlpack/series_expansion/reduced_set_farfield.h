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
class ReducedSetFarField {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

    /** @brief The random permutation of indices to assist in building
     *         the dictionary.
     */
    std::vector<int> random_permutation_;

    /** @brief Tells whether each depth-first indexed point is in the
     *         dictionary or not.
     */
    std::vector<bool> in_dictionary_;

    /** @brief The list of point depth-first indices in the
     *         dictionary.
     */
    std::vector<int> point_indices_in_dictionary_;

    /** @brief Maps each depth-first index to the index in the
     *         dictionary. -1 if the point does not exist in the
     *         dictionary.
     */
    std::vector<int> training_index_to_dictionary_position_;

    /** @brief The current kernel matrix.
     */
    core::table::DenseMatrix *current_kernel_matrix_;

    /** @brief The inverse of the current kernel matrix.
     */
    core::table::DenseMatrix *current_kernel_matrix_inverse_;

  private:

    void UpdateDictionary_(
      int new_point_index,
      const Vector &new_column_vector,
      double self_value,
      double projection_error,
      const Vector &inverse_times_column_vector);

  public:

    void inactive_indices(std::vector<int> *inactive_indices_out) const;

    bool in_dictionary(int training_point_index) const;

    int position_to_training_index_map(int position) const;

    int training_index_to_dictionary_position(int training_index) const;

    int point_indices_in_dictionary(int nth_dictionary_point_index) const;

    void Init(const Matrix *table_in);

    void AddBasis(
      int new_point_index,
      const std::vector<double> &new_column_vector_in,
      double self_value);

    const std::vector<int> &random_permutation() const;

    const std::deque<bool> &in_dictionary() const;

    const std::vector<int> &point_indices_in_dictionary() const;

    const std::vector<int> &training_index_to_dictionary_position() const;

    const Matrix *current_kernel_matrix() const;

    Matrix *current_kernel_matrix();

    const Matrix *current_kernel_matrix_inverse() const;

    Matrix *current_kernel_matrix_inverse();

    int size() const;

  public:

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
    template<typename KernelAuxType, typename TreeIteratorType>
    void AccumulateCoeffs(
      const KernelAuxType &kernel_aux_in,
      const core::table::DensePoint &weights,
      TreeIteratorType &it, int order);

    /** @brief Evaluates the far-field coefficients at the given point.
     */
    template<typename KernelAuxType>
    double EvaluateField(
      const KernelAuxType &kernel_aux_in,
      const core::table::DensePoint &point) const;

    template<typename TreeIteratorType>
    void Init(const TreeIteratorType &it);

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
    template<typename KernelAuxType>
    void TranslateFromFarField(
      const KernelAuxType &kernel_aux_in,
      const ReducedSetFarField &se);

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
