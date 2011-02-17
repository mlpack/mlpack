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

  public:

    /** @brief Serializes the far field object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }

    ////////// Getters/Setters //////////

    /** @brief Gets the center of expansion.
     *
     *  @return The center of expansion for the current far-field expansion.
     */
    core::table::DensePoint &get_center();

    const core::table::DensePoint &get_center() const;

    /** @brief Gets the set of far-field coefficients.
     *
     *  @return The const reference to the vector containing the
     *          far-field coefficients.
     */
    const core::table::DensePoint& get_coeffs() const;

    /** @brief Gets the approximation order.
     *
     *  @return The integer representing the current approximation order.
     */
    short int get_order() const;

    /** @brief Gets the weight sum.
     */
    double get_weight_sum() const;

    /** @brief Sets the approximation order of the far-field expansion.
     *
     *  @param new_order The desired new order of the approximation.
     */
    void set_order(short int new_order);

    ////////// User-level Functions //////////

    /** @brief Accumulates the far field moment represented by the given
     *         reference data into the coefficients.
     */
    template<typename KernelAuxType>
    void AccumulateCoeffs(
      const KernelAuxType &kernel_aux_in,
      const core::table::DenseMatrix &data,
      const core::table::DensePoint &weights,
      int begin, int end, int order);

    /** @brief Refine the far field moment that has been computed before
     *         up to a new order.
     */
    template<typename KernelAuxType>
    void RefineCoeffs(
      const KernelAuxType &kernel_aux_in,
      const core::table::DenseMatrix &data,
      const core::table::DensePoint &weights,
      int begin, int end, int order);

    /** @brief Evaluates the far-field coefficients at the given point.
     */
    template<typename KernelAuxType>
    double EvaluateField(
      const KernelAuxType &kernel_aux_in,
      const core::table::DensePoint &point) const;

    /** @brief Initializes the current far field expansion object with
     *         the given center.
     */
    template<typename KernelAuxType>
    void Init(const KernelAuxType &ka, const core::table::DensePoint& center);

    template<typename KernelAuxType>
    void Init(const KernelAuxType &ka);

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
      const KernelIndependentFarField<ExpansionType> &se);

    /** @brief Translate to the given local expansion. The translated
     *         coefficients are added up to the passed-in local
     *         expansion coefficients.
     */
    template<typename KernelAuxType, typename KernelIndependentLocalType>
    void TranslateToLocal(
      const KernelAuxType &kernel_aux_in,
      int truncation_order, KernelIndependentLocalType *se) const;
};
}
}

#endif
