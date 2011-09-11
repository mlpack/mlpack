/** @file cartesian_local.h
 *
 *  This file contains a templatized class implementing $O(D^p)$ or
 *  $O(p^D)$ expansion for computing the coefficients for a local
 *  expansion for an arbitrary kernel function.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_CARTESIAN_LOCAL_H
#define MLPACK_SERIES_EXPANSION_CARTESIAN_LOCAL_H

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/tracking_enum.hpp>
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"
#include "mlpack/series_expansion/cartesian_expansion_global_dev.h"

namespace mlpack {
namespace series_expansion {

/** @brief The general Cartesian local expansion.
 */
template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
class CartesianLocal {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

    /** @brief The center of the expansion. */
    arma::vec center_;

    /** @brief The coefficients. */
    arma::vec coeffs_;

    /** @brief The truncation order. */
    int order_;

  public:

    /** @brief The default constructor.
     */
    CartesianLocal() {
      order_ = -1;
    }

    /** @brief Serializes the far field object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & center_;
      ar & coeffs_;
      ar & order_;
    }

    /** @brief Get the center of expansion. */
    arma::vec &get_center();

    /** @brief Get the center of expansion. */
    const arma::vec &get_center() const;

    /** @brief Get the coefficients. */
    const arma::vec& get_coeffs() const;

    /** @brief Get the approximation order. */
    int get_order() const;

    /** @brief Set the approximation order. */
    void set_order(int new_order);

    /** @brief Accumulates the local moment represented by the given
     *         reference data into the coefficients.
     */
    template<typename KernelAuxType, typename TreeIteratorType>
    void AccumulateCoeffs(
      const KernelAuxType &kernel_aux_in,
      TreeIteratorType &it, int order);

    /** @brief Evaluates the local coefficients at the given point.
     */
    template<typename KernelAuxType>
    double EvaluateField(
      const KernelAuxType &kernel_aux_in,
      const arma::vec &x_q) const;

    /** @brief Initializes the current local expansion object with the
     *         given center.
     */
    template<typename KernelAuxType>
    void Init(
      const KernelAuxType &kernel_aux_in,
      const arma::vec& center);

    /** @brief Initializes the local expansion with the given kernel
     *         and the given number of maximum coefficients.
     */
    void Init(const arma::vec& center, int max_num_coeffs_in);

    /** @brief Initializes the local expansion with the given kernel.
     */
    template<typename KernelAuxType>
    void Init(const KernelAuxType &ka);

    /** @brief Prints out the series expansion represented by this
     *         object.
     */
    template<typename KernelAuxType>
    void Print(
      const KernelAuxType &kernel_aux_in,
      const char *name = "", FILE *stream = stderr) const;

    /** @brief Sets the coefficients to zero.
     */
    void SetZero();

    /** @brief Translate from a far field expansion to the expansion
     *         here.The translated coefficients are added up to the
     *         ones here.
     */
    template<typename KernelAuxType, typename CartesianFarFieldType>
    void TranslateFromFarField(
      const KernelAuxType &kernel_aux_in, const CartesianFarFieldType &se);

    /** @brief Translate to the given local expansion. The translated
     *         coefficients are added up to the passed-in local
     *         expansion coefficients.
     */
    template<typename KernelAuxType>
    void TranslateToLocal(
      const KernelAuxType &kernel_aux_in,
      CartesianLocal<ExpansionType> *se) const;
};

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianLocal<ExpansionType>::SetZero() {
  order_ = -1;
  coeffs_.zeros();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
arma::vec &CartesianLocal<ExpansionType>::get_center() {
  return center_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const arma::vec &CartesianLocal <
ExpansionType >::get_center() const {
  return center_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const arma::vec& CartesianLocal <
ExpansionType >::get_coeffs() const {
  return coeffs_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
int CartesianLocal<ExpansionType>::get_order() const {
  return order_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianLocal<ExpansionType>::set_order(int new_order) {
  order_ = new_order;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianLocal<ExpansionType>::Init(
  const arma::vec& center, int max_num_coeffs_in) {

  // Copy the center.
  center_ = center;
  order_ = -1;

  // Initialize coefficient array.
  coeffs_.set_size(max_num_coeffs_in);
  coeffs_.zeros();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianLocal<ExpansionType>::Init(
  const KernelAuxType &kernel_aux_in, const arma::vec& center) {

  // Copy the center.
  center_ = center;
  order_ = -1;

  // Initialize coefficient array.
  coeffs_.set_size(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.zeros();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianLocal<ExpansionType>::Init(const KernelAuxType &kernel_aux_in) {

  // Initialize the center to be zero.
  order_ = -1;
  center_.set_size(kernel_aux_in.global().get_dimension());
  center_.zeros();

  // Initialize coefficient array.
  coeffs_.set_size(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.zeros();
}
}
}

namespace boost {
namespace serialization {
template<>
template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
struct tracking_level <
    mlpack::series_expansion::CartesianLocal<ExpansionType> > {
  typedef mpl::integral_c_tag tag;
  typedef mpl::int_< boost::serialization::track_never > type;
  BOOST_STATIC_CONSTANT(
    int,
    value = tracking_level::type::value
  );
  BOOST_STATIC_ASSERT((
                        mpl::greater <
                        implementation_level <
                        mlpack::series_expansion::CartesianLocal<ExpansionType> > ,
                        mpl::int_<primitive_type>
                        >::value
                      ));
};
}
}

#endif
