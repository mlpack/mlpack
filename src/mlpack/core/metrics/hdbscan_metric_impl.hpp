/**
 * @file hdbscan_metric_impl.hpp
 */
#ifndef MLPACK_CORE_METRICS_HDBSAN_METRIC_IMPL_HPP
#define MLPACK_CORE_METRICS_HDBSCAN_METRIC_IMPL_HPP


namespace mlpack {
namespace metric {

// HDBSCAN specialization (the return type is double)
// d_core for both points is appended at the end
// So now a 3-d vector is ppased as a 4-d vector
// where first 3 entries are its values and last
// entry is its d_core value.
// distSq finds sum of square for all the entries
// in (a-b) vector but this contains the last
// entry too. Remove it using lastEle.
  template<typename VecTypeA, typename VecTypeB>
  typename VecTypeA::elem_type HdbscanMetric::Evaluate(
      const VecTypeA& a,
      const VecTypeB& b)
  {
    typename VecTypeA::elem_type dcoreA = a(a.n_rows-1);
    typename VecTypeB::elem_type dcoreB = b(b.n_rows-1);
    typename VecTypeB::elem_type distSq = arma::accu(arma::square(a - b));
    typename VecTypeB::elem_type lastEle = (dcoreA - dcoreB)*(dcoreA - dcoreB);
    typename VecTypeB::elem_type dist = sqrt(distSq - lastEle);

    return std::max(std::max(dcoreA, dcoreB), dist);
  }

} // namespace metric
} // namespace mlpack

#endif
