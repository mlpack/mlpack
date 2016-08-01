/**
 * @file cellbound.hpp
 *
 */
#ifndef MLPACK_CORE_TREE_CELLBOUND_HPP
#define MLPACK_CORE_TREE_CELLBOUND_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/math/range.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include "bound_traits.hpp"
#include "address.hpp"

namespace mlpack {
namespace bound {

template<typename MetricType = metric::LMetric<2, true>,
         typename ElemType = double>
class CellBound
{
 public:
  typedef typename std::conditional<sizeof(ElemType) * CHAR_BIT <= 32,
                                    uint32_t,
                                    uint64_t>::type AddressElemType;

  /**
   * Empty constructor; creates a bound of dimensionality 0.
   */
  CellBound();

  /**
   * Initializes to specified dimensionality with each dimension the empty
   * set.
   */
  CellBound(const size_t dimension);

  //! Copy constructor; necessary to prevent memory leaks.
  CellBound(const CellBound& other);
  //! Same as copy constructor; necessary to prevent memory leaks.
  CellBound& operator=(const CellBound& other);

  //! Move constructor: take possession of another bound's information.
  CellBound(CellBound&& other);

  //! Destructor: clean up memory.
  ~CellBound();

  /**
   * Resets all dimensions to the empty set (so that this bound contains
   * nothing).
   */
  void Clear();

  //! Gets the dimensionality.
  size_t Dim() const { return dim; }

  //! Get the range for a particular dimension.  No bounds checking.  Be
  //! careful: this may make MinWidth() invalid.
  math::RangeType<ElemType>& operator[](const size_t i) { return bounds[i]; }
  //! Modify the range for a particular dimension.  No bounds checking.
  const math::RangeType<ElemType>& operator[](const size_t i) const
  { return bounds[i]; }

  arma::Col<AddressElemType>& LoAddress() { return loAddress; }

  const arma::Col<AddressElemType>& LoAddress() const {return loAddress; }
  
  arma::Col<AddressElemType>& HiAddress() { return hiAddress; }

  const arma::Col<AddressElemType>& HiAddress() const {return hiAddress; }

  const arma::Mat<ElemType>& LoBound() const { return loBound; }

  const arma::Mat<ElemType>& HiBound() const { return hiBound; }

  size_t NumBounds() const { return numBounds; }

  //! Get the minimum width of the bound.
  ElemType MinWidth() const { return minWidth; }
  //! Modify the minimum width of the bound.
  ElemType& MinWidth() { return minWidth; }

  /**
   * Calculates the center of the range, placing it into the given vector.
   *
   * @param center Vector which the center will be written to.
   */
  void Center(arma::Col<ElemType>& center) const;

  /**
   * Calculates minimum bound-to-point distance.
   *
   * @param point Point to which the minimum distance is requested.
   */
  template<typename VecType>
  ElemType MinDistance(const VecType& point,
                       typename boost::enable_if<IsVector<VecType>>* = 0) const;

  /**
   * Calculates minimum bound-to-bound distance.
   *
   * @param other Bound to which the minimum distance is requested.
   */
  ElemType MinDistance(const CellBound& other) const;

  /**
   * Calculates maximum bound-to-point squared distance.
   *
   * @param point Point to which the maximum distance is requested.
   */
  template<typename VecType>
  ElemType MaxDistance(const VecType& point,
                       typename boost::enable_if<IsVector<VecType>>* = 0) const;

  /**
   * Computes maximum distance.
   *
   * @param other Bound to which the maximum distance is requested.
   */
  ElemType MaxDistance(const CellBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-bound distance.
   *
   * @param other Bound to which the minimum and maximum distances are
   *     requested.
   */
  math::RangeType<ElemType> RangeDistance(const CellBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point distance.
   *
   * @param point Point to which the minimum and maximum distances are
   *     requested.
   */
  template<typename VecType>
  math::RangeType<ElemType> RangeDistance(
      const VecType& point,
      typename boost::enable_if<IsVector<VecType>>* = 0) const;

  /**
   * Expands this region to include new points.
   *
   * @tparam MatType Type of matrix; could be Mat, SpMat, a subview, or just a
   *   vector.
   * @param data Data points to expand this region to include.
   */
  template<typename MatType>
  CellBound& operator|=(const MatType& data);

  /**
   * Expands this region to encompass another bound.
   */
  CellBound& operator|=(const CellBound& other);

  /**
   * Determines if a point is within this bound.
   */
  template<typename VecType>
  bool Contains(const VecType& point) const;

  void UpdateAddressBounds();

  /**
   * Returns the diameter of the hyperrectangle (that is, the longest diagonal).
   */
  ElemType Diameter() const;

  /**
   * Serialize the bound object.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int version);

 private:
  static constexpr size_t order = sizeof(AddressElemType) * CHAR_BIT;
  const size_t maxNumBounds = 10;
  //! The dimensionality of the bound.
  size_t dim;
  //! The bounds for each dimension.
  math::RangeType<ElemType>* bounds;
  arma::Mat<ElemType> loBound;
  arma::Mat<ElemType> hiBound;
  size_t numBounds;

  arma::Col<AddressElemType> loAddress;
  arma::Col<AddressElemType> hiAddress;

  ElemType minWidth;

  void AddBound(const arma::Col<ElemType>& loCorner,
                const arma::Col<ElemType>& hiCorner);
  void InitHighBound(size_t numEqualBits);
  void InitLowerBound(size_t numEqualBits);
};

// A specialization of BoundTraits for this class.
template<typename MetricType, typename ElemType>
struct BoundTraits<CellBound<MetricType, ElemType>>
{
  //! These bounds are always tight for each dimension.
  const static bool HasTightBounds = true;
};

} // namespace bound
} // namespace mlpack

#include "cellbound_impl.hpp"

#endif // MLPACK_CORE_TREE_CELLBOUND_HPP

