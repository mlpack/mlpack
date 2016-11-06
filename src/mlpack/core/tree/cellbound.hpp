/**
 * @file cellbound.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the CellBound class. The class describes a bound that consists
 * of a number of hyperrectangles. These hyperrectangles do not overlap each
 * other. The bound is limited by an outer hyperrectangle and two addresses,
 * the lower address and the high address. Thus, the bound contains all points
 * included between the lower and the high addresses.
 *
 * The notion of addresses is described in the following paper.
 * @code
 * @inproceedings{bayer1997,
 *   author = {Bayer, Rudolf},
 *   title = {The Universal B-Tree for Multidimensional Indexing: General
 *       Concepts},
 *   booktitle = {Proceedings of the International Conference on Worldwide
 *       Computing and Its Applications},
 *   series = {WWCA '97},
 *   year = {1997},
 *   isbn = {3-540-63343-X},
 *   pages = {198--209},
 *   numpages = {12},
 *   publisher = {Springer-Verlag},
 *   address = {London, UK, UK},
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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

/**
 * The CellBound class describes a bound that consists of a number of
 * hyperrectangles. These hyperrectangles do not overlap each other. The bound
 * is limited by an outer hyperrectangle and two addresses, the lower address
 * and the high address. Thus, the bound contains all points included between
 * the lower and the high addresses. The class caches the minimum bounding
 * rectangle, the lower and the high addresses and the hyperrectangles
 * that are described by the addresses.
 *
 * The notion of addresses is described in the following paper.
 * @code
 * @inproceedings{bayer1997,
 *   author = {Bayer, Rudolf},
 *   title = {The Universal B-Tree for Multidimensional Indexing: General
 *       Concepts},
 *   booktitle = {Proceedings of the International Conference on Worldwide
 *       Computing and Its Applications},
 *   series = {WWCA '97},
 *   year = {1997},
 *   isbn = {3-540-63343-X},
 *   pages = {198--209},
 *   numpages = {12},
 *   publisher = {Springer-Verlag},
 *   address = {London, UK, UK},
 * }
 * @endcode
 */
template<typename MetricType = metric::LMetric<2, true>,
         typename ElemType = double>
class CellBound
{
 public:
  //! Depending on the precision of the tree element type, we may need to use
  //! uint32_t or uint64_t.
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

  //! Get lower address.
  arma::Col<AddressElemType>& LoAddress() { return loAddress; }
  //! Modify lower address.
  const arma::Col<AddressElemType>& LoAddress() const {return loAddress; }
  
  //! Get high address.
  arma::Col<AddressElemType>& HiAddress() { return hiAddress; }
  //! Modify high address.
  const arma::Col<AddressElemType>& HiAddress() const {return hiAddress; }

  //! Get lower bound of each subrectangle.
  const arma::Mat<ElemType>& LoBound() const { return loBound; }
  //! Get high bound of each subrectangle.
  const arma::Mat<ElemType>& HiBound() const { return hiBound; }

  //! Get the number of subrectangles.
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

  /**
   * Calculate the bounds of all subrectangles. You should set the lower and the
   * high addresses.
   *
   * @param data Points that are contained in the node.
   */
  template<typename MatType>
  void UpdateAddressBounds(const MatType& data);

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
  //! The precision of the tree element type.
  static constexpr size_t order = sizeof(AddressElemType) * CHAR_BIT;
  //! Maximum number of subrectangles.
  const size_t maxNumBounds = 10;
  //! The dimensionality of the bound.
  size_t dim;
  //! The bounds for each dimension.
  math::RangeType<ElemType>* bounds;
  //! Lower bounds of subrectangles.
  arma::Mat<ElemType> loBound;
  //! High bounds of subrectangles.
  arma::Mat<ElemType> hiBound;
  //! The numbre of subrectangles.
  size_t numBounds;
  //! The lowest address that the bound may contain.
  arma::Col<AddressElemType> loAddress;
  //! The highest address that the bound may contain.
  arma::Col<AddressElemType> hiAddress;
  //! The minimal width of the outer rectangle.
  ElemType minWidth;

  /**
   * Add a subrectangle to the bound.
   *
   * @param loCorner The lower corner of the subrectangle that is being added.
   * @param hiCorner The high corner of the subrectangle that is being added.
   * @param data Points that are contained in the node.
   */
  template<typename MatType>
  void AddBound(const arma::Col<ElemType>& loCorner,
                const arma::Col<ElemType>& hiCorner,
                const MatType& data);
  /**
   * Initialize all subrectangles that touches the lower address. This function
   * should be called before InitLowerBound().
   *
   * @param numEqualBits The number of equal leading bits of the lower address
   * and the high address.
   * @param data Points that are contained in the node.
   */
  template<typename MatType>
  void InitHighBound(size_t numEqualBits, const MatType& data);

  /**
   * Initialize all subrectangles that touches the high address. This function
   * should be called after InitHighBound().
   *
   * @param numEqualBits The number of equal leading bits of the lower address
   * and the high address.
   * @param data Points that are contained in the node.
   */
  template<typename MatType>
  void InitLowerBound(size_t numEqualBits, const MatType& data);
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

