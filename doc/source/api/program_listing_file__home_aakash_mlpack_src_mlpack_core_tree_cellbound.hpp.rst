
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_cellbound.hpp:

Program Listing for File cellbound.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_cellbound.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/cellbound.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author = {Bayer, Rudolf},
     title = {The Universal B-Tree for Multidimensional Indexing: General
         Concepts},
     booktitle = {Proceedings of the International Conference on Worldwide
         Computing and Its Applications},
     series = {WWCA '97},
     year = {1997},
     isbn = {3-540-63343-X},
     pages = {198--209},
     numpages = {12},
     publisher = {Springer-Verlag},
     address = {London, UK, UK},
   }
   
   #ifndef MLPACK_CORE_TREE_CELLBOUND_HPP
   #define MLPACK_CORE_TREE_CELLBOUND_HPP
   
   #include <mlpack/prereqs.hpp>
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
   
     CellBound();
   
     CellBound(const size_t dimension);
   
     CellBound(const CellBound& other);
     CellBound& operator=(const CellBound& other);
   
     CellBound(CellBound&& other);
   
     ~CellBound();
   
     void Clear();
   
     size_t Dim() const { return dim; }
   
     math::RangeType<ElemType>& operator[](const size_t i) { return bounds[i]; }
     const math::RangeType<ElemType>& operator[](const size_t i) const
     { return bounds[i]; }
   
     arma::Col<AddressElemType>& LoAddress() { return loAddress; }
     const arma::Col<AddressElemType>& LoAddress() const {return loAddress; }
   
     arma::Col<AddressElemType>& HiAddress() { return hiAddress; }
     const arma::Col<AddressElemType>& HiAddress() const {return hiAddress; }
   
     const arma::Mat<ElemType>& LoBound() const { return loBound; }
     const arma::Mat<ElemType>& HiBound() const { return hiBound; }
   
     size_t NumBounds() const { return numBounds; }
   
     ElemType MinWidth() const { return minWidth; }
     ElemType& MinWidth() { return minWidth; }
   
     const MetricType& Metric() const { return metric; }
     MetricType& Metric() { return metric; }
   
     void Center(arma::Col<ElemType>& center) const;
   
     template<typename VecType>
     ElemType MinDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const;
   
     ElemType MinDistance(const CellBound& other) const;
   
     template<typename VecType>
     ElemType MaxDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const;
   
     ElemType MaxDistance(const CellBound& other) const;
   
     math::RangeType<ElemType> RangeDistance(const CellBound& other) const;
   
     template<typename VecType>
     math::RangeType<ElemType> RangeDistance(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
   
     template<typename MatType>
     CellBound& operator|=(const MatType& data);
   
     CellBound& operator|=(const CellBound& other);
   
     template<typename VecType>
     bool Contains(const VecType& point) const;
   
     template<typename MatType>
     void UpdateAddressBounds(const MatType& data);
   
     ElemType Diameter() const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     static constexpr size_t order = sizeof(AddressElemType) * CHAR_BIT;
     const size_t maxNumBounds = 10;
     size_t dim;
     math::RangeType<ElemType>* bounds;
     arma::Mat<ElemType> loBound;
     arma::Mat<ElemType> hiBound;
     size_t numBounds;
     arma::Col<AddressElemType> loAddress;
     arma::Col<AddressElemType> hiAddress;
     ElemType minWidth;
     MetricType metric;
   
     template<typename MatType>
     void AddBound(const arma::Col<ElemType>& loCorner,
                   const arma::Col<ElemType>& hiCorner,
                   const MatType& data);
     template<typename MatType>
     void InitHighBound(size_t numEqualBits, const MatType& data);
   
     template<typename MatType>
     void InitLowerBound(size_t numEqualBits, const MatType& data);
   };
   
   // A specialization of BoundTraits for this class.
   template<typename MetricType, typename ElemType>
   struct BoundTraits<CellBound<MetricType, ElemType>>
   {
     const static bool HasTightBounds = true;
   };
   
   } // namespace bound
   } // namespace mlpack
   
   #include "cellbound_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_CELLBOUND_HPP
   
