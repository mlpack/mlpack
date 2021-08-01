
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_discrete_hilbert_value.hpp:

Program Listing for File discrete_hilbert_value.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_discrete_hilbert_value.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/discrete_hilbert_value.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree  {
   
   template<typename TreeElemType>
   class DiscreteHilbertValue
   {
    public:
     typedef typename std::conditional<sizeof(TreeElemType) * CHAR_BIT <= 32,
                                       uint32_t,
                                       uint64_t>::type HilbertElemType;
   
     DiscreteHilbertValue();
   
     template<typename TreeType>
     DiscreteHilbertValue(const TreeType* tree);
   
     template<typename TreeType>
     DiscreteHilbertValue(const DiscreteHilbertValue& other,
                          TreeType* tree,
                          bool deepCopy);
   
     DiscreteHilbertValue(DiscreteHilbertValue&& other);
   
     ~DiscreteHilbertValue();
   
     template<typename VecType1, typename VecType2>
     static int ComparePoints(
         const VecType1& pt1,
         const VecType2& pt2,
         typename std::enable_if_t<IsVector<VecType1>::value>* = 0,
         typename std::enable_if_t<IsVector<VecType2>::value>* = 0);
   
     static int CompareValues(const DiscreteHilbertValue& val1,
                              const DiscreteHilbertValue& val2);
   
     int CompareWith(const DiscreteHilbertValue& val) const;
   
     template<typename VecType>
     int CompareWith(
         const VecType& pt,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
   
     template<typename VecType>
     int CompareWithCachedPoint(
         const VecType& pt,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
   
     template<typename TreeType, typename VecType>
     size_t InsertPoint(TreeType *node,
                        const VecType& pt,
                        typename std::enable_if_t<IsVector<VecType>::value>* = 0);
   
     template<typename TreeType>
     void InsertNode(TreeType* node);
   
     template<typename TreeType>
     void DeletePoint(TreeType* node, const size_t localIndex);
   
     template<typename TreeType>
     void RemoveNode(TreeType* node, const size_t nodeIndex);
   
     DiscreteHilbertValue& operator=(const DiscreteHilbertValue& other);
   
     DiscreteHilbertValue& operator=(DiscreteHilbertValue&& other);
   
     void NullifyData();
   
     template<typename TreeType>
     void UpdateLargestValue(TreeType* node);
   
     template<typename TreeType>
     void RedistributeHilbertValues(TreeType* parent,
                                    const size_t firstSibling,
                                    const size_t lastSibling);
   
     template<typename VecType>
     static arma::Col<HilbertElemType> CalculateValue(
         const VecType& pt,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0);
   
     static int CompareValues(const arma::Col<HilbertElemType>& value1,
                              const arma::Col<HilbertElemType>& value2);
   
     size_t NumValues() const { return numValues; }
     size_t& NumValues() { return numValues; }
   
     const arma::Mat<HilbertElemType>* LocalHilbertValues() const
     { return localHilbertValues; }
     arma::Mat<HilbertElemType>*& LocalHilbertValues()
     { return localHilbertValues; }
   
     bool OwnsLocalHilbertValues() const { return ownsLocalHilbertValues; }
     bool& OwnsLocalHilbertValues() { return ownsLocalHilbertValues; }
   
     const arma::Col<HilbertElemType>* ValueToInsert() const
     { return valueToInsert; }
     arma::Col<HilbertElemType>* ValueToInsert() { return valueToInsert; }
   
     bool OwnsValueToInsert() const { return ownsValueToInsert; }
     bool& OwnsValueToInsert() { return ownsValueToInsert; }
    private:
     static constexpr size_t order = sizeof(HilbertElemType) * CHAR_BIT;
     arma::Mat<HilbertElemType>* localHilbertValues;
     bool ownsLocalHilbertValues;
     size_t numValues;
     arma::Col<HilbertElemType>* valueToInsert;
     bool ownsValueToInsert;
   
    public:
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "discrete_hilbert_value_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
