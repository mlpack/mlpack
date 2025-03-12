/**
 * @file core/tree/rectangle_tree/discrete_hilbert_value_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the DiscreteHilbertValue class, a class that calculates
 * the ordering of points using the Hilbert curve.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_IMPL_HPP

#include "discrete_hilbert_value.hpp"

namespace mlpack {

template<typename TreeElemType>
DiscreteHilbertValue<TreeElemType>::DiscreteHilbertValue() :
    localHilbertValues(NULL),
    ownsLocalHilbertValues(false),
    numValues(0),
    valueToInsert(NULL),
    ownsValueToInsert(false)
{ }

template<typename TreeElemType>
DiscreteHilbertValue<TreeElemType>::~DiscreteHilbertValue()
{
  if (ownsLocalHilbertValues)
    delete localHilbertValues;
  if (ownsValueToInsert)
    delete valueToInsert;
}

template<typename TreeElemType>
template<typename TreeType>
DiscreteHilbertValue<TreeElemType>::DiscreteHilbertValue(const TreeType* tree) :
    localHilbertValues(NULL),
    ownsLocalHilbertValues(false),
    numValues(0),
    valueToInsert(tree->Parent() ?
        tree->Parent()->AuxiliaryInfo().HilbertValue().ValueToInsert() :
        new arma::Col<HilbertElemType>(tree->Dataset().n_rows)),
    ownsValueToInsert(tree->Parent() ? false : true)
{
  // Calculate the Hilbert value for all points.
  if (!tree->Parent()) // This is the root node.
    ownsLocalHilbertValues = true;
  else if (tree->Parent()->Child(0).IsLeaf())
  {
    // This is a leaf node.
    assert(tree->Parent()->NumChildren() > 0);
    ownsLocalHilbertValues = true;
  }

  if (ownsLocalHilbertValues)
  {
    localHilbertValues = new arma::Mat<HilbertElemType>(tree->Dataset().n_rows,
        tree->MaxLeafSize() + 1);
  }
}

template<typename TreeElemType>
template<typename TreeType>
DiscreteHilbertValue<TreeElemType>::
DiscreteHilbertValue(const DiscreteHilbertValue& other,
                     TreeType* tree,
                     bool deepCopy) :
    localHilbertValues(NULL),
    ownsLocalHilbertValues(other.ownsLocalHilbertValues),
    numValues(other.NumValues()),
    valueToInsert(NULL),
    ownsValueToInsert(other.ownsValueToInsert)
{
  if (deepCopy)
  {
    // Only leaf nodes own the localHilbertValues dataset.
    // Intermediate nodes store the pointer to the corresponding dataset.
    if (ownsLocalHilbertValues)
      localHilbertValues = new arma::Mat<HilbertElemType>(
          *other.LocalHilbertValues());
    else
      localHilbertValues = NULL;

    // Only the root owns ownsValueToInsert. Other nodes the pointer.
    if (ownsValueToInsert)
      valueToInsert = new arma::Col<HilbertElemType>(
          *other.ValueToInsert());
    else
    {
      assert(tree->Parent() != NULL);
      // Copy the pointer from the parent node.
      valueToInsert = const_cast<arma::Col<HilbertElemType>*>
        (tree->Parent()->AuxiliaryInfo().HilbertValue().ValueToInsert());
    }

    if (tree->NumChildren() == 0)
    {
      // We have to update pointers to the localHilbertValues dataset in
      // intermediate nodes.
      TreeType* node = tree;

      while (node->Parent() != NULL)
      {
        if (node->Parent()->NumChildren() > 1)
        {
          const std::vector<TreeType*> parentChildren =
              node->AuxiliaryInfo().Children(node->Parent());
          // If node is not the last child of its parent, we shouldn't copy
          // the localHilbertValues pointer.
          if (parentChildren[node->Parent()->NumChildren() - 2] == NULL)
            break;
        }
        node->Parent()->AuxiliaryInfo().HilbertValue().LocalHilbertValues() =
          localHilbertValues;
        node = node->Parent();
      }
    }
  }
  else
  {
    localHilbertValues = const_cast<arma::Mat<HilbertElemType>*>
        (other.LocalHilbertValues());
    valueToInsert = const_cast<arma::Col<HilbertElemType>*>
        (other.ValueToInsert());
  }
}

template<typename TreeElemType>
DiscreteHilbertValue<TreeElemType>::
DiscreteHilbertValue(DiscreteHilbertValue&& other) :
    localHilbertValues(other.localHilbertValues),
    ownsLocalHilbertValues(other.ownsLocalHilbertValues),
    numValues(other.numValues),
    valueToInsert(other.valueToInsert),
    ownsValueToInsert(other.ownsValueToInsert)
{
  other.localHilbertValues = NULL;
  other.ownsLocalHilbertValues = false;
  other.numValues = 0;
  other.valueToInsert = NULL;
  other.ownsValueToInsert = false;
}

template<typename TreeElemType>
template<typename VecType>
arma::Col<typename DiscreteHilbertValue<TreeElemType>::HilbertElemType>
DiscreteHilbertValue<TreeElemType>::
CalculateValue(const VecType& pt,
               typename std::enable_if_t<IsVector<VecType>::value>*)
{
  using VecElemType = typename VecType::elem_type;
  arma::Col<HilbertElemType> res(pt.n_rows);
  // Calculate the number of bits for the exponent.
  const int numExpBits = std::ceil(std::log2(
      std::numeric_limits<VecElemType>::max_exponent -
      std::numeric_limits<VecElemType>::min_exponent + 1.0));

  // Calculate the number of bits for the mantissa.
  const int numMantBits = order - numExpBits - 1;

  for (size_t i = 0; i < pt.n_rows; ++i)
  {
    int e;
    VecElemType normalizedVal = std::frexp(pt(i), &e);
    bool sgn = std::signbit(normalizedVal);

    if (pt(i) == 0)
      e = std::numeric_limits<VecElemType>::min_exponent;

    if (sgn)
      normalizedVal = -normalizedVal;

    if (e < std::numeric_limits<VecElemType>::min_exponent)
    {
      HilbertElemType tmp = (HilbertElemType) 1 <<
          (std::numeric_limits<VecElemType>::min_exponent - e);

      e = std::numeric_limits<VecElemType>::min_exponent;
      normalizedVal /= tmp;
    }

    // Extract the mantissa.
    HilbertElemType tmp = (HilbertElemType) 1 << numMantBits;
    res(i) = std::floor(normalizedVal * tmp);

    // Add the exponent.
    assert(res(i) < ((HilbertElemType) 1 << numMantBits));
    res(i) |= ((HilbertElemType)
        (e - std::numeric_limits<VecElemType>::min_exponent)) << numMantBits;

    assert(res(i) < ((HilbertElemType) 1 << (order - 1)) - 1);

    // Negative values should be inverted.
    if (sgn)
    {
      res(i) = ((HilbertElemType) 1 << (order - 1)) - 1 - res(i);
      assert((res(i) >> (order - 1)) == 0);
    }
    else
    {
      res(i) |= (HilbertElemType) 1 << (order - 1);
      assert((res(i) >> (order - 1)) == 1);
    }
  }

  HilbertElemType M = (HilbertElemType) 1 << (order - 1);

  // Since the Hilbert curve is continuous we should permutate and intend
  // coordinate axes depending on the position of the point.
  for (HilbertElemType Q = M; Q > 1; Q >>= 1)
  {
    HilbertElemType P = Q - 1;

    for (size_t i = 0; i < pt.n_rows; ++i)
    {
      if (res(i) & Q) // Invert.
        res(0) ^= P;
      else // Permutate.
      {
        HilbertElemType t = (res(0) ^ res(i)) & P;
        res(0) ^= t;
        res(i) ^= t;
      }
    }
  }

  // Gray encode.
  for (size_t i = 1; i < pt.n_rows; ++i)
    res(i) ^= res(i - 1);

  HilbertElemType t = 0;

  // Some coordinate axes should be inverted.
  for (HilbertElemType Q = M; Q > 1; Q >>= 1)
    if (res(pt.n_rows - 1) & Q)
      t ^= Q - 1;

  for (size_t i = 0; i < pt.n_rows; ++i)
    res(i) ^= t;

  // We should rearrange bits in order to compare two Hilbert values faster.
  arma::Col<HilbertElemType> rearrangedResult(pt.n_rows);

  for (size_t i = 0; i < order; ++i)
    for (size_t j = 0; j < pt.n_rows; ++j)
    {
      size_t bit = (i * pt.n_rows + j) % order;
      size_t row = (i * pt.n_rows + j) / order;

      rearrangedResult(row) |= (((res(j) >> (order - 1 - i)) & 1) <<
          (order - 1 - bit));
    }

  return rearrangedResult;
}

template<typename TreeElemType>
int DiscreteHilbertValue<TreeElemType>::
CompareValues(const arma::Col<HilbertElemType>& value1,
              const arma::Col<HilbertElemType>& value2)
{
  for (size_t i = 0; i < value1.n_rows; ++i)
  {
    if (value1(i) > value2(i))
      return 1;
    else if (value1(i) < value2(i))
      return -1;
  }

  return 0;
}



template<typename TreeElemType>
template<typename VecType1, typename VecType2>
int DiscreteHilbertValue<TreeElemType>::
ComparePoints(const VecType1& pt1,
              const VecType2& pt2,
              typename std::enable_if_t<IsVector<VecType1>::value>*,
              typename std::enable_if_t<IsVector<VecType2>::value>*)
{
  arma::Col<HilbertElemType> val1 = CalculateValue(pt1);
  arma::Col<HilbertElemType> val2 = CalculateValue(pt2);

  return CompareValues(val1, val2);
}

template<typename TreeElemType>
int DiscreteHilbertValue<TreeElemType>::
CompareValues(const DiscreteHilbertValue& val1,
              const DiscreteHilbertValue& val2)
{
  if (val1.NumValues() > 0 && val2.NumValues() == 0)
    return 1;
  else if (val1.NumValues() == 0 && val2.NumValues() > 0)
    return -1;
  else if (val1.NumValues() == 0 && val2.NumValues() == 0)
    return 0;

  return CompareValues(val1.LocalHilbertValues()->col(val1.NumValues() - 1),
                       val2.LocalHilbertValues()->col(val2.NumValues() - 1));
}

template<typename TreeElemType>
int DiscreteHilbertValue<TreeElemType>::
CompareWith(const DiscreteHilbertValue& val) const
{
  return CompareValues(*this, val);
}

template<typename TreeElemType>
template<typename VecType>
int DiscreteHilbertValue<TreeElemType>::
CompareWith(const VecType& pt,
            typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  arma::Col<HilbertElemType> val = CalculateValue(pt);

  if (numValues == 0)
    return -1;

  return CompareValues(localHilbertValues->col(numValues - 1), val);
}

template<typename TreeElemType>
template<typename VecType>
int DiscreteHilbertValue<TreeElemType>::
CompareWithCachedPoint(const VecType& ,
            typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  if (numValues == 0)
    return -1;

  return CompareValues(localHilbertValues->col(numValues - 1), *valueToInsert);
}

template<typename TreeElemType>
template<typename TreeType, typename VecType>
size_t DiscreteHilbertValue<TreeElemType>::
InsertPoint(TreeType *node,
            const VecType& pt,
            typename std::enable_if_t<IsVector<VecType>::value>*)
{
  size_t i = 0;

  // All points are inserted to the root node.
  if (!node->Parent())
    *valueToInsert = CalculateValue(pt);
  if (node->IsLeaf())
  {
    // Find an appropriate place.
    for (i = 0; i < numValues; ++i)
      if (CompareValues(localHilbertValues->col(i), *valueToInsert) > 0)
        break;

    for (size_t j = numValues; j > i; j--)
      localHilbertValues->col(j) = localHilbertValues->col(j-1);

    localHilbertValues->col(i) = *valueToInsert;
    numValues++;
    // Propagate changes of the largest Hilbert value downward.
    TreeType* root = node->Parent();

    while (root != NULL)
    {
      root->AuxiliaryInfo().HilbertValue().UpdateLargestValue(root);

      root = root->Parent();
    }
  }

  return i;
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::InsertNode(TreeType* node)
{
  DiscreteHilbertValue &val = node->AuxiliaryInfo().HilbertValue();

  if (node->AuxiliaryInfo().HilbertValue().CompareWith(val) < 0)
  {
    localHilbertValues = val.LocalHilbertValues();
    numValues = val.NumValues();
  }
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::
DeletePoint(TreeType* /* node */, const size_t localIndex)
{
  // Delete the Hilbert value from the local dataset
  for (size_t i = numValues - 1; i > localIndex; i--)
    localHilbertValues->col(i - 1) = localHilbertValues->col(i);

  numValues--;
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::
RemoveNode(TreeType* node, const size_t nodeIndex)
{
  if (node->NumChildren() <= 1)
  {
    localHilbertValues = NULL;
    numValues = 0;
    return;
  }
  if (nodeIndex + 1 == node->NumChildren())
  {
    // Update the largest Hilbert value if the value exists
    TreeType& child = node->Child(nodeIndex - 1);
    if (child.AuxiliaryInfo().HilbertValue().NumValues() != 0)
    {
      numValues = child.AuxiliaryInfo().HilbertValue().NumValues();
      localHilbertValues =
          child.AuxiliaryInfo().HilbertValue().LocalHilbertValues();
    }
    else
    {
      localHilbertValues = NULL;
      numValues = 0;
    }
  }
}

template<typename TreeElemType>
DiscreteHilbertValue<TreeElemType>& DiscreteHilbertValue<TreeElemType>::
operator=(const DiscreteHilbertValue& other)
{
  if (this == &other)
    return *this;

  if (ownsLocalHilbertValues)
    delete localHilbertValues;

  localHilbertValues = const_cast<arma::Mat<HilbertElemType>* >
      (other.LocalHilbertValues());
  ownsLocalHilbertValues = false;
  numValues = other.NumValues();

  return *this;
}

template<typename TreeElemType>
DiscreteHilbertValue<TreeElemType>& DiscreteHilbertValue<TreeElemType>::
operator=(DiscreteHilbertValue&& other)
{
  if (this != &other)
  {
    localHilbertValues = other.localHilbertValues;
    ownsLocalHilbertValues = other.ownsLocalHilbertValues;
    numValues = other.numValues;
    valueToInsert = other.valueToInsert;
    ownsValueToInsert = other.ownsValueToInsert;

    other.localHilbertValues = nullptr;
    other.ownsLocalHilbertValues = false;
    other.numValues = 0;
    other.valueToInsert = nullptr;
    other.ownsValueToInsert = false;
  }
  return *this;
}

template<typename TreeElemType>
void DiscreteHilbertValue<TreeElemType>::NullifyData()
{
  ownsLocalHilbertValues = false;
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::UpdateLargestValue(TreeType* node)
{
  if (!node->IsLeaf())
  {
    // Update the largest Hilbert value
    localHilbertValues = node->Child(node->NumChildren() -
        1).AuxiliaryInfo().HilbertValue().LocalHilbertValues();
    numValues = node->Child(node->NumChildren() -
        1).AuxiliaryInfo().HilbertValue().NumValues();
  }
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::RedistributeHilbertValues(
    TreeType* parent,
    const size_t firstSibling,
    const size_t lastSibling)
{
  // We need to update the local dataset if points were redistributed.
  size_t numPoints = 0;
  for (size_t i = firstSibling; i <= lastSibling; ++i)
    numPoints += parent->Child(i).NumPoints();

  // Copy the local Hilbert values.
  arma::Mat<HilbertElemType> tmp(localHilbertValues->n_rows, numPoints);

  size_t iPoint = 0;
  for (size_t i = firstSibling; i<= lastSibling; ++i)
  {
    DiscreteHilbertValue<TreeElemType> &value =
        parent->Child(i).AuxiliaryInfo().HilbertValue();

    for (size_t j = 0; j < value.NumValues(); ++j)
    {
      tmp.col(iPoint) = value.LocalHilbertValues()->col(j);
      iPoint++;
    }
  }
  assert(iPoint == numPoints);

  iPoint = 0;

  // Redistribute the Hilbert values.
  for (size_t i = firstSibling; i <= lastSibling; ++i)
  {
    DiscreteHilbertValue<TreeElemType> &value =
        parent->Child(i).AuxiliaryInfo().HilbertValue();

    for (size_t j = 0; j < parent->Child(i).NumPoints(); ++j)
    {
      value.LocalHilbertValues()->col(j) = tmp.col(iPoint);
      iPoint++;
    }
    value.NumValues() = parent->Child(i).NumPoints();
  }

  assert(iPoint == numPoints);
}

template<typename TreeElemType>
template<typename Archive>
void DiscreteHilbertValue<TreeElemType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_POINTER(localHilbertValues));
  ar(CEREAL_NVP(ownsLocalHilbertValues));
  ar(CEREAL_NVP(numValues));
  ar(CEREAL_POINTER(valueToInsert));
  ar(CEREAL_NVP(ownsValueToInsert));
}

} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_IMPL_HPP
