/**
 * @file discrete_hilbert_value.hpp
 * @author Mikhail Lozhnikov
 *
 * Defintion of the DiscreteHilbertValue class, a class that calculates
 * the ordering of points using the Hilbert curve.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_IMPL_HPP

#include "discrete_hilbert_value.hpp"

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

template<typename TreeElemType>
DiscreteHilbertValue<TreeElemType>::DiscreteHilbertValue() :
    localDataset(NULL),
    ownsLocalDataset(false),
    numValues(0),
    valueToInsert(NULL),
    ownsValueToInsert(false)
{

}

template<typename TreeElemType>
DiscreteHilbertValue<TreeElemType>::~DiscreteHilbertValue()
{
  if(ownsLocalDataset)
    delete localDataset;
  if(ownsValueToInsert)
    delete valueToInsert;
}

template<typename TreeElemType>
template<typename TreeType>
DiscreteHilbertValue<TreeElemType>::DiscreteHilbertValue(const TreeType* tree) :
    localDataset(NULL),
    ownsLocalDataset(false),
    numValues(0),
    valueToInsert(tree->Parent() ? 
                tree->Parent()->AuxiliaryInfo().HilbertValue().ValueToInsert() :
                new arma::Col<HilbertElemType>(tree->LocalDataset().n_rows)),
    ownsValueToInsert(tree->Parent() ? false : true)
{
  // Calculate the Hilbert value for all points
  if(!tree->Parent()) //  This is the root node
    ownsLocalDataset = true;
  else if(tree->Parent()->Children()[0]->IsLeaf())
  {
    // This is a leaf node
    assert(tree->Parent()->NumChildren() > 0);
    ownsLocalDataset = true;
  }
    
  if(ownsLocalDataset)
  {
    localDataset =  new arma::Mat<HilbertElemType>(tree->LocalDataset().n_rows,
                                            tree->MaxLeafSize() + 1);
  }

}

template<typename TreeElemType>
DiscreteHilbertValue<TreeElemType>::
DiscreteHilbertValue(const DiscreteHilbertValue& other) :
    localDataset(const_cast<arma::Mat<HilbertElemType>*>(other.LocalDataset())),
    ownsLocalDataset(other.ownsLocalDataset),
    numValues(other.NumValues()),
    valueToInsert(const_cast<arma::Col<HilbertElemType>*>(other.ValueToInsert())),
    ownsValueToInsert(false)
{

}

template<typename TreeElemType>
template<typename VecType>
arma::Col<typename DiscreteHilbertValue<TreeElemType>::HilbertElemType>
DiscreteHilbertValue<TreeElemType>::
CalculateValue(const VecType& pt,typename boost::enable_if<IsVector<VecType>>*)
{
  typedef typename VecType::elem_type VecElemType;
  arma::Col<HilbertElemType> res(pt.n_rows);
  // The number of bits for the exponent
  const int numExpBits =
             std::ceil(std::log2(std::numeric_limits<VecElemType>::max_exponent -
                      std::numeric_limits<VecElemType>::min_exponent + 1.0));
  // The number of bits for the mantissa
  const int numMantBits = order - numExpBits - 1;

  for(size_t i = 0; i < pt.n_rows; i++)
  {
    int e;
    VecElemType normalizedVal = std::frexp(pt(i),&e);
    bool sgn = std::signbit(normalizedVal);

    if(sgn)
      normalizedVal = -normalizedVal;

    if(e < std::numeric_limits<VecElemType>::min_exponent)
    {
      HilbertElemType tmp = 1 << (std::numeric_limits<VecElemType>::min_exponent - e);
      e = std::numeric_limits<VecElemType>::min_exponent;
      normalizedVal /= tmp;
    }
    //  Extract the mantissa
    HilbertElemType tmp = (HilbertElemType)1 << numMantBits;
    res(i) = std::floor(normalizedVal / tmp);
    //  Add the exponent
    res(i) |= ((HilbertElemType)(e - std::numeric_limits<VecElemType>::min_exponent)) << numMantBits;

    // Negative values should be inverted
    if(sgn)
      res(i) = ((HilbertElemType)1 << (order - 1)) - 1 - res(i);
    else
      res(i) |= (HilbertElemType)1 << (order - 1);
  }

  HilbertElemType M = (HilbertElemType)1 << (order - 1);

  // Since the Hilbert curve is continuous we should permutate and intend
  // coordinate axes depending on the position of the point
  for(HilbertElemType Q = M; Q > 1; Q >>= 1)
  {
    HilbertElemType P = Q - 1;

    for(size_t i = 0; i < pt.n_rows; i++)
    {
      if(res(i) & Q)  // Invert
        res(0) ^= P;
      else            // Permutate
      {
        HilbertElemType t = (res(0) ^ res(i)) & P;
        res(0) ^= t;
        res(i) ^= t;
      }
    }
  }

  // Gray encode
  for(size_t i = 1; i < pt.n_rows; i++)
    res(i) ^= res(i-1);

  HilbertElemType t = 0;

  // Some coordinate axes should be inverted
  for(HilbertElemType Q = M; Q > 1; Q >>= 1)
    if( res(pt.n_rows - 1) & Q)
      t ^= Q - 1;

  for(size_t i = 0; i < pt.n_rows; i++)
    res(i) ^= t;

  // We should rearrange bits in order to compare two Hilbert values faster
  arma::Col<HilbertElemType> rearrangedResult(pt.n_rows,arma::fill::zeros);

  for(size_t i = 0; i < order; i++)
    for(size_t j = 0; j < pt.n_rows; j++)
    {
      size_t bit = (i * pt.n_rows + j) % order;
      size_t row = (i * pt.n_rows + j) / order;

      rearrangedResult(row) |= (res(j) & (1 << i)) >> (i - bit);
    }
      
  return rearrangedResult;
}

template<typename TreeElemType>
int DiscreteHilbertValue<TreeElemType>::
CompareValues(const arma::Col<HilbertElemType>& value1,
              const arma::Col<HilbertElemType>& value2)
{
  for(size_t i = 0;i < value1.n_rows; i++)
  {
    if(value1(i) > value2(i))
      return 1;
    else if(value1(i) < value2(i))
      return -1;
  }

  return 0;
}



template<typename TreeElemType>
template<typename VecType1, typename VecType2>
int DiscreteHilbertValue<TreeElemType>::
ComparePoints(const VecType1& pt1, const VecType2& pt2, 
              typename boost::enable_if<IsVector<VecType1>>*,
              typename boost::enable_if<IsVector<VecType2>>*)
{
  arma::Col<HilbertElemType> val1 = CalculateValue(pt1);
  arma::Col<HilbertElemType> val2 = CalculateValue(pt2);

  return CompareValues(val1,val2);
}

template<typename TreeElemType>
int DiscreteHilbertValue<TreeElemType>::
CompareValues(const DiscreteHilbertValue& val1,
              const DiscreteHilbertValue& val2)
{
  if(val1.HasValue() && !val2.HasValue())
    return 1;
  else if(!val1.HasValue() && val2.HasValue())
    return -1;
  else if(!val1.HasValue() && !val2.HasValue())
    return 0;

  return CompareValues(val1.LocalDataset()->col(val1.NumValues() - 1),
                       val2.LocalDataset()->col(val2.NumValues() - 1));
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
            typename boost::enable_if<IsVector<VecType>>*) const
{
  arma::Col<HilbertElemType> val = CalculateValue(pt);

  if(!HasValue())
    return -1;

  return CompareValues(localDataset->col(numValues - 1),val);
}

template<typename TreeElemType>
template<typename VecType>
int DiscreteHilbertValue<TreeElemType>::
CompareWithCachedPoint(const VecType& ,
            typename boost::enable_if<IsVector<VecType>>*) const
{
  if(!HasValue())
    return -1;

  return CompareValues(localDataset->col(numValues - 1),*valueToInsert);
}

template<typename TreeElemType>
template<typename TreeType, typename VecType>
size_t DiscreteHilbertValue<TreeElemType>::
InsertPoint(TreeType *node, const VecType& pt,
                            typename boost::enable_if<IsVector<VecType>>*)
{
  size_t i = 0;

  // All point are inserted to the root node
  if(!node->Parent())
    *valueToInsert = CalculateValue(pt);
  if(node->IsLeaf())
  {
    // Find an appropriate place
    for(i = 0; i < numValues; i++)
      if(CompareValues(localDataset->col(i), *valueToInsert) > 0)
        break;

    for(size_t j = numValues; j > i; j--)
      localDataset->col(j) = localDataset->col(j-1);

    localDataset->col(i) = *valueToInsert;
    numValues++;
    // Propogate changes of the largest Hilbert value downward
    TreeType* root = node->Parent();

    while(root != NULL)
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
  
  if(CompareWith(node,val) < 0)
  {
    localDataset = val.LocalDataset();
    numValues = val.NumValues();
  }
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::
DeletePoint(TreeType* node, const size_t localIndex)
{

  // Delete the Hilbert value from the local dataset
  for(size_t i = numValues - 1; i > localIndex; i--)
    localDataset->col(i-1) = localDataset->col(i);

  numValues--;
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::
RemoveNode(TreeType* node, const size_t nodeIndex)
{
  if(node->NumChildren() <= 1)
  {
    localDataset = NULL;
    numValues = 0;
    return;
  }
  if(nodeIndex + 1 == node->NumChildren())
  {
    // Update the largest Hilbert value if the value exists
    TreeType* child = node->Children()[nodeIndex-1];
    if(child->AuxiliaryInfo.HilbertValue().NumValues() != 0)
    {
      numValues = child->AuxiliaryInfo.HilbertValue().NumValues();
      localDataset = child->AuxiliaryInfo.HilbertValue().LocalDataset();
    }
    else
    {
      localDataset = NULL;
      numValues = 0;
    }
  }
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::Copy(TreeType* dst, TreeType* src)
{
  DiscreteHilbertValue<TreeElemType> &dstVal = dst->AuxiliaryInfo().HilbertValue();
  DiscreteHilbertValue<TreeElemType> &srcVal = src->AuxiliaryInfo().HilbertValue();

  dst.LocalDataset() = src.LocalDataset();
  dst.NumValues() = src.NumValues();
}

template<typename TreeElemType>
void DiscreteHilbertValue<TreeElemType>::NullifyData()
{
  ownsLocalDataset = false;
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::UpdateLargestValue(TreeType* node)
{
  if(!node->IsLeaf())
  {
    // Update the largest Hilbert value
    localDataset = node->Children()[node->NumChildren()-1]->AuxiliaryInfo().HilbertValue().LocalDataset();
    numValues = node->Children()[node->NumChildren()-1]->AuxiliaryInfo().HilbertValue().NumValues();
  }
}

template<typename TreeElemType>
template<typename TreeType>
void DiscreteHilbertValue<TreeElemType>::
UpdateHilbertValues(TreeType* parent, size_t firstSibling, size_t lastSibling)
{
  //  We should update the local dataset if points were redistributed
  
  size_t numPoints = 0;

  for(size_t i = firstSibling; i<= lastSibling; i++)
    numPoints += parent->Children()[i]->NumPoints();

  // Copy the local datasets
  arma::Mat<HilbertElemType> tmp(localDataset->n_rows,numPoints);

  size_t iPoint = 0;
  for(size_t i = firstSibling; i<= lastSibling; i++)
  {
    DiscreteHilbertValue<TreeElemType> &value =
                          parent->Children()[i]->AuxiliaryInfo().HilbertValue();
    
    for(size_t j = 0; j < value.NumValues(); j++)
    {
      tmp.col(iPoint) = value.LocalDataset()->col(j);
      iPoint++;
    }
  }
  assert(iPoint == numPoints);

  iPoint = 0;

  //  Redistribute the Hilbert values
  for(size_t i = firstSibling; i<= lastSibling; i++)
  {
    DiscreteHilbertValue<TreeElemType> &value =
                          parent->Children()[i]->AuxiliaryInfo().HilbertValue();
    
    for(size_t j = 0; j < parent->Children()[i]->NumPoints(); j++)
    {
      value.LocalDataset()->col(j) = tmp.col(iPoint);
      iPoint++;
    }
    value.NumValues() = parent->Children()[i]->NumPoints();
  }

  assert(iPoint == numPoints);

}


template<typename TreeElemType>
bool DiscreteHilbertValue<TreeElemType>::HasValue() const
{
  return numValues > 0;
}

template<typename TreeElemType>
template<typename Archive>
void DiscreteHilbertValue<TreeElemType>::
Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(localDataset, "localDataset");
  ar & CreateNVP(ownsLocalDataset, "ownsLocalDataset");
  ar & CreateNVP(numValues, "numValues");
  ar & CreateNVP(valueToInsert, "valueToInsert");
  ar & CreateNVP(ownsValueToInsert, "ownsValueToInsert");
}

} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_IMPL_HPP
