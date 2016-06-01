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

inline DiscreteHilbertValue::DiscreteHilbertValue() :
    dataset(new arma::Mat<uint64_t>()),
    ownsDataset(true),
    localDataset(new std::list<arma::Col<uint64_t>>()),
    largestValue(localDataset->end())
{

};

inline DiscreteHilbertValue::~DiscreteHilbertValue()
{
  delete localDataset;
  if(ownsDataset)
    delete dataset;
};

template<typename TreeType>
DiscreteHilbertValue::DiscreteHilbertValue(const TreeType *tree) :
    dataset(tree->Parent() ?
          tree->Parent()->AuxiliaryInfo().LargestHilbertValue().Dataset() :
          new arma::Mat<uint64_t>(tree->Dataset().n_rows,
                                  tree->Dataset().n_cols)),
    ownsDataset(!tree->Parent()),
    localDataset(new std::list<arma::Col<uint64_t>>()),
    largestValue(localDataset->end())
{
  typedef typename TreeType::ElemType ElemType;
  // Calculate the Hilbert value for all points
  if(!tree->Parent())
  {
    for(size_t i = 0; i < tree->Dataset().n_cols; i++)
      dataset->col(i) = CalculateValue((arma::Col<ElemType>)tree->Dataset().col(i));
  }
}

template<typename TreeType>
DiscreteHilbertValue::DiscreteHilbertValue(const TreeType &other) :
    dataset(other.AuxiliaryInfo().LargestHilbertValue().Dataset()),
    ownsDataset(false),
    localDataset(new std::list<arma::Col<uint64_t>>()),
    largestValue(other.AuxiliaryInfo().LargestHilbertValue().LargestValue())
{
  if(other.IsLeaf())
  {
    std::list<arma::Col<uint64_t>> *otherDataset =
                     other.AuxiliaryInfo().LargestHilbertValue().LocalDataset();
    for(std::list<arma::Col<uint64_t>>::iterator it = otherDataset->begin(); it != otherDataset->end(); it++)
    {
      localDataset->push_back(*it);
    }
    largestValue = localDataset->end();
    if(otherDataset->size() > 0)
      largestValue--;
  }
}

template<typename ElemType>
arma::Col<uint64_t> DiscreteHilbertValue::
CalculateValue(const arma::Col<ElemType> &pt)
{
  arma::Col<uint64_t> res(pt.n_rows);
  constexpr int order = 64;   // The number of bits that we can store
  constexpr double numPowers =
                        std::log2(std::numeric_limits<ElemType>::max_exponent -
                        std::numeric_limits<ElemType>::min_exponent + 1.0);

  // The number of bits for the exponent
  constexpr int numExpBits = std::ceil(numPowers);
  // The number of bits for the mantissa
  constexpr int numMantBits = order - numExpBits - 1;

  for(size_t i = 0; i < pt.n_rows; i++)
  {
    int e;
    ElemType normalizedVal = std::frexp(pt(i),&e);
    bool sgn = std::signbit(normalizedVal);

    if(sgn)
      normalizedVal = -normalizedVal;

    if(e < std::numeric_limits<ElemType>::min_exponent)
    {
      uint64_t tmp = 1 << (std::numeric_limits<ElemType>::min_exponent - e);
      e = std::numeric_limits<ElemType>::min_exponent;
      normalizedVal /= tmp;
    }
    //  Extract the mantissa
    uint64_t tmp = (uint64_t)1 << numMantBits;
    res(i) = std::floor(normalizedVal / tmp);
    //  Add the exponent
    res(i) |= ((uint64_t)(e - std::numeric_limits<ElemType>::min_exponent)) << numMantBits;

    // Negative values should be inverted
    if(sgn)
      res(i) = ((uint64_t)1 << (order - 1)) - 1 - res(i);
    else
      res(i) |= (uint64_t)1 << (order - 1);
  }

  uint64_t M = (uint64_t)1 << (order - 1);

  // Since the Hilbert curve is continuous we should permutate and intend
  // coordinate axes depending on the position of the point
  for(uint64_t Q = M; Q > 1; Q >>= 1)
  {
    uint64_t P = Q - 1;

    for(size_t i = 0; i < pt.n_rows; i++)
    {
      if(res(i) & Q)  // Invert
        res(0) ^= P;
      else            // Permutate
      {
        uint64_t t = (res(0) ^ res(i)) & P;
        res(0) ^= t;
        res(i) ^= t;
      }
    }
  }

  // Gray encode
  for(size_t i = 1; i < pt.n_rows; i++)
    res(i) ^= res(i-1);

  uint64_t t = 0;

  // Some coordinate axes should be inverted
  for(uint64_t Q = M; Q > 1; Q >>= 1)
    if( res(pt.n_rows - 1) & Q)
      t ^= Q - 1;

  for(size_t i = 0; i < pt.n_rows; i++)
    res(i) ^= t;

  // We should rearrange bits in order to compare two Hilbert values faster
  arma::Col<uint64_t> rearrangedResult(pt.n_rows,arma::fill::zeros);

  for(size_t i = 0; i < order; i++)
    for(size_t j = 0; j < pt.n_rows; j++)
    {
      size_t bit = (i * pt.n_rows + j) % order;
      size_t row = (i * pt.n_rows + j) / order;

      rearrangedResult(row) |= (res(j) & (1 << i)) >> (i - bit);
    }
      
  return rearrangedResult;
}

inline int DiscreteHilbertValue::
CompareValues(const arma::Col<uint64_t> &value1,
              const arma::Col<uint64_t> &value2)
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



template<typename ElemType>
int DiscreteHilbertValue::ComparePoints(const arma::Col<ElemType> &pt1,
                                        const arma::Col<ElemType> &pt2)
{
  arma::Col<uint64_t> val1 = CalculateValue(pt1);
  arma::Col<uint64_t> val2 = CalculateValue(pt2);

  return CompareValues(val1,val2);
}

template<typename TreeType>
int DiscreteHilbertValue::CompareValues(TreeType *,
                         DiscreteHilbertValue &val1, DiscreteHilbertValue &val2)
{
  if(val1.HasValue() && !val2.HasValue())
    return 1;
  else if(!val1.HasValue() && val2.HasValue())
    return -1;
  else if(!val1.HasValue() && !val2.HasValue())
    return 0;

  return CompareValues(*val1.LargestValue(),*val2.LargestValue());
}

template<typename TreeType>
int DiscreteHilbertValue::CompareWith(TreeType *, DiscreteHilbertValue &val)
{
  return CompareValues(*largestValue,*val.LargestValue());
}

template<typename TreeType,typename ElemType>
int DiscreteHilbertValue::CompareWith(TreeType *tree,
                                      const arma::Col<ElemType> &pt)
{
  arma::Col<uint64_t> val = CalculateValue(pt);

  if(!HasValue())
    return -1;

  return CompareValues(*largestValue,val);
}

template<typename TreeType>
int DiscreteHilbertValue::CompareWith(TreeType *,
                                      const size_t point)
{
  if(!HasValue())
    return -1;
  return CompareValues(*largestValue,dataset->col(point));
}

template<typename TreeType>
size_t DiscreteHilbertValue::InsertPoint(TreeType *node, const size_t point)
{
  size_t i = 0;
  std::list<arma::Col<uint64_t>>::iterator it = localDataset->end();

  if(node->IsLeaf())
  {
    // Find an appropriate place
    for(it = localDataset->begin(); it != localDataset->end(); it++)
    {
      if(CompareValues(*it, dataset->col(point)) > 0)
        break;
      i++;
    }
    std::list<arma::Col<uint64_t>>::iterator insertedIterator =
                                   localDataset->insert(it,dataset->col(point));
    // Update the largest Hilbert value
    if(it == localDataset->end())
      largestValue = insertedIterator;

    // Propogate changes of the largest Hilbert value downward
    TreeType *root = node->Parent();

    while(root != NULL)
    {
      if(root->AuxiliaryInfo().LargestHilbertValue().LargestValue() ==
          root->AuxiliaryInfo().LargestHilbertValue().LocalDataset()->end())
        root->AuxiliaryInfo().LargestHilbertValue().LargestValue() = insertedIterator;

      root = root->Parent();
    }
  }
  else if(largestValue != localDataset->end())
  {
    // We do not update the largest Hilbert value since we do not know the
    // iterator
    if(CompareValues(*largestValue,dataset->col(point)) < 0)
      largestValue = localDataset->end();
  }

  return i;
}

template<typename TreeType>
void DiscreteHilbertValue::InsertNode(TreeType *node)
{
  std::list<arma::Col<uint64_t>>::iterator it =
                    node->AuxiliaryInfo().LargestHilbertValue().LargestValue();

  // Update the largest Hilbert value
  if(largestValue != localDataset->end() &&
        it != node->AuxiliaryInfo().LargestHilbertValue().LocalDataset()->end())
    if(*it > *largestValue)
      largestValue = it;
}

template<typename TreeType>
void DiscreteHilbertValue::DeletePoint(TreeType *node, const size_t localIndex)
{
  std::list<arma::Col<uint64_t>>::iterator it = localDataset->begin();

  // Delete the Hilbert value from the local dataset
  for(size_t i=0; i < localIndex; i++)
    it++;
  localDataset->erase(it);

  // Update the largest Hilbert value
  if(localDataset->size() == 0)
    largestValue = localDataset->end();
  else
  {
    largestValue = localDataset->end();
    largestValue--;
  }
}

template<typename TreeType>
void DiscreteHilbertValue::RemoveNode(TreeType *node, const size_t nodeIndex)
{
  if(node->NumChildren() <= 1)
  {
    largestValue = localDataset->end();
    return;
  }
  if(nodeIndex + 1 == node->NumChildren())
  {
    // Update the largest Hilbert value if the value exists
    TreeType *child = node->Children()[nodeIndex-1];
    if(child->AuxiliaryInfo.LargestHilbertValue().LargestValue() !=
       child->AuxiliaryInfo.LargestHilbertValue().LocalDataset()->end())
      largestValue = child->AuxiliaryInfo.LargestHilbertValue().LargestValue();
    else
      largestValue = localDataset->end();
  }
}

template<typename TreeType>
void DiscreteHilbertValue::Copy(TreeType *dst, TreeType *src)
{
  DiscreteHilbertValue &dstVal = dst->AuxiliaryInfo().LargestHilbertValue();
  DiscreteHilbertValue &srcVal = src->AuxiliaryInfo().LargestHilbertValue();

  // Copy the largest Hilbert value and the local dataset
  dstVal.LargestValue() = srcVal.LargestValue();

  dstVal.LocalDataset()->clear();
  std::list<arma::Col<uint64_t>>::iterator it = srcVal.LocalDataset()->begin();
  for( ; it != srcVal.LocalDataset()->end(); it++)
    dstVal.LocalDataset()->push_back(*it);

  if(dst->IsLeaf())
  {
    dstVal.LargestValue() = dstVal.LocalDataset()->end();
    if(dst->NumPoints() > 0)
      dstVal.LargestValue()--;
  }
}

inline DiscreteHilbertValue DiscreteHilbertValue::operator = (const DiscreteHilbertValue &val)
{
  // Copy the largest Hilbert value
  largestValue = val.LargestValue();

  return *this;
}

template<typename TreeType>
void DiscreteHilbertValue::UpdateLargestValue(TreeType *node)
{
  if(node->IsLeaf())
  {
    // Update the largest Hilbert value and the local dataset
    localDataset->clear();
    if(node->NumPoints() == 0)
    {
      largestValue = localDataset->end();
      return;
    }
    for(size_t i = 0; i < node->NumPoints(); i++)
      localDataset->push_back(dataset->col(node->Points()[i]));
    largestValue = localDataset->end();
    largestValue--;
  }
  else
  {
    if(localDataset->size() > 0)
      localDataset->clear();
    //  Update the largest Hilbert value;
    if(node->NumChildren() == 0)
      largestValue = localDataset->end();
    else if(node->Children()[node->NumChildren()-1]->AuxiliaryInfo().LargestHilbertValue().HasValue())
      largestValue = node->Children()[node->NumChildren()-1]->AuxiliaryInfo().LargestHilbertValue().LargestValue();
    else
      largestValue = localDataset->end();
  }
}

inline bool DiscreteHilbertValue::HasValue()
{
  return largestValue != localDataset->end();
}

} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_IMPL_HPP
