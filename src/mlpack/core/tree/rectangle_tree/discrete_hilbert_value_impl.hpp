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
          new arma::Mat<uint64_t>(tree->Dataset()->n_rows,
                                  tree->MaxLeafSize()+1)),
    ownsDataset(!tree->Parent()),
    localDataset(new std::list<arma::Col<uint64_t>>()),
    largestValue(localDataset->end())
{
  if(!tree->Parent())
  {
    for(size_t i = 0; i < tree->Dataset()->n_rows; i++)
      dataset->col(i) = CalculateValue(tree->Dataset()->col(i));
  }
};

template<typename TreeType>
DiscreteHilbertValue::DiscreteHilbertValue(const TreeType &other) :
    dataset(other.AuxiliaryInfo().LargestHilbertValue().Dataset()),
    ownsDataset(!other.Parent()),
    localDataset(other.AuxiliaryInfo().LargestHilbertValue().LocalDataset()),
    largestValue(other.AuxiliaryInfo().LargestHilbertValue().LargestValue())
{
};

template<typename ElemType>
arma::Col<uint64_t> CalculateValue(const arma::Col<ElemType> &pt)
{
  arma::Col<uint64_t> res(pt.n_rows);
  constexpr int order = 64;
  constexpr double numPowers =
                        std::log2(std::numeric_limits<ElemType>::max_exponent -
                        std::numeric_limits<ElemType>::min_exponent + 1.0);

  constexpr int numExpBits = std::ceil(numPowers);
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

    uint64_t tmp = 1 << numMantBits;
    res(i) = std::floor(normalizedVal / numMantBits);
    res(i) |= (e - std::numeric_limits<ElemType>::min_exponent) << numMantBits;

    if(sgn)
      res(i) = 1 << (order - 1) - 1 - res(i);
    else
      res(i) |= 1 << (order - 1);
  }

  uint64_t M = 1 << (order - 1);

  for(uint64_t Q = M; Q > 1; Q >>= 1)
  {
    uint64_t P = Q - 1;

    for(size_t i = 0; i < pt.n_rows; i++)
    {
      if(res(i) & Q)
        res(0) ^= P;
      else
      {
        uint64_t t = (res(0) ^ res(i)) & P;
        res(0) ^= t;
        res(i) ^= t;
      }
    }
  }

  for(size_t i = 1; i < pt.n_rows; i++)
    res(i) ^= res(i-1);

  uint64_t t = 0;

  for(uint64_t Q = M; Q > 1; Q >>= 1)
    if( res(pt.n_rows - 1) & Q)
      t ^= Q - 1;

  for(size_t i = 0; i < pt.n_rows; i++)
    res(i) ^= t;

  return res;
}


template<typename ElemType>
int DiscreteHilbertValue::ComparePoints(const arma::Col<ElemType> &pt1,
                                        const arma::Col<ElemType> &pt2)
{
  arma::Col<uint64_t> val1 = CalculateValue(pt1);
  arma::Col<uint64_t> val2 = CalculateValue(pt2);

  if(val1 > val2)
    return 1;
  else if(val2 > val1)
    return -1;
  return 0;
}

template<typename TreeType>
int DiscreteHilbertValue::CompareValues(TreeType *tree,
                         DiscreteHilbertValue &val1, DiscreteHilbertValue &val2)
{
  if(*val1.LargestValue() > *val1.LargestValue())
    return 1;
  else if(*val1.LargestValue() < *val1.LargestValue())
    return -1;

  return 0;
}

template<typename TreeType>
int DiscreteHilbertValue::CompareWith(TreeType *tree, DiscreteHilbertValue &val)
{
  if(*largestValue > *val.LargestValue())
    return 1;
  else if(*largestValue < *val.LargestValue())
    return -1;

  return 0;
}

template<typename TreeType,typename ElemType>
int DiscreteHilbertValue::CompareWith(TreeType *tree,
                                      const arma::Col<ElemType> &pt)
{
  arma::Col<uint64_t> val = CalculateValue(pt);

  if(*largestValue > val)
    return 1;
  else if(*largestValue < val)
    return -1;

  return 0;
}

template<typename TreeType>
size_t DiscreteHilbertValue::InsertPoint(TreeType *node, const size_t point)
{
  size_t i = 0;
  std::list<arma::Col<uint64_t>>::iterator it;

  if(node->IsLeaf())
  {
    for(it = localDataset->begin(); it != localDataset->end(); it++)
    {
      if(*it > dataset->col(point))
        break;
      i++;
    }
    std::list<arma::Col<uint64_t>>::iterator insertedIterator =
                                   localDataset->insert(it,dataset->col(point));
    if(it == localDataset->end())
      largestValue = insertedIterator;

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
    if(*largestValue < dataset->col(point))
      largestValue = localDataset->end();
  }

  return i;
}

template<typename TreeType>
void DiscreteHilbertValue::InsertNode(TreeType *node)
{
  std::list<arma::Col<uint64_t>>::iterator it =
                    node->AuxiliaryInfo().LargestHilbertValue().LargestValue();

  if(largestValue != localDataset->end() &&
        it != node->AuxiliaryInfo().LargestHilbertValue().LocalDataset()->end())
    if(*it > *largestValue)
      largestValue = it;
}

template<typename TreeType>
void DiscreteHilbertValue::DeletePoint(TreeType *node, const size_t localIndex)
{
  std::list<arma::Col<uint64_t>>::iterator it = localDataset->begin();

  for(size_t i=0; i < localIndex; i++)
    it++;
  localDataset->erase(it);
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

  dst.LargestValue() = src.LargestValue();

  dst.LocalDataset()->clear();
  std::list<arma::Col<uint64_t>>::iterator it = src.LocalDataset()->begin();
  for( ; it != src.LocalDataset()->end(); it++)
    dst.LocalDataset()->push_back(*it);
    
}

inline DiscreteHilbertValue DiscreteHilbertValue::operator = (DiscreteHilbertValue &val)
{
  largestValue = val.LargestValue();

  return *this;
}

} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_IMPL_HPP
