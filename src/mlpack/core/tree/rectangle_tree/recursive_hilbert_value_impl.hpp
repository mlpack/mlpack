/**
 * @file recursive_hilbert_value_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the RecursiveHilbertValue class, a class that measures
 * ordering of points recursively.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RECURSIVE_HILBERT_VALUE_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_RECURSIVE_HILBERT_VALUE_IMPL_HPP

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

template<typename TreeElemType>
RecursiveHilbertValue<TreeElemType>::RecursiveHilbertValue() :
    largestValue(NULL),
    ownsLargestValue(false),
    hasLargestValue(false)
{

}

template<typename TreeElemType>
template<typename TreeType>
RecursiveHilbertValue<TreeElemType>::
RecursiveHilbertValue(const TreeType* tree) :
    largestValue(NULL),
    ownsLargestValue(false),
    hasLargestValue(false)
{
  if(!tree->Parent()) //  This is the root node
    ownsLargestValue = true;
  else if(tree->Parent()->Children()[0]->IsLeaf())
  {
    // This is a leaf node
    assert(tree->Parent()->NumChildren() > 0);
    ownsLargestValue = true;
  }
    
  if(ownsLargestValue)
  {
    largestValue =  new arma::Col<TreeElemType>(tree->LocalDataset().n_rows);
  }
}

template<typename TreeElemType>
RecursiveHilbertValue<TreeElemType>::
RecursiveHilbertValue(const RecursiveHilbertValue& other) :
    largestValue(const_cast<arma::Col<TreeElemType>*>(other.LargestValue())),
    ownsLargestValue(other.ownsLargestValue),
    hasLargestValue(other.hasLargestValue)
{

}

template<typename TreeElemType>
RecursiveHilbertValue<TreeElemType>::~RecursiveHilbertValue()
{
  if(ownsLargestValue)
    delete largestValue;
}

template<typename TreeElemType>
template<typename VecType1, typename VecType2>
int RecursiveHilbertValue<TreeElemType>::
ComparePoints(const VecType1& pt1, const VecType2& pt2,
                                typename boost::enable_if<IsVector<VecType1>>*,
                                typename boost::enable_if<IsVector<VecType2>>* )
{
  size_t dim = pt1.n_rows;
  CompareStruct comp(dim);

  return ComparePoints(pt1, pt2, comp);
};

template<typename TreeElemType>
int RecursiveHilbertValue<TreeElemType>::
CompareValues(const RecursiveHilbertValue& val1,
              const RecursiveHilbertValue& val2)
{
  if(!val1.hasLargestValue && val2.hasLargestValue)
    return -1;
  else if(val1.hasLargestValue && !val2.hasLargestValue)
    return 1;
  else if(!val1.hasLargestValue && !val2.hasLargestValue)
    return 0;

  return ComparePoints(*val1.LargestValue(),
                       *val2.LargestValue());
}

template<typename TreeElemType>
int RecursiveHilbertValue<TreeElemType>::
CompareWith(const RecursiveHilbertValue& val) const
{
  if(!hasLargestValue)
    return -1;
  return CompareValues(*this,val);
}

template<typename TreeElemType>
template<typename VecType>
int RecursiveHilbertValue<TreeElemType>::
CompareWith(const VecType& point,
            typename boost::enable_if<IsVector<VecType>>* ) const
{
  if(!hasLargestValue)
    return -1;
  return ComparePoints(*largestValue, point);
}

template<typename TreeElemType>
template<typename VecType>
int RecursiveHilbertValue<TreeElemType>::
CompareWithCachedPoint(const VecType& point,
                       typename boost::enable_if<IsVector<VecType>>* ) const
{
  return CompareWith(point);
}

template<typename TreeElemType>
template<typename VecType1, typename VecType2>
int RecursiveHilbertValue<TreeElemType>::
ComparePoints(const VecType1& pt1, const VecType2& pt2,
           CompareStruct& comp, typename boost::enable_if<IsVector<VecType1>>*,
                                typename boost::enable_if<IsVector<VecType2>>* )

{
  comp.center = comp.Hi * 0.5;
  comp.vec = comp.Lo * 0.5;

  comp.center += comp.vec;

  // Get bits in order to use the Gray code
  for(size_t i = 0; i < pt1.n_rows; i++)
  {
    size_t j = comp.permutation[i];
    comp.bits[i] = (pt1(j) > comp.center(j) && !comp.inversion[j]) ||
       (pt1(j) <= comp.center(j) && !comp.inversion[j]);

    comp.bits2[i] = (pt2(j) > comp.center(j) && !comp.inversion[j]) ||
       (pt2(j) <= comp.center(j) && !comp.inversion[j]);
  }

  // Gray encode
  for(size_t i = 1; i < pt1.n_rows; i++)
  {
    comp.bits[i] ^= comp.bits[i-1];
    comp.bits2[i] ^= comp.bits2[i-1];
  }

  if(comp.invertResult)
  {
    for(size_t i = 0; i < pt1.n_rows; i++)
    {
      comp.bits[i] = !comp.bits[i];
      comp.bits2[i] = !comp.bits2[i];
    }
  }

  for(size_t i = 0; i < pt1.n_rows; i++)
  {
    if(comp.bits[i] < comp.bits2[i])
      return -1;
    if(comp.bits[i] > comp.bits2[i])
      return 1;
  }

  if(comp.recursionLevel >= recursionDepth)
    return 0;

  comp.recursionLevel++;

  if(comp.bits[pt1.n_rows-1])
    comp.invertResult = !comp.invertResult;

  // Since the Hilbert curve is continuous we should permutate and intend
  // coordinate axes depending on the position of the point
  for(size_t i = 0; i < pt1.n_rows; i++)
  {
    size_t j = comp.permutation[i];
    size_t j0 = comp.permutation[0];
    if((pt1(j) > comp.center(j) && !comp.inversion[j]) ||
       (pt1(j) <= comp.center(j) && !comp.inversion[j]))
      comp.inversion[j0] = !comp.inversion[j0];
    else
    {
      size_t tmp;
      tmp = comp.permutation[0];
      comp.permutation[0] = comp.permutation[i];
      comp.permutation[i] = tmp;
    }
  }

  // Choose an appropriate subhypercube
  for(size_t i = 0; i < pt1.n_rows; i++)
  {
    if(pt1(i) > comp.center(i))
      comp.Lo(i) = comp.center(i);
    else
      comp.Hi(i) = comp.center(i);
  }

  return ComparePoints(pt1,pt2,comp);
}

template<typename TreeElemType>
template<typename TreeType, typename VecType>
size_t RecursiveHilbertValue<TreeElemType>::
InsertPoint(TreeType* node, const VecType& point,
                                 typename boost::enable_if<IsVector<VecType>>* )
{
  if(node->IsLeaf())
  {
    size_t i;

    for(i = 0; i < node->NumPoints(); i++)
      if(ComparePoints(node->LocalDataset().col(i), point) > 0)
        break;
    if(i == node->NumPoints())
      *largestValue = point;

    hasLargestValue = true;

    // Propogate changes of the largest Hilbert value downward
    TreeType* root = node->Parent();

    while(root != NULL)
    {
      root->AuxiliaryInfo().HilbertValue().LargestValue() = largestValue;
      root->AuxiliaryInfo().HilbertValue().hasLargestValue = true;

      root = root->Parent();
    }

    return i;
  }

  return 0;
}


template<typename TreeElemType>
template<typename TreeType>
void RecursiveHilbertValue<TreeElemType>::InsertNode(TreeType* node)
{
  if(CompareWith(node->AuxiliaryInfo().HilbertValue()) < 0)
  {
    largestValue = node->AuxiliaryInfo().HilbertValue().LargestValue();
    hasLargestValue = true;
  }
}

template<typename TreeElemType>
template<typename TreeType>
void RecursiveHilbertValue<TreeElemType>::
DeletePoint(TreeType* node, const size_t localIndex)
{
  if(node->NumPoints() <= 1)
  {
    hasLargestValue = false;
    return;
  }
  if(localIndex + 1 == node->NumPoints())
    *largestValue = node->LocalDataset()[localIndex-1];
  
}

template<typename TreeElemType>
template<typename TreeType>
void RecursiveHilbertValue<TreeElemType>::
RemoveNode(TreeType* node, const size_t nodeIndex)
{
  if(node->NumChildren() <= 1)
  {
    hasLargestValue = false;
    return;
  }
  if(nodeIndex + 1 == node->NumChildren())
    largestValue = node->Children()[nodeIndex-1]->AuxiliaryInfo.HilbertValue().LargestValue();

}

template<typename TreeElemType>
template<typename TreeType>
void RecursiveHilbertValue<TreeElemType>::Copy(TreeType* dst, TreeType* src)
{
  dst->AuxiliaryInfo().HilbertValue().LargestValue() =
    src->AuxiliaryInfo().HilbertValue().LargestValue();
  dst->AuxiliaryInfo().HilbertValue().hasLargestValue =
    src->AuxiliaryInfo().HilbertValue().hasLargestValue;
}

template<typename TreeElemType>
void RecursiveHilbertValue<TreeElemType>::NullifyData()
{
  ownsLargestValue = false;
}

template<typename TreeElemType>
template<typename TreeType>
void RecursiveHilbertValue<TreeElemType>::UpdateLargestValue(TreeType* node)
{
  if(!node->IsLeaf())
  {
    largestValue = (node->NumChildren() > 0 ?
                    node->Children()[node->NumChildren() - 1]->AuxiliaryInfo().HilbertValue().LargestValue() : NULL);
    hasLargestValue =  (node->NumChildren() > 0 ?
                    node->Children()[node->NumChildren() - 1]->AuxiliaryInfo().HilbertValue().hasLargestValue : false);
  }
}

template<typename TreeElemType>
template<typename TreeType>
void RecursiveHilbertValue<TreeElemType>::
UpdateHilbertValues(TreeType* parent, size_t firstSibling,  size_t lastSibling)
{
  for(size_t i = firstSibling; i<= lastSibling; i++)
  {
    RecursiveHilbertValue<TreeElemType> &value =
                          parent->Children()[i]->AuxiliaryInfo().HilbertValue();

    assert(parent->Children()[i]->NumPoints() > 0);

    *value.LargestValue() = parent->Children()[i]->LocalDataset().col(parent->Children()[i]->NumPoints() - 1);
    value.hasLargestValue = true;
  }

}

template<typename TreeElemType>
template<typename Archive>
void RecursiveHilbertValue<TreeElemType>::
Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(largestValue, "largestValue");
  ar & CreateNVP(ownsLargestValue, "ownsLargestValue");
  ar & CreateNVP(hasLargestValue, "hasLargestValue");
}

} // namespace tree
} // namespace mlpack

#endif //MLPACK_CORE_TREE_RECTANGLE_TREE_RECURSIVE_HILBERT_VALUE_IMPL_HPP
