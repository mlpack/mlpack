/**
 * @file hilbert_r_tree_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the HilbertRTreeAuxiliaryInformation class,
 * a class that provides some Hilbert r-tree specific information
 * about the nodes.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_IMPL_HPP

#include "hilbert_r_tree_auxiliary_information.hpp"

namespace mlpack {
namespace tree {


template<typename TreeType,typename HilbertValue>
HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
HilbertRTreeAuxiliaryInformation()
{

};

template<typename TreeType,typename HilbertValue>
HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
HilbertRTreeAuxiliaryInformation(const TreeType *node) :
    largestHilbertValue(node)
{

};

template<typename TreeType,typename HilbertValue>
HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
HilbertRTreeAuxiliaryInformation(const TreeType &other) :
    largestHilbertValue(other)
{

};
  
template<typename TreeType,typename HilbertValue>
bool HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
HandlePointInsertion(TreeType *node,const size_t point)
{
  if(node->IsLeaf())
  {
    size_t pos = largestHilbertValue.InsertPoint(node,point);
    
    for(size_t i = node->NumPoints(); i > pos; i--)
    {
      node->Points()[i] = node->Points()[i-1];
      node->LocalDataset()->col(i) = node->LocalDataset()->col(i-1);
    }
    node->Points()[pos] = point;
    node->LocalDataset()->col(pos) = node->Dataset()->col(point);
    node->NumPoints()++;
  }
  else
    largestHilbertValue.InsertPoint(node,point);

  return true;
}

template<typename TreeType,typename HilbertValue>
bool HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
HandleNodeInsertion(TreeType *node,TreeType *nodeToInsert,bool insertionLevel)
{
  if(insertionLevel)
  {
    size_t pos;
    
    for(pos = 0; pos < node->NumChildren(); pos++)
      if(HilbertValue::CompareValues(
                 node->Children()[pos]->AuxiliaryInfo().LargestHilbertValue(),
                 nodeToInsert->AuxiliaryInfo().LargestHilbertValue()) < 0)
          break;
    
    for(size_t i = node->NumChildren(); i > pos; i--)
      node->Children()[i] = node->Children()[i-1];
    
    node->Children()[pos] = nodeToInsert;
    nodeToInsert->Parent() = node;
    largestHilbertValue.InsertNode(nodeToInsert);
  }
  else
    largestHilbertValue.InsertNode(nodeToInsert);

  return true;
}

template<typename TreeType,typename HilbertValue>
bool HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
HandlePointDeletion(TreeType *node,const size_t localIndex)
{
  largestHilbertValue.DeletePoint(node,localIndex);

  for(size_t i = localIndex + 1; localIndex < node->NumPoints(); i++)
  {
    node->Points()[i-1] = node->Points()[i];
    node->LocalDataset()->col(i-1) = node->LocalDataset()->col(i);
  }
  node->NumPoints()--;
  return true;
}

template<typename TreeType,typename HilbertValue>
bool HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
HandleNodeRemoval(TreeType *node,const size_t nodeIndex)
{
  largestHilbertValue.RemoveNode(node,nodeIndex);

  for(size_t i = nodeIndex + 1; nodeIndex < node->NumChildren(); i++)
    node->Children()[i-1] = node->Children()[i];

  node->NumChildren()--;
  return true;
}

template<typename TreeType,typename HilbertValue>
bool HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
ShrinkAuxiliaryInfo(TreeType *node)
{
  if(node->IsLeaf())
    return true;

  TreeType *child = node->Children()[node->NumChildren()-1];
  if(HilbertValue::CompareValues(largestHilbertValue,
                            child->AuxiliaryInfo().LargestHilbertValue()) > 0)
  {
    largestHilbertValue = child->AuxiliaryInfo().LargestHilbertValue();
    return true;
  }
  return false;
}

template<typename TreeType,typename HilbertValue>
void HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
Copy(TreeType *dst,TreeType *src)
{
  largestHilbertValue.Copy(dst,src);
}

template<typename TreeType,typename HilbertValue>
template<typename Archive>
void HilbertRTreeAuxiliaryInformation<TreeType,HilbertValue>::
Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(largestHilbertValue, "largestHilbertValue");
}


} // namespace tree
} // namespace mlpack

#endif//MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_IMPL_HPP
