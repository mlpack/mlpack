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


template<typename TreeType,
         template<typename> class HilbertValueType>
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HilbertRTreeAuxiliaryInformation()
{

};

template<typename TreeType,
         template<typename> class HilbertValueType>
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HilbertRTreeAuxiliaryInformation(const TreeType* node) :
    hilbertValue(node)
{

};

template<typename TreeType,
         template<typename> class HilbertValueType>
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HilbertRTreeAuxiliaryInformation(const HilbertRTreeAuxiliaryInformation& other) :
    hilbertValue(other.HilbertValue())
{

};

template<typename TreeType,
         template<typename> class HilbertValueType>
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
~HilbertRTreeAuxiliaryInformation()
{

}
  
template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HandlePointInsertion(TreeType* node, const size_t point)
{
  if(node->IsLeaf())
  {
    // Get the position at which the point should be inserted
    // Update the largest Hilbert value of the node
    size_t pos = hilbertValue.InsertPoint(node, node->Dataset().col(point));

    // Move points
    for(size_t i = node->NumPoints(); i > pos; i--)
    {
      node->Points()[i] = node->Points()[i-1];
      node->LocalDataset().col(i) = node->LocalDataset().col(i-1);
    }
    // Insert the point
    node->Points()[pos] = point;
    node->LocalDataset().col(pos) = node->Dataset().col(point);
    node->Count()++;
  }
  else
  {
    // Calculate the Hilbert value
    hilbertValue.InsertPoint(node, node->Dataset().col(point));
  }

  return true;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
template<typename VecType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HandlePointInsertion(TreeType* node, const VecType& point,
                                  typename boost::enable_if<IsVector<VecType>>*)
{
  if(node->IsLeaf())
  {
    // Get the position at which the point should be inserted
    // Update the largest Hilbert value of the node
    size_t pos = hilbertValue.InsertPoint(node, point);

    // Move points
    for(size_t i = node->NumPoints(); i > pos; i--)
    {
      node->Points()[i] = node->Points()[i-1];
      node->LocalDataset().col(i) = node->LocalDataset().col(i-1);
    }
    // Insert the point
    node->Points()[pos] = node->Dataset().n_cols;
    node->LocalDataset().col(pos) = point;
    node->Count()++;
  }
  else
  {
    // Calculate the Hilbert value
    hilbertValue.InsertPoint(node, point);
  }

  return true;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HandleNodeInsertion(TreeType* node,TreeType* nodeToInsert,bool insertionLevel)
{
  if(insertionLevel)
  {
    size_t pos;

    // Find the best position for the node being inserted.
    // The node should be inserted according to its Hilbert value.
    for(pos = 0; pos < node->NumChildren(); pos++)
      if(HilbertValueType<ElemType>::CompareValues(
                 node->Children()[pos]->AuxiliaryInfo().HilbertValue(),
                 nodeToInsert->AuxiliaryInfo().HilbertValue()) < 0)
          break;

    // Move nodes
    for(size_t i = node->NumChildren(); i > pos; i--)
      node->Children()[i] = node->Children()[i-1];

    // Insert the node
    node->Children()[pos] = nodeToInsert;
    nodeToInsert->Parent() = node;

    // Update the largest Hilbert value
    hilbertValue.InsertNode(nodeToInsert);
  }
  else
    hilbertValue.InsertNode(nodeToInsert); //  Update LHV

  return true;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HandlePointDeletion(TreeType* node,const size_t localIndex)
{
  // Update the largest Hilbert value
  hilbertValue.DeletePoint(node,localIndex);

  for(size_t i = localIndex + 1; localIndex < node->NumPoints(); i++)
  {
    node->Points()[i-1] = node->Points()[i];
    node->LocalDataset()->col(i-1) = node->LocalDataset()->col(i);
  }
  node->NumPoints()--;
  return true;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HandleNodeRemoval(TreeType* node,const size_t nodeIndex)
{
  // Update the largest Hilbert value
  hilbertValue.RemoveNode(node,nodeIndex);

  for(size_t i = nodeIndex + 1; nodeIndex < node->NumChildren(); i++)
    node->Children()[i-1] = node->Children()[i];

  node->NumChildren()--;
  return true;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
UpdateAuxiliaryInfo(TreeType* node)
{
  if(node->IsLeaf())  //  Should already be updated
    return true;

  TreeType *child = node->Children()[node->NumChildren()-1];
  if(HilbertValueType<ElemType>::CompareValues(hilbertValue,
                            child->AuxiliaryInfo().hilbertValue()) < 0)
  {
    hilbertValue.Copy(node,child);
//    hilbertValue = child->AuxiliaryInfo().hilbertValue();
    return true;
  }
  return false;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
void HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
Copy(TreeType* dst,TreeType* src)
{
  hilbertValue.Copy(dst,src);
}

template<typename TreeType,
         template<typename> class HilbertValueType>
void HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
NullifyData()
{
  hilbertValue.NullifyData();
}

template<typename TreeType,
         template<typename> class HilbertValueType>
template<typename Archive>
void HilbertRTreeAuxiliaryInformation<TreeType ,HilbertValueType>::
Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(hilbertValue, "hilbertValue");
}


} // namespace tree
} // namespace mlpack

#endif//MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_IMPL_HPP
