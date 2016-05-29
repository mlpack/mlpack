/**
 * @file no_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the XTreeAuxiliaryInformation class, a class that provides
 * some x-tree specific information about the nodes.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_AUXILIARY_INFORMATION_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_AUXILIARY_INFORMATION_HPP

namespace mlpack {
namespace tree {

template<typename TreeType>
class XTreeAuxiliaryInformation
{
 public:
  XTreeAuxiliaryInformation() :
    normalNodeMaxNumChildren(0),
    splitHistory(0)
  { };

  XTreeAuxiliaryInformation(const TreeType *node) :
    normalNodeMaxNumChildren(node->Parent() ? 
                    node->Parent()->AuxiliaryInfo().NormalNodeMaxNumChildren() :
                    node->MaxNumChildren()),
    splitHistory(node->Bound().Dim())
  { };

  XTreeAuxiliaryInformation(const TreeType &other) :
    normalNodeMaxNumChildren(other.AuxiliaryInfo().NormalNodeMaxNumChildren()),
    splitHistory(other.AuxiliaryInfo().SplitHistory())
  { };

  bool HandlePointInsertion(TreeType *, const size_t)
  {
    return false;
  }

  bool HandleNodeInsertion(TreeType *,TreeType *,bool)
  {
    return false;
  }

  bool HandlePointDeletion(TreeType *,const size_t)
  {
    return false;
  }

  bool HandleNodeRemoval(TreeType *,const size_t)
  {
    return false;
  }

  bool ShrinkAuxiliaryInfo(TreeType *)
  {
    return false;
  }

  void Copy(TreeType *,TreeType *)
  { }

  /**
   * The X tree requires that the tree records it's "split history".  To make
   * this easy, we use the following structure.
   */
  typedef struct SplitHistoryStruct
  {
    int lastDimension;
    std::vector<bool> history;

    SplitHistoryStruct(int dim) : lastDimension(0), history(dim)
    {
      for (int i = 0; i < dim; i++)
        history[i] = false;
    }

    template<typename Archive>
    void Serialize(Archive& ar, const unsigned int /* version */)
    {
      ar & data::CreateNVP(lastDimension, "lastDimension");
      ar & data::CreateNVP(history, "history");
    }
  } SplitHistoryStruct;

 private:
    //! The max number of child nodes a non-leaf normal node can have.
  size_t normalNodeMaxNumChildren;
  //! A struct to store the "split history" for X trees.
  SplitHistoryStruct splitHistory;

 public:
  //! Return the maximum number of a normal node's children.
  size_t NormalNodeMaxNumChildren() const { return normalNodeMaxNumChildren; }
  //! Modify the maximum number of a normal node's children.
  size_t& NormalNodeMaxNumChildren() { return normalNodeMaxNumChildren; }
  //! Return the split history of the node assosiated with this object.
  const SplitHistoryStruct& SplitHistory() const { return splitHistory; }
  //! Modify the split history of the node assosiated with this object.
  SplitHistoryStruct& SplitHistory() { return splitHistory; }

  /**
   * Serialize the information.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    using data::CreateNVP;

    ar & CreateNVP(normalNodeMaxNumChildren, "normalNodeMaxNumChildren");
    ar & CreateNVP(splitHistory, "splitHistory");
  }

};

} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_AUXILIARY_INFORMATION_HPP
