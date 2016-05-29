/**
 * @file no_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the NoAuxiliaryInformation class, a class that provides
 * no additional information about the nodes.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_NO_AUXILIARY_INFORMATION_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_NO_AUXILIARY_INFORMATION_HPP

namespace mlpack {
namespace tree {

template<typename TreeType>
class NoAuxiliaryInformation
{
 public:
  NoAuxiliaryInformation() { };
  NoAuxiliaryInformation(const TreeType *) { };
  NoAuxiliaryInformation(const TreeType &) { };

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
   * Serialize the information.
   */
  template<typename Archive>
  void Serialize(Archive &, const unsigned int /* version */) { };
};

} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_NO_AUXILIARY_INFORMATION_HPP
