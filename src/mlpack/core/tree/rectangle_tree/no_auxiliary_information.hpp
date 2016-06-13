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
  NoAuxiliaryInformation(const TreeType* ) { };
  NoAuxiliaryInformation(const TreeType& ) { };

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method should return false if it does not handle the process.
   */
  bool HandlePointInsertion(TreeType* , const size_t)
  {
    return false;
  }

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method should return false if it does not handle the process.
   */
  bool HandleNodeInsertion(TreeType* , TreeType* ,bool)
  {
    return false;
  }

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method should return false if it does not handle the process.
   */
  bool HandlePointDeletion(TreeType* , const size_t)
  {
    return false;
  }

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method should return false if it does not handle the process.
   */
  bool HandleNodeRemoval(TreeType* , const size_t)
  {
    return false;
  }

  /**
   * Some tree types require to propagate the information downward.
   * This method should return false if this is not the case.
   */
  bool UpdateAuxiliaryInfo(TreeType* )
  {
    return false;
  }

  /**
   * Nothing to copy.
   */
  void Copy(TreeType* , TreeType* )
  { }

  void NullifyData()
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
