
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree_first_point_is_root.hpp:

Program Listing for File first_point_is_root.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree_first_point_is_root.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/cover_tree/first_point_is_root.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP
   #define MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree {
   
   class FirstPointIsRoot
   {
    public:
     template<typename MatType>
     static size_t ChooseRoot(const MatType& /* dataset */) { return 0; }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif // MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP
