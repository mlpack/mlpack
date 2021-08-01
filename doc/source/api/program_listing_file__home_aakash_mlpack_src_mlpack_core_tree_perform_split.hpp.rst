
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_perform_split.hpp:

Program Listing for File perform_split.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_perform_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/perform_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_PERFORM_SPLIT_HPP
   #define MLPACK_CORE_TREE_PERFORM_SPLIT_HPP
   
   namespace mlpack {
   namespace tree  {
   namespace split {
   
   template<typename MatType, typename SplitType>
   size_t PerformSplit(MatType& data,
                       const size_t begin,
                       const size_t count,
                       const typename SplitType::SplitInfo& splitInfo)
   {
     // This method modifies the input dataset.  We loop both from the left and
     // right sides of the points contained in this node.
     size_t left = begin;
     size_t right = begin + count - 1;
   
     // First half-iteration of the loop is out here because the termination
     // condition is in the middle.
     while ((left <= right) &&
         (SplitType::AssignToLeftNode(data.col(left), splitInfo)))
       left++;
     while ((!SplitType::AssignToLeftNode(data.col(right), splitInfo)) &&
         (left <= right) && (right > 0))
       right--;
   
     // Shortcut for when all points are on the right.
     if (left == right && right == 0)
       return left;
   
     while (left <= right)
     {
       // Swap columns.
       data.swap_cols(left, right);
   
       // See how many points on the left are correct.  When they are correct,
       // increase the left counter accordingly.  When we encounter one that isn't
       // correct, stop.  We will switch it later.
       while (SplitType::AssignToLeftNode(data.col(left), splitInfo) &&
           (left <= right))
         left++;
   
       // Now see how many points on the right are correct.  When they are correct,
       // decrease the right counter accordingly.  When we encounter one that isn't
       // correct, stop.  We will switch it with the wrong point we found in the
       // previous loop.
       while ((!SplitType::AssignToLeftNode(data.col(right), splitInfo)) &&
           (left <= right))
         right--;
     }
   
     Log::Assert(left == right + 1);
   
     return left;
   }
   
   template<typename MatType, typename SplitType>
   size_t PerformSplit(MatType& data,
                       const size_t begin,
                       const size_t count,
                       const typename SplitType::SplitInfo& splitInfo,
                       std::vector<size_t>& oldFromNew)
   {
     // This method modifies the input dataset.  We loop both from the left and
     // right sides of the points contained in this node.
     size_t left = begin;
     size_t right = begin + count - 1;
   
     // First half-iteration of the loop is out here because the termination
     // condition is in the middle.
     while ((left <= right) &&
            (SplitType::AssignToLeftNode(data.col(left), splitInfo)))
       left++;
     while ((!SplitType::AssignToLeftNode(data.col(right), splitInfo)) &&
            (left <= right) && (right > 0))
       right--;
   
     // Shortcut for when all points are on the right.
     if (left == right && right == 0)
       return left;
   
     while (left <= right)
     {
       // Swap columns.
       data.swap_cols(left, right);
   
       // Update the indices for what we changed.
       size_t t = oldFromNew[left];
       oldFromNew[left] = oldFromNew[right];
       oldFromNew[right] = t;
   
       // See how many points on the left are correct.  When they are correct,
       // increase the left counter accordingly.  When we encounter one that isn't
       // correct, stop.  We will switch it later.
       while (SplitType::AssignToLeftNode(data.col(left), splitInfo) &&
           (left <= right))
         left++;
   
       // Now see how many points on the right are correct.  When they are correct,
       // decrease the right counter accordingly.  When we encounter one that isn't
       // correct, stop.  We will switch it with the wrong point we found in the
       // previous loop.
       while ((!SplitType::AssignToLeftNode(data.col(right), splitInfo)) &&
           (left <= right))
         right--;
     }
   
     Log::Assert(left == right + 1);
     return left;
   }
   
   } // namespace split
   } // namespace tree
   } // namespace mlpack
   
   
   #endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_PERFORM_SPLIT_HPP
