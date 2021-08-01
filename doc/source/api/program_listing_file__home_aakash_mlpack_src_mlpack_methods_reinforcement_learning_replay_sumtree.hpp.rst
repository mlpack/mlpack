
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_replay_sumtree.hpp:

Program Listing for File sumtree.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_replay_sumtree.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/replay/sumtree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_SUMTREE_HPP
   #define MLPACK_METHODS_RL_SUMTREE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace rl {
   
   template<typename T>
   class SumTree
   {
    public:
     SumTree() : capacity(0)
     { /* Nothing to do here. */ }
   
     SumTree(const size_t capacity) : capacity(capacity)
     {
       element = std::vector<T>(2 * capacity);
     }
   
     void Set(size_t idx, const T value)
     {
       idx += capacity;
       element[idx] = value;
       idx /= 2;
       while (idx >= 1)
       {
         element[idx] = element[2 * idx] + element[2 * idx + 1];
         idx /= 2;
       }
     }
   
     void BatchUpdate(const arma::ucolvec& indices, const arma::Col<T>& data)
     {
       for (size_t i = 0; i < indices.n_rows; ++i)
       {
         element[indices[i] + capacity] = data[i];
       }
       // update the total tree with bottom-up technique.
       for (size_t i = capacity - 1; i > 0; i--)
       {
         element[i] = element[2 * i] + element[2 * i + 1];
       }
     }
   
     T Get(size_t idx)
     {
       idx += capacity;
       return element[idx];
     }
   
     T SumHelper(const size_t start,
                 const size_t end,
                 const size_t node,
                 const size_t nodeStart,
                 const size_t nodeEnd)
     {
       if (start == nodeStart && end == nodeEnd)
       {
         return element[node];
       }
       size_t mid = (nodeStart + nodeEnd) / 2;
       if (end <= mid)
       {
         return SumHelper(start, end, 2 * node, nodeStart, mid);
       }
       else
       {
         if (mid + 1 <= start)
         {
           return SumHelper(start, end, 2 * node + 1, mid + 1 , nodeEnd);
         }
         else
         {
           return SumHelper(start, mid, 2 * node, nodeStart, mid) +
               SumHelper(mid + 1, end, 2 * node + 1, mid + 1 , nodeEnd);
         }
       }
     }
   
     T Sum(const size_t start, size_t end)
     {
       end -= 1;
       return SumHelper(start, end, 1, 0, capacity - 1);
     }
   
     T Sum()
     {
       return Sum(0, capacity);
     }
   
     size_t FindPrefixSum(T mass)
     {
       size_t idx = 1;
       while (idx < capacity)
       {
         if (element[2 * idx] > mass)
         {
           idx = 2 * idx;
         }
         else
         {
           mass -= element[2 * idx];
           idx = 2 * idx + 1;
         }
       }
       return idx - capacity;
     }
   
    private:
     size_t capacity;
   
     std::vector<T> element;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
