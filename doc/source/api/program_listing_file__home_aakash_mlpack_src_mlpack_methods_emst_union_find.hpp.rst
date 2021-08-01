
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_emst_union_find.hpp:

Program Listing for File union_find.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_emst_union_find.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/emst/union_find.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_EMST_UNION_FIND_HPP
   #define MLPACK_METHODS_EMST_UNION_FIND_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace emst {
   
   class UnionFind
   {
    private:
     arma::Col<size_t> parent;
     arma::ivec rank;
   
    public:
     UnionFind(const size_t size) : parent(size), rank(size)
     {
       for (size_t i = 0; i < size; ++i)
       {
         parent[i] = i;
         rank[i] = 0;
       }
     }
   
     ~UnionFind() { }
   
     size_t Find(const size_t x)
     {
       if (parent[x] == x)
       {
         return x;
       }
       else
       {
         // This ensures that the tree has a small depth.
         parent[x] = Find(parent[x]);
         return parent[x];
       }
     }
   
     void Union(const size_t x, const size_t y)
     {
       const size_t xRoot = Find(x);
       const size_t yRoot = Find(y);
   
       if (xRoot == yRoot)
       {
         return;
       }
       else if (rank[xRoot] == rank[yRoot])
       {
         parent[yRoot] = parent[xRoot];
         rank[xRoot] = rank[xRoot] + 1;
       }
       else if (rank[xRoot] > rank[yRoot])
       {
         parent[yRoot] = xRoot;
       }
       else
       {
         parent[xRoot] = yRoot;
       }
     }
   }; // class UnionFind
   
   } // namespace emst
   } // namespace mlpack
   
   #endif // MLPACK_METHODS_EMST_UNION_FIND_HPP
