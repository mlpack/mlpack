
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_pelleg_moore_kmeans_statistic.hpp:

Program Listing for File pelleg_moore_kmeans_statistic.hpp
==========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_pelleg_moore_kmeans_statistic.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/pelleg_moore_kmeans_statistic.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_STATISTIC_HPP
   #define MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_STATISTIC_HPP
   
   namespace mlpack {
   namespace kmeans {
   
   class PellegMooreKMeansStatistic
   {
    public:
     PellegMooreKMeansStatistic() { }
   
     template<typename TreeType>
     PellegMooreKMeansStatistic(TreeType& node)
     {
       centroid.zeros(node.Dataset().n_rows);
   
       // Hope it's a depth-first build procedure.  Also, this won't work right for
       // trees that have self-children or stuff like that.
       for (size_t i = 0; i < node.NumChildren(); ++i)
       {
         centroid += node.Child(i).NumDescendants() *
             node.Child(i).Stat().Centroid();
       }
   
       for (size_t i = 0; i < node.NumPoints(); ++i)
       {
         centroid += node.Dataset().col(node.Point(i));
       }
   
       if (node.NumDescendants() > 0)
         centroid /= node.NumDescendants();
       else
         centroid.fill(DBL_MAX); // Invalid centroid.  What else can we do?
     }
   
     const arma::uvec& Blacklist() const { return blacklist; }
     arma::uvec& Blacklist() { return blacklist; }
   
     const arma::vec& Centroid() const { return centroid; }
     arma::vec& Centroid() { return centroid; }
   
    private:
     arma::uvec blacklist;
     arma::vec centroid;
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   #endif
