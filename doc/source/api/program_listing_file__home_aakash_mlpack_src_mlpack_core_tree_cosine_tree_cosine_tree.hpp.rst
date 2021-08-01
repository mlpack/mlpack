
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_cosine_tree_cosine_tree.hpp:

Program Listing for File cosine_tree.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_cosine_tree_cosine_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/cosine_tree/cosine_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_HPP
   #define MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <boost/heap/priority_queue.hpp>
   
   namespace mlpack {
   namespace tree {
   
   // Predeclare classes for CosineNodeQueue typedef.
   class CompareCosineNode;
   class CosineTree;
   
   // CosineNodeQueue typedef.
   typedef boost::heap::priority_queue<CosineTree*,
       boost::heap::compare<CompareCosineNode> > CosineNodeQueue;
   
   class CosineTree
   {
    public:
     CosineTree(const arma::mat& dataset);
   
     CosineTree(CosineTree& parentNode, const std::vector<size_t>& subIndices);
   
     CosineTree(const arma::mat& dataset,
                const double epsilon,
                const double delta);
   
     CosineTree(const CosineTree& other);
   
     CosineTree(CosineTree&& other);
   
     CosineTree& operator=(const CosineTree& other);
   
     CosineTree& operator=(CosineTree&& other);
   
     ~CosineTree();
   
     void ModifiedGramSchmidt(CosineNodeQueue& treeQueue,
                              arma::vec& centroid,
                              arma::vec& newBasisVector,
                              arma::vec* addBasisVector = NULL);
   
     double MonteCarloError(CosineTree* node,
                            CosineNodeQueue& treeQueue,
                            arma::vec* addBasisVector1 = NULL,
                            arma::vec* addBasisVector2 = NULL);
   
     void ConstructBasis(CosineNodeQueue& treeQueue);
   
     void CosineNodeSplit();
   
     void ColumnSamplesLS(std::vector<size_t>& sampledIndices,
                          arma::vec& probabilities, size_t numSamples);
   
     size_t ColumnSampleLS();
   
     size_t BinarySearch(arma::vec& cDistribution, double value, size_t start,
                         size_t end);
   
     void CalculateCosines(arma::vec& cosines);
   
     void CalculateCentroid();
   
     void GetFinalBasis(arma::mat& finalBasis) { finalBasis = basis; }
   
     const arma::mat& GetDataset() const { return *dataset; }
   
     std::vector<size_t>& VectorIndices() { return indices; }
   
     void L2Error(const double error) { this->l2Error = error; }
     double L2Error() const { return l2Error; }
   
     arma::vec& Centroid() { return centroid; }
   
     void BasisVector(arma::vec& bVector) { this->basisVector = bVector; }
   
     arma::vec& BasisVector() { return basisVector; }
   
     CosineTree* Parent() const { return parent; }
     CosineTree*& Parent() { return parent; }
   
     CosineTree* Left() const { return left; }
     CosineTree*& Left() { return left; }
   
     CosineTree* Right() const { return right; }
     CosineTree*& Right() { return right; }
   
     size_t NumColumns() const { return numColumns; }
   
     double FrobNormSquared() const { return frobNormSquared; }
   
     size_t SplitPointIndex() const { return indices[splitPointIndex]; }
   
    private:
     const arma::mat* dataset;
     double delta;
     arma::mat basis;
     CosineTree* parent;
     CosineTree* left;
     CosineTree* right;
     std::vector<size_t> indices;
     arma::vec l2NormsSquared;
     arma::vec centroid;
     arma::vec basisVector;
     size_t splitPointIndex;
     size_t numColumns;
     double l2Error;
     double frobNormSquared;
     bool localDataset;
   };
   
   class CompareCosineNode
   {
    public:
     // Comparison function for construction of priority queue.
     bool operator() (const CosineTree* a, const CosineTree* b) const
     {
       return a->L2Error() < b->L2Error();
     }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
