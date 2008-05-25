#ifndef INSIDE_MATRIX_FACTORIZED_FMM_IMPL_H
#error "This is not a public header file!"
#endif

class MatrixFactorizedFMMReferenceNodeStat {
 public:
  
  /** @brief The default constructor.
   */
  MatrixFactorizedFMMReferenceNodeStat() {
  }
  
  /** @brief The default destructor.
   */
  ~MatrixFactorizedFMMReferenceNodeStat() {}

  /** @brief Far field expansion created by the reference points in
   *         this node.
   */
  typename TKernelAux::TFarFieldExpansion farfield_expansion_;
  
  void Init(const TKernelAux &ka) {
    farfield_expansion_.Init(ka);
  }
  
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
  
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const MatrixFactorizedFMMReferenceNodeStat& left_stat,
	    const MatrixFactorizedFMMReferenceNodeStat& right_stat) {
  }
      
};

class MatrixFactorizedFMMQueryNodeStat {
 public:

  ////////// Member Variables //////////

  /** @brief The local expansion for the query points in this node.
   */
  typename TKernelAux::TLocalExpansion local_expansion_;

  /** @brief The lower bound on the kernel sum for the query points
   *         owned by this node.
   */
  double mass_l_;

  ////////// Constructor/Destructor //////////

  /** @brief The default constructor.
   */
  MatrixFactorizedFMMQueryNodeStat() {
    mass_l_ = 0;
  }
  
  /** @brief The default destructor.
   */
  ~MatrixFactorizedFMMQueryNodeStat() {}

  ////////// Member Functions //////////

  void Init(const TKernelAux &ka) {
    local_expansion_.Init(ka);
  }
  
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
  
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const MatrixFactorizedFMMQueryNodeStat& left_stat,
	    const MatrixFactorizedFMMQueryNodeStat& right_stat) {
  }  
};

/** @brief The type of our query tree.
 */
typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix, MatrixFactorizedFMMQueryNodeStat > QueryTree;

/** @brief The type of our reference tree.
 */
typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix, MatrixFactorizedFMMReferenceNodeStat > ReferenceTree;
