#ifndef SVM_LIB_H
#define SVM_LIB_H

#include <fastlib/fastlib.h>

namespace SVMLib {
  /** Kernel function definition */
  typedef double (*KernelFunc)(const Vector& x, const Vector& y);

  class KernelFunction {
    double d; // for polynomial kernel
    double sigma2; // for rbf kernel = -0.5/sigma^2
    KernelFunc kfunc;
  public:
    enum TYPE { LINEAR, POLYNOMIAL, RBF, CUSTOM } type;

    KernelFunction(TYPE type_ = LINEAR, double param = 1);

    KernelFunction(KernelFunc kfunc_);

    double operator() (const Vector& x, const Vector& y);
  };  

  /** SMO options */
  struct SMOOptions {
    enum VERBOSE {NONE = 0, ITER, FINAL} verbose;
    double tolKKT; // default 1e-6
    double KKTViolationLevel; // default 1e-3
  };

  /* Implement a set of indices, O(1) read, write */
  class IndexSet {
    index_t n_total, n_set; // number of points
    ArrayList<index_t> index_set; // index subset, size n_set
    ArrayList<index_t> set_index; // index of points in the index_set, size n_total
  public:
    IndexSet(index_t n);
    IndexSet(const IndexSet& ind) {
      n_total = ind.n_total; n_set = ind.n_set;
      index_set.InitCopy(ind.index_set); set_index.InitCopy(ind.set_index);
    }
    void operator=(const IndexSet& ind) {
      this->n_total = ind.n_total;
      this->n_set = ind.n_set;
      this->index_set.Renew(); this->index_set.InitCopy(ind.index_set);
      this->set_index.Renew(); this->set_index.InitCopy(ind.set_index);
    }

    void InitEmpty();

    void addremove(index_t i, bool b = true);

    index_t get_n() const { return n_set; }
    index_t get_n_total() const { return n_total; }
    const ArrayList<index_t>& get_index() const { return index_set; }
    bool is_set(index_t i) const { 
      DEBUG_ASSERT(i < n_total);
      return set_index[i] != -1; 
    }
    int operator[] (index_t i) const {
      DEBUG_ASSERT(i < n_set);
      return index_set[i];
    }
    void print();

    void print(const Vector& x);
  };

  class Kernel {
    int n_points;
    bool kernel_stored;
    const Matrix& X_;
    KernelFunction kfunc_;

    Vector kernel_diag;

    Matrix full_kernel;

    Matrix sub_kernel;
    ArrayList<index_t> col_index;
    ArrayList<index_t> lru_col;
    index_t n_cols, lru_ptr;
    index_t n_loads;
  public:
    Kernel(const KernelFunction& kfunc, const Matrix& X);

    void get_element(index_t i, index_t j, 
		     double& Kii, double& Kjj, double& Kij);

    double get_element(index_t i, index_t j) {
      if (kernel_stored)
	return full_kernel.get(i, j);
      else
	if (col_index[i] >= 0)
	  return sub_kernel.get(j, col_index[i]);
	else if (col_index[j] >= 0)
	  return sub_kernel.get(i, col_index[j]);
	else 
	  return sub_kernel.get(i, loadColumn(j));
    }

    /** Get a kernel matrix column, col_i must be uninitialized */
    void get_column(index_t i, Vector* col_i);
    
    /** Get the diagonal of the kernel matrix, diag must be uninitialized*/
    void get_diag(Vector* diag);

    /** Number of load columns */
    index_t n_load() { return n_loads; }
    index_t n_point() { return n_points; }
  private:
    double kernel_call(index_t i, index_t j);
    index_t loadColumn(index_t i);
  };

  /** SMO main function
   *  Solve l1-SVM with box constraints
   *  min 1/2 alpha^T K alpha - alpha^T 1
   *  s.t 0 \leq alpha \leq box
   *      alpha^T y = 0
   */
  int seqminopt(const Matrix& X, const Vector& y, const Vector& box,
		Kernel& kernel, SMOOptions options,
		Vector& alpha, IndexSet& SVindex, double& offset);

  double svm_output_on_sample(index_t i, Kernel& kernel, 
			      const Vector& y, const Vector& alpha, 
			      const IndexSet& SVindex, double offset);

  index_t svm_total_error(Kernel& kernel, 
			  const Vector& y, const Vector& alpha, 
			  const IndexSet& SVindex, double offset);

  int ptswarmopt(const Matrix& X, const Vector& y, const Vector& box,
		 Kernel& kernel, /* PSOOptions options, */
		 Vector& alpha, IndexSet& SVs, double& offset);

  /** Utility function for data processing */
  void SplitDataToXy(const Matrix& data, Matrix& X, Vector& y);

  void ScaleData(Matrix& X, Vector& shift, Vector& scale);

  void SetBoxConstraint(const Vector& y, double C, Vector& box);

};

#endif
