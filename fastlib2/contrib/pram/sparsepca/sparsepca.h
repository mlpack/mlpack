/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file sparsepca.h
 *
 * Defines a Sparse Principal Components
 * object and calculates the requested number of 
 * sparse loadings for a given data.
 * 
 * This implements the Sparse Principal Component
 * Analysis described in the following paper:
 *  @misc{ zou04sparse,
 *  author = "H. Zou and T. Hastie and R. Tibshirani",
 *  title = "Sparse principal component analysis",
 *  text = "Technical report, statistics department, Stanford University, 2004.",
 *  year = "2004"}
 *
 * This paper use the extension of the elastic 
 * nets described in the following paper:
 *  @article{zou05regularization,
 *  author = "Hui Zou and Trevor Hastie",
 *  title = "Regularization and variable selection via the elastic net",
 *  journal = "Journal Of The Royal Statistical Society Series B",
 *  year = "2005"}
 *
 * The method for solving the resulting Lasso problem 
 * is obtained from the following paper:
 *  @misc{ efron02least,
 *  author = "B. Efron and T. Hastie and I. Johnstone and R. Tibshirani",
 *  title = "Least Angle Regression",
 *  text = "Technical report, Department of Statistics, Stanford University.",
 *  year = "2002"}
 *
 */

#ifndef SPARSEPCA_H
#define SPARSEPCA_H

#include <fastlib/fastlib.h>

/**
 * This is a Sparse Principal Component class
 * 
 * It calculates the sparse loadings of the 
 * requested number of principal components
 * on a given dataset.
 * 
 * 
 * Example use:
 *
 * @code
 * SparsePCA spca;
 * Matrix dataset;
 * double quadratic_regularization_param;
 * ArrayList<double> lasso_regularization_param;
 * datanode *spca_module;
 * ........
 *
 * spca.Init(dataset, quadratic_regularization_param,
 *           lasso_regularization_param, spca_module);
 * spca.Sparsify();
 * 
 * Matrix sparse_loadings;
 * double percent_variance_explained = spca.OutputResults(&sparse_loadings);
 * @endcode
 * 
 * The numerical algorithm used:
 * 1. Let $\alpha$ start at \textbf{V}[,1:$k$], 
 *    the loadings of the first $k$ ordinary principal components
 * 2. Given fixed $\alpha$, solve the following naive elastic net 
 *    problem for $j = 1,2,\ldots,k$
 *    \[ \beta_j = \arg \min_{\beta^*} \(X^TX+\lambda_2\)\beta^* - 
 *                                     2\alpha_j^TX^TX\beta^* +
 *                                     \lambda_{1,j}|\beta^*|_1 \]
 * 3. For each fixed $\beta$, do the SVD of $X^TX\beta = UDV^T$, 
 *    then update $\alpha = UV^T$.
 * 4. Repeat steps 2-3, until $\beta$ converges.
 * 5. Normalization: $\hat{V}_j = \frac{\beta_j}{||\beta_j||}, j = 1,2,\ldots,k.
 *
 */
class SparsePCA {

 private:

  // The data matrix and the data matrix centered
  Matrix x_mat_, x_centered_mat_;

  // The quadratic regularization parameter
  double lambda_quad_;

  // The Lasso regularization parameter for each of the 
  // K principal components
  ArrayList<double> lambda_L1_;

  // The module for the object
  datanode *spca_module_;

  // The total variance of the dataset and 
  // the variance explained by the sparse loadings respectively
  double var_total_, var_explained_;

  // The loading of the standard PCA and the sparse loadings respectively
  Matrix v_mat_, v_sparse_mat_;

 public:

  SparsePCA() {
    x_mat_.Init(0,0);
    x_centered_mat_.Init(0,0);
    lambda_L1_.Init(0);
  }

  ~SparsePCA() {
  }

  void set_x(const Matrix& mat) {
    x_mat_.Destruct();
    x_mat_.Copy(mat);
    return;
  }

  void set_x_centered(Matrix& mat) {
    x_centered_mat_.Destruct();
    x_centered_mat_.Copy(mat);
    return;
  }

  void set_lambda_quad(double val) {
    lambda_quad_ = val;
    return;
  }

  void set_lambda_L1(ArrayList<double>& list) {
    lambda_L1_.Resize(list.size());
    for (index_t i = 0; i < list.size(); i++) {
      lambda_L1_[i] = list[i];
    }
    return;
  }

  void set_var_total(double val) {
    var_total_ = val;
    return;
  }

  void set_var_explained(double val) {
    var_explained_ = val;
    return;
  }

  void set_v(Matrix& mat) {
    v_mat_.Copy(mat);
    return;
  }

  void set_v_sparse(Matrix& mat) {
    v_sparse_mat_.Copy(mat);
    return;
  }

  const Matrix& x() {
    return x_mat_;
  }

  const Matrix& x_centered() {
    return x_centered_mat_;
  }

  double lambda_quad(){
    return lambda_quad_;
  }

  double lambda_L1(index_t in) {
    return lambda_L1_[in];
  }

  double var_total() {
    return var_total_;
  }

  double var_explained() {
    return var_explained_;
  }

  const Matrix& v() {
    return v_mat_;
  }

  const Matrix& v_sparse() {
    return v_sparse_mat_;
  }

  /**
   * This is an internal function which centers 
   * the data so that the row means are zero 
   * as the data is stored in Column major format 
   * in fastlib
   */
  void Center_(const Matrix& mat, Matrix *centered_mat) {

    index_t c = mat.n_cols();
    double factor = 1.0 / c;
    
    if (c < sqrt(BIG_BAD_NUMBER)) {
      // doing fast centering when the matrix size is manageable 
      // by the compiler  
      Matrix summing_mat, mean_mat;
      summing_mat.Init(c, c);
      summing_mat.SetAll(factor);
      la::MulInit(mat, summing_mat, &mean_mat);
      la::SubInit(mean_mat, mat, centered_mat);
  
    }
    else {
      // doing brute centering when the number of datapoints 
      // is very high
      Vector mean_vec, sub_vec;
      mean_vec.Init(c);
      mean_vec.SetAll(factor);
      la::MulInit(mat, mean_vec, &sub_vec);

      centered_mat->Init(mat.n_rows(), mat.n_cols());
 
      for (index_t i = 0; i < c; i++) {
	Vector centered_col_vec, mat_col_vec;
	centered_mat->MakeColumnVector(i, &centered_col_vec);
	mat_col_vec.Copy(mat.GetColumnPtr(i), mat.n_rows());
	la::SubOverwrite(sub_vec, mat_col_vec, &centered_col_vec);
      }

    }
    return;
  }

  /**
   * This function initializes the SparsePCA object
   * with the dataset and the regularization parameters
   * It performs the SVD on the centered data to 
   * obtain the ordinary principal component loadings
   */
  void Init(const Matrix& data, double lambda_2, 
	    ArrayList<double> lambda_1, datanode *spca_module) {

    spca_module_ = spca_module;
    set_x(data);
    set_lambda_quad(lambda_2);
    set_lambda_L1(lambda_1);

    index_t dim = fx_param_int_req(spca_module_, "D");
    index_t n = fx_param_int_req(spca_module_, "N");

    // centering the data matrix to so that 
    // the column means of X are all zero.
    printf("entering center\n");
    fflush(NULL);
    Matrix centered_mat;
    Center_(data, &centered_mat);
    printf("returned from center\n");
    fflush(NULL);

    // if the number of points are much higher than the 
    // dimension of the data, it is more efficient 
    // to use the covariance matrix of the data
    if (n / dim > 100) {

      NOTIFY("More efficient computation by using the covariance matrix\n");

      Matrix temp_covariance_mat, x_trans_mat;
      double factor = 1.0 / n;

      // S = 1/N * X_centered^T * X_centered
      la::TransposeInit(centered_mat, &x_trans_mat);
      la::MulInit(centered_mat, x_trans_mat, &temp_covariance_mat);
      la::Scale(factor, &temp_covariance_mat);

      // S = V * L * V^T
      // L <- (L + abs(L)) / 2
      // S_mod <- V * sqrt(L) * V^T
      Matrix eigenvectors_mat, eigenvectors_t_mat;
      Vector eigenvalues_vec, eigenvalues_abs_vec;

      la::EigenvectorsInit(temp_covariance_mat, &eigenvalues_vec, 
			   &eigenvectors_mat);

      DEBUG_ASSERT(eigenvalues_vec.length() == dim);
      eigenvalues_abs_vec.Init(dim);
      for (index_t i = 0; i < dim; i++) {
	eigenvalues_abs_vec.ptr()[i] = fabs(eigenvalues_vec.get(i));
      }

      la::AddTo(eigenvalues_abs_vec, &eigenvalues_vec);
      la::Scale(0.5, &eigenvalues_vec);

      for (index_t i = 0; i < dim; i++) {
	eigenvalues_vec.ptr()[i] = sqrt(eigenvalues_vec.get(i));
      }

      Matrix temp_x_mat, temp_eigen_mat, temp_prod_mat;

      temp_eigen_mat.Init(dim, dim);
      temp_eigen_mat.SetZero();
      temp_eigen_mat.SetDiagonal(eigenvalues_vec);

      la::TransposeInit(eigenvectors_mat, &eigenvectors_t_mat);

      la::MulInit(eigenvectors_mat, temp_eigen_mat, &temp_prod_mat);
      la::MulInit(temp_prod_mat, eigenvectors_t_mat, &temp_x_mat);

      centered_mat.Destruct();
      la::TransposeInit(temp_x_mat, &centered_mat);
    }

    set_x_centered(centered_mat);
    // performing SVD on the centered matrix to obtain
    // X_centered = U * L * V^T

    Vector s_vec;
    Matrix u_mat, v_t_mat, x_trans_mat;

    la::TransposeInit(x_centered(), &x_trans_mat);
    success_t svd_op = la::SVDInit(x_trans_mat, &s_vec, &u_mat, &v_t_mat);
    DEBUG_ASSERT_MSG(svd_op == SUCCESS_PASS, 
		     "SVD of dataset not successful\n");

    // storing the value of V, the standard PCA loadings
    Matrix v_mat;
    la::TransposeInit(v_t_mat, &v_mat);
    set_v(v_mat);

    // calculating the value of the total variance
    double var = 0;
    for (index_t i = 0; i < dim; i++) {
      double temp = s_vec.get(i);
      var += temp*temp;
    }
    set_var_total(var);

    return;
  }

  /**
   * This function outputs the sparse loadings and 
   * the percent of the variance it explains
   */
  double OutputResults(Matrix *sparse_loadings_mat) {

    sparse_loadings_mat->Copy(v_sparse());
    double percent_var = var_explained() * 100 / var_total();

    return percent_var;
  }

  /**
   * This implements the general algorithm for the 
   * case where dimension >> number of points
   * to obtain the sparse loadings starting from 
   * the ordinary principal component loadings
   */
  void SparsifyMicroArray();

  /**
   * This is the soft thresholding function which 
   * is used to obtain the value of beta as a result 
   * of the minimization. In the case where dim >> n,
   * the quadratic regularization parameter tends to 
   * infinity, causing the minimization problem to 
   * become a soft thresholding function
   */
  void Soft_(Vector&, Vector*, double);

  /**
   * This function implements the algorithm to obtain 
   * the sparse loadings starting from the ordinary
   * principal component loadings
   */
  void Sparsify();

  /** 
   * This function does the minimization step 
   * in every iteration of the algorithm to 
   * obtain the updated $\beta_j$
   */
  void SolveForBeta_(Vector&, Vector*, double, double);

  /**
   * This function updates the upper triangular 
   * matrix every time a new variable is included 
   * in the active set by adding a column and 
   * hence increasing its rank
   */
  void UpdateR_(Vector&, Matrix&, double, index_t*, Matrix*);

  /**
   * This function downdates the same upper triangular 
   * matrix every time a particular variable is 
   * dropped from the active set by removing its 
   * corresponding column, and adjusting the following 
   * columns to maintain the upper triangularity.
   */
  void DowndateR_(Matrix*, index_t);

  /**
   * This function returns the maximum absolute 
   * value present in a matrix
   */
  double MaxAbsValue_(Matrix&);

  /**
   * This function returns the maximum absolute 
   * value present in a vector
   */
  double MaxAbsValue_(Vector&);

  /**
   * This function returns the maximum absolute 
   * value in a vector along with its index
   */
  double MaxAbsValue_(Vector&, index_t*);

  /**
   * This function calculates the number of occurrence of a 
   * particular marker in an array (used for calculating the 
   * cardinality of the active, ignored and inactive sets)
   */
  index_t SubsetLength_(index_t*, index_t, index_t);
    
  /**
   * This function makes a subvector out of a vector
   * corresponding to those indices which match a 
   * particular marker (used for obtaining the 
   * inactive parts of the $\beta$ vector 
   * and the sign vector)
   */
  void MakeSubvector_(Vector&, index_t*, index_t, Vector*);

  /**
   * This function makes a subvector out of a vector
   * corresponding to those indices which are in the 
   * set of indices provided to the function
   * (used for obtaining the active parts of 
   * the $\beta$ vector and the sign vector in the 
   * proper sequence of their entrance in the active set)
   */
  void MakeSubvector_(Vector&, ArrayList<index_t>&, Vector*);

 /**
   * This function makes a submatrix out of a matrix
   * by choosing those columns whose corresponding  
   * indices match the provided marker
   * (used for obtaining the inactive part of 
   * the data matrix)
   */
  void MakeSubmatrix_(Matrix&, index_t*, index_t, Matrix*);

  /**
   * This function makes a submatrix out of a matrix
   * by choosing those columns whose corresponding
   * indices are in the set of indices provided to the function
   * (used for obtaining the active part of 
   * the dataset in the proper sequence 
   * of their entrance in the active set)
   */
  void MakeSubmatrix_(Matrix&, ArrayList<index_t>&, Matrix*);

};

#endif
