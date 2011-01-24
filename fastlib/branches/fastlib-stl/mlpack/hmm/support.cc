#include "fastlib/fastlib.h"
#include "support.h"

namespace hmm_support {

  double RAND_UNIFORM_01() {
    return (double) rand() / (double) RAND_MAX;
  }

  double RAND_UNIFORM(double a, double b) {
    return RAND_UNIFORM_01() * (b - a) + a;
  }

  void print_matrix(const arma::mat& a, const char* msg) {
    printf("%s - Matrix (%d x %d) = \n", msg, a.n_rows, a.n_cols);
    for (int i = 0; i < a.n_rows; i++) {
      for (int j = 0; j < a.n_cols; j++)
	printf("%8.4f", a(i, j));

      printf("\n");
    }
  }

  void print_matrix(TextWriter& writer, const arma::mat& a, const char* msg, const char* format) {
    writer.Printf("%s - Matrix (%d x %d) = \n", msg, a.n_rows, a.n_cols);
    for (int j = 0; j < a.n_cols; j++) {
      for (int i = 0; i < a.n_rows; i++)
	writer.Printf(format, a(i, j));

      writer.Printf("\n");
    }
  }

  void print_vector(const arma::vec& a, const char* msg) {
    printf("%s - Vector (%d) = \n", msg, a.n_elem);
    for (int i = 0; i < a.n_elem; i++)
      printf("%8.4f", a[i]);
    printf("\n");
  }

  void print_vector(TextWriter& writer, const arma::vec& a, const char* msg, const char* format) {
    writer.Printf("%s - Vector (%d) = \n", msg, a.n_elem);
    for (int i = 0; i < a.n_elem; i++)
      writer.Printf(format, a[i]);
    writer.Printf("\n");
  }

  double RAND_NORMAL_01() {
    double r = 2, u, v;
    while (r > 1) {
      u = RAND_UNIFORM(-1, 1);
      v = RAND_UNIFORM(-1, 1);
      r = (u * u) + (v * v);
    }
    return sqrt(-2 * log(r) / r) * u;
  }

  void RAND_NORMAL_01_INIT(int N, arma::vec& v) {
    double r, u, t;
    v.set_size(N);
    for (int i = 0; i < N; i+=2) {
      r = 2;
      while (r > 1) {
	u = RAND_UNIFORM(-1, 1);
	t = RAND_UNIFORM(-1, 1);
	r = (u * u) + (t * t);
      }
      v[i] = sqrt(-2 * log(r) / r) * u;
      if (i + 1 < N)
        v[i + 1] = sqrt(-2 * log(r) / r) * t;
    }
  }

  void RAND_NORMAL_INIT(const arma::vec& mean, const arma::mat& cov, arma::vec& v) {
    int N = mean.n_elem;
    arma::vec v01;
    RAND_NORMAL_01_INIT(N, v01);
    v = cov * v01;
    v += mean;
  }

  // return x'Ay
  double MyMulExpert(const arma::vec& x, const arma::mat& A, const arma::vec& y) {
    int M = A.n_rows;
    int N = A.n_cols;
    DEBUG_ASSERT_MSG((M == x.n_elem && N == y.n_elem), "MyMulExpert: sizes do not match");

    double s = 0;
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
	s += x[i] * A(i, j) * y[j];

    return s;
  }

  double NORMAL_DENSITY(const arma::vec& x, const arma::vec& mean, const arma::mat& inv_cov, double det_cov) {
    arma::vec d = mean - x;
    return det_cov * exp(-0.5 * MyMulExpert(d, inv_cov, d));
  }

  bool kmeans(const std::vector<arma::mat>& data, int num_clusters, 
	      std::vector<int>& labels_, std::vector<arma::vec>& centroids_, 
	      int max_iter, double error_thresh)
  {
    std::vector<int> counts; //number of points in each cluster
    std::vector<arma::vec> tmp_centroids;
    int num_points, num_dims;
    int i, j, num_iter = 0;
    double error, old_error;

    num_points = 0;
    for (int i = 0; i < data.size(); i++)
      num_points += data[i].n_cols;

    if (num_points < num_clusters) 
      return false;

    num_dims = data[0].n_rows;

    centroids_.reserve(num_clusters);
    tmp_centroids.reserve(num_clusters);

    counts.reserve(num_clusters);
    labels_.reserve(num_points);

    //Initialize the clusters to k points
    for (j = 0; j < num_clusters; j++) {
      int i = (int) math::Random(0, data.size() - 0.5);
      int k = (int) math::Random(0, data[i].n_cols - 0.5);

      arma::vec temp_vector = data[i].unsafe_col(k);
      centroids_[j] = temp_vector;
      tmp_centroids[j].set_size(num_dims);
    }

    error = DBL_MAX;

    do {
      old_error = error; error = 0;

      for (i = 0; i < num_clusters; i++) {
	tmp_centroids[i].zeros();
        counts[i] = 0;
      }
      i = 0;
      for (int t = 0, i = 0; t < data.size(); t++) {
        for (int k = 0; k < data[t].n_cols; k++, i++) {
          // Find the cluster closest to this point and update its label
          double min_distance = DBL_MAX;
          arma::vec data_i_Vec = data[t].unsafe_col(k);

          for (j = 0; j < num_clusters; j++) {
            double distance = la::DistanceSqEuclidean(data_i_Vec, centroids_[j]);

            if (distance < min_distance) {
              labels_[i] = j;
              min_distance = distance;
            }
          }

          // Accumulate the stats for the new centroid of the target cluster
          tmp_centroids[labels_[i]] += data_i_Vec;
          counts[labels_[i]]++;
	  error += min_distance;
        }

        // Now update all the centroids
        for (int j = 0; j < num_clusters; j++) {
          if (counts[j] > 0)
            centroids_[j] = tmp_centroids[j] / (double) counts[j];
        }
        num_iter++;
      }

    } while ((fabs(error - old_error) > error_thresh)
	     && (num_iter < max_iter));

    return true;
  }

  bool kmeans(const arma::mat& data, int num_clusters, 
	      std::vector<int>& labels_, std::vector<arma::vec>& centroids_, 
	      int max_iter, double error_thresh)
  {
    std::vector<int> counts; //number of points in each cluster
    std::vector<arma::vec> tmp_centroids;
    int num_points, num_dims;
    int i, j, num_iter = 0;
    double error, old_error;
    
    if (data.n_cols < num_clusters) 
      return false;
    
    num_points = data.n_cols;
    num_dims = data.n_rows;
    
    centroids_.reserve(num_clusters);
    tmp_centroids.reserve(num_clusters);
    
    counts.reserve(num_clusters);
    labels_.reserve(num_points);
    
    // Initialize the clusters to k points
    for (i = 0, j = 0; j < num_clusters; i += num_points / num_clusters, j++) {
      arma::vec temp_vector = data.unsafe_col(i);
      centroids_[j] = temp_vector;
      tmp_centroids[j].set_size(num_dims);
    }
    
    error = DBL_MAX;
    
    do {
      old_error = error; error = 0;

      for (i = 0; i < num_clusters; i++) {
	tmp_centroids[i].zeros();
	counts[i] = 0;
      }
      
      for (i = 0; i<num_points; i++) {
        // Find the cluster closest to this point and update its label
        double min_distance = DBL_MAX;
        arma::vec data_i_Vec = data.unsafe_col(i);
       
        for (j = 0; j < num_clusters; j++) {
	  double distance = la::DistanceSqEuclidean(data_i_Vec, centroids_[j]);
          if (distance < min_distance) {
	    labels_[i] = j;
	    min_distance = distance;
	  }
        }
       
        // Accumulate the stats for the new centroid of the target cluster
        tmp_centroids[labels_[i]] += data_i_Vec;
        counts[labels_[i]]++;
        error += min_distance;
      }
     
      // Now update all the centroids
      for (int j = 0; j < num_clusters; j++) {
        if (counts[j] > 0)
          centroids_[j] = tmp_centroids[j] / (double) counts[j];
      }
      num_iter++;
    
    } while ((fabs(error - old_error) > error_thresh)
	     && (num_iter < max_iter));
    
    return true;
  }
  
  void mat2arrlst(arma::mat& a, std::vector<arma::vec>& seqs) {
    int n = a.n_cols;
    for (int i = 0; i < n; i++) {
      arma::vec seq = a.unsafe_col(i);
      seqs.push_back(seq);
    }
  }

  void mat2arrlstmat(int N, arma::mat& a, std::vector<arma::mat>& seqs) {
    int n = a.n_cols;
    for (int i = 0; i < n; i+=N) {
      arma::mat b = a.cols(i, i + N);
      seqs.push_back(b);
    }
  }

  bool skip_blank(TextLineReader& reader) {
    for (;;) {
      if (!reader.MoreLines())
        return false;
//      char* pos = reader.Peek().begin();

      std::string::iterator pos = reader.Peek().begin();
      while (*pos == ' ' || *pos == ',' || *pos == '\t')
	pos++;

      if (*pos == '\0' || *pos == '%')
        reader.Gobble();
      else
        break;
    }
    return true;
  }

  success_t read_matrix(TextLineReader& reader, arma::mat& matrix) {
    if (!skip_blank(reader)) { // EOF ?
      matrix.set_size(0, 0);
      return SUCCESS_FAIL;
    } else {
      int n_rows = 0;
      int n_cols = 0;
      bool is_done;
     
      // How many columns ?
      std::vector<std::string> num_str;
      tokenizeString(reader.Peek(), ", \t", num_str);
      n_cols = num_str.size();
     
      std::vector<double> num_double;

      for(;;) { // read each rows
        n_rows++;
        // deprecated: double* point = num_double.AddBack(n_cols);
        std::vector<std::string> num_str;
        tokenizeString(reader.Peek(), ", \t", num_str);
//       reader.Peek().Split(", \t", &num_str);

        DEBUG_ASSERT(num_str.size() == n_cols);

        std::istringstream is;
        for (int i = 0; i < n_cols; i++) {
          double d;
          is.str(num_str[i]);
          if( !(is >> d ) )
            abort();
          num_double.push_back(d);
        }

        is_done = false;

        reader.Gobble();

        for (;;) {
	  if (!reader.MoreLines()) {
	    is_done = true;
	    break;
	  }
          std::string::iterator pos = reader.Peek().begin();
	 
          while (*pos == ' ' || *pos == '\t')
	    pos++;
	 
          if (*pos == '\0')
            reader.Gobble();
	  else if (*pos == '%') {
	    is_done = true;
	    break;
	  }
	  else
            break;
        }

        if (is_done) {
          std::vector<double> empty;
          num_double.swap(empty);
         
          //FIXME
//	  matrix->Own(num_double.ReleasePtr(), n_cols, n_rows);
	 
          return SUCCESS_PASS;
        }
      }
    }
  }

  success_t read_vector(TextLineReader& reader, arma::vec& vec) {
    if (!skip_blank(reader)) { // EOF ?
      vec.set_size(0);
      return SUCCESS_FAIL;
    }
    
    std::vector<double> num_double;

    for(;;) { // read each rows
      bool is_done = false;

      std::vector<std::string> num_str;
      tokenizeString(reader.Peek(), ", \t", num_str);

      // deprecated: double* point = num_double.AddBack(num_str.size());
      
      std::istringstream is;
      for (int i = 0; i < num_str.size(); i++) {
        double d;
        is.str(num_str[i]);
        if( !(is >> d) )
          abort();
      }

      reader.Gobble();

      for (;;) {
        if (!reader.MoreLines()) {
	  is_done = true;
	  break;
	}
        std::string::iterator pos = reader.Peek().begin();
	while (*pos == ' ' || *pos == '\t')
	  pos++;
	if (*pos == '\0')
          reader.Gobble();
	else if (*pos == '%') {
	  is_done = true;
	  break;
	} else
          break;
      }

      if (is_done) {
        //num_double.Trim();
	int length = num_double.size();

        // FIXME
//	vec->Own(num_double.ReleasePtr(), length);
	
        return SUCCESS_PASS;
      }
    }
  }

  success_t load_matrix_list(const char* filename, std::vector<arma::mat>& matlst) {
    TextLineReader reader;
    if (!PASSED(reader.Open(filename)))
      return SUCCESS_FAIL;
    do {
      arma::mat tmp;
      if (read_matrix(reader, tmp) == SUCCESS_PASS)
	matlst.push_back(tmp);
      else
        break;
    } while (1);

    return SUCCESS_PASS;
  }

  success_t load_vector_list(const char* filename, std::vector<arma::vec>& veclst) {
    TextLineReader reader;
    if (!PASSED(reader.Open(filename)))
      return SUCCESS_FAIL;

    do {
      arma::vec vec;
      if (read_vector(reader, vec) == SUCCESS_PASS)
	veclst.push_back(vec);
      else
        break;
    } while (1);

    return SUCCESS_PASS;
  }
} // end namespace
