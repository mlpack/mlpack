#include <fastlib/fastlib.h>
#include <string>
#include <vector>

int main (int argc, char *argv[]) {
 
  fx_module *root
    = fx_init(argc, argv, NULL);

  Matrix Q, R;
  //GenMatrix<size_t> ranks;
  std::string rfile = fx_param_str_req(root, "R");
  std::string qfile = fx_param_str_req(root, "Q");
  std::string ofile = fx_param_str_req(root, "output");
  NOTIFY("Loading files...");
  data::Load(rfile.c_str(), &R);
  data::Load(qfile.c_str(), &Q);

  FILE *pfile = fopen(ofile.c_str(), "w");

  NOTIFY("File loaded...");
  NOTIFY("R (%zud X %zud) ; Q (%zud X %zud)",
	 R.n_cols(), R.n_rows(), Q.n_cols(), Q.n_rows());

  size_t num_queries = Q.n_cols();
  size_t num_refs = R.n_cols();
	
  /** Obtaining the distance matrix **/
  for (size_t i = 0; i < num_queries; i++) {
    Vector q;
    Q.MakeColumnVector(i, &q);
    //dists.Init(num_refs);

    std::vector<std::pair<double, size_t> > neighbors(num_refs);
    for (size_t j = 0; j < num_refs; j++) {
      Vector r;
      R.MakeColumnVector(j, &r);

      double sq_dist = la::DistanceSqEuclidean(q, r);
      neighbors[j] = std::make_pair(sq_dist, j);

      //dists[j] = sq_dist;
    }

    std::sort(neighbors.begin(), neighbors.end());
    GenVector<size_t> ranks;
    ranks.Init(num_refs);
    for (size_t j = 0; j < num_refs; j++) {
      ranks[neighbors[j].second] = j+1;
    }

    for (size_t j = 0; j < num_refs; j++) {
      if (pfile != NULL) {
	if (j == 0) 
	  fprintf(pfile, "%zud", ranks[j]);
	else
	  fprintf(pfile, ",%zud", ranks[j]);
      }
    }
    if (pfile != NULL)
      fprintf(pfile, "\n");
  }
	
  fclose(pfile);

  fx_done(fx_root);
}
