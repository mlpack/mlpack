#include <fastlib/fastlib.h>
#include <string>
#include <vector>

int main (int argc, char *argv[]) {
 
  fx_module *root
    = fx_init(argc, argv, NULL);

	Matrix data, distances;
	GenMatrix<index_t> ranks;
  std::string file = fx_param_str_req(root, "data");
  std::string ofile = fx_param_str_req(root, "output");
  NOTIFY("Loading files...");
  data::Load(file.c_str(), &data);

  FILE *pfile = fopen(ofile.c_str(), "w");

  NOTIFY("File loaded...");
  NOTIFY("Data has %"LI"d points in %"LI"d dimensions.",
	 data.n_cols(), data.n_rows());

	index_t num_points = data.n_cols();
	
	/** Obtaining the distance matrix **/
	distances.Init(num_points, num_points);
  for (index_t i = 0; i < num_points; i++) {
    Vector p;
    data.MakeColumnVector(i, &p);
		distances.set(i, i, 0.0);
    for (index_t j = i + 1; j < num_points; j++) {
      Vector q;
      data.MakeColumnVector(j, &q);

      double sq_dist = la::DistanceSqEuclidean(p, q);
      double dist = sqrt(sq_dist);
			distances.set(i, j, dist);
			distances.set(j, i, dist);
    }
  }
	
	NOTIFY("DISTANCES COMPUTED!");
	/** Obtaining the ranks **/
	ranks.Init(num_points, num_points);
	for (index_t i = 0; i < num_points; i++) {
		std::vector<std::pair<double, index_t> > neighbors(num_points);
		for (index_t j = 0; j < num_points; j++) {
			neighbors[j] = std::make_pair(distances.get(i,j), j);
		}
		
		std::sort(neighbors.begin(), neighbors.end());
		DEBUG_ASSERT(neighbors[0].second == i);
		ranks.set(i, i, 0);
		for (index_t j = 1; j < num_points; j++) {
			ranks.set(i, neighbors[j].second, j);
		}
	}
	NOTIFY("RANKS COMPUTED!");
	
/*	for (index_t i = 0; i < num_points; i++) {
		for (index_t j = 0; j < num_points; j++)
			if (pfile != NULL)
				fprintf(pfile, "%lg ", distances.get(i,j));
		fprintf(pfile, "\n");
	}
	fprintf(pfile, "\n");
	fprintf(pfile, "\n");
	fprintf(pfile, "\n");*/
	for (index_t i = 0; i < num_points; i++) {
		for (index_t j = 0; j < num_points; j++)
			if (pfile != NULL)
				fprintf(pfile, "%"LI"d,", ranks.get(i,j));
		fprintf(pfile, "\b\n");
	}
	
  fclose(pfile);

  fx_done(fx_root);
}
