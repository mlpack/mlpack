//#include "fl_par_em.h"
#include <unistd.h>
#include <stdio.h>
#include "mpi.h"
//#include <fastlib/fastlib.h>
#include "fl_data_io.h"
#include "fl_kmeans.h"

Dataset alldata;

int 
main(int argc, char* argv[])
{
  int        my_rank, com_size, source, dest;
  int        silen = 128;
  char       hname[128];
  MPI_Status status;
  int        tag = 50;
  int        gherr;
  const char *datafile;
  FILE       *fp;
  int        total_rows, cols, my_startrow, my_numrows;
  Matrix     data;

  char message[800];

  fx_init(argc, argv);
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &com_size);

  gherr = gethostname( hname, silen);

  // Get the config file name
  if (!(datafile = fx_param_str_req(NULL, "data"))) {
    fprintf(stderr, "Error in reading Data File\n");
    exit(1);
  }

  if (fp= fopen(datafile, "r")) {
    fread(&total_rows, sizeof(int), 1, fp);
    fread(&cols, sizeof(int), 1, fp);
  }

  par_split_data(&my_startrow, &my_numrows, total_rows, my_rank, com_size);
  data.Init(cols, my_numrows); // Loading column major data

  if (fp)
    read_subset_bin2matrix(fp, data, my_startrow, my_numrows, cols);
  
  MPI_Barrier(MPI_COMM_WORLD);

  fprintf(stderr, "\nProcessor %d\n", my_rank);
  data.PrintDebug("Data Subset");

  if (!PASSED(alldata.InitFromFile(datafile))) {
    fprintf(stderr, "Couldn't open file '%s'.\n", datafile);
    exit(1);
  }

  MPI_Finalize();
}


double 
parallel_em(Matrix const &data, int num_clusters,                   
		   ArrayList<Vector> *centroids_, 
		   ArrayList<Matrix> *cov_mats_,
		   Vector *log_cltr_wts_,
		   int my_rank, int com_size, int tag,
		   int max_iter, double error_thresh)
{

  ArrayList<Vector> &centroids = *centroids_;

  // Seed the model

  if (my_rank == 0) {
    // FOR NOW: Let the process 0 load all data, run k-means ONCE, and 
    // distribute means, weights and cov_mats to all other processes as seeds

    ArrayList<int> labels;
    //kmeans_CV(alldata, 1, 3, 3);
    kmeans(alldata.matrix(), num_clusters, &labels, &centroids);
  }
  else {
    // The other processes block on receiving the seed from process 0
    //Once the seed is received, compute the gaussian densities and
    // posterior probabilities and use it to compute the log likelihood
    
  } 

  // Compute parallel log-likelihood

  {
    // Re-estimate distributed params

    // Recompute the new log-likelihood and compare
  }

  return 0;
}




  /*
  if (my_rank != 0) {
    sprintf(message, "Greetings from process %d on %s!", my_rank, hname);
    dest = 0;
    MPI_Send(message, strlen (message)+1, MPI_CHAR, dest,
	     tag, MPI_COMM_WORLD);
  }
  else{
    printf ("Messages received by process %d on %s.\n\n", my_rank, hname);
    for (source = 1; source < com_size; source++) {
      MPI_Recv(message, 800, MPI_CHAR, source, tag,
	       MPI_COMM_WORLD, &status);
      printf("%s\n", message);
    }
  }
  */



























   /*  
  Dataset data;


  if (!PASSED(data.InitFromFile(srcname))) {
    fprintf(stderr, "Couldn't open file '%s'.\n", srcname);
    exit(1);
  }

  write_matrix2bin(dstname, data.matrix());

  mat.Init(data.n_features(), data.n_points());

  int start_row, num_rows;

  
  FILE *fp;
  if (fp= fopen(dstname, "r"))
    read_subset_bin2matrix(fp, mat, 50, 101, data.n_features());

  Matrix m;
  la::TransposeInit(mat, &m);

  m.PrintDebug("");
    */
