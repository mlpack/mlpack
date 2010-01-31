/*
 * These standard declarations for the C library are needed.
 */

#include <stdlib.h>
#include <stdio.h>

/*
 * Include CSDP declarations so that we'll know the calling interfaces.
 */

#include "declarations.h"

/*
 * The main program.  Setup data structures with the problem data, write
 * the problem out in SDPA sparse format, and then solve the problem.
 */

int main()
{
  /*
   * The problem and solution data.
   */

  struct blockmatrix C;
  double *b;
  struct constraintmatrix *constraints;

  /*
   * Storage for the initial and final solutions.
   */

  struct blockmatrix X,Z;
  double *y;
  double pobj,dobj;

  /*
   * blockptr will be used to point to blocks in constraint matrices.
   */

  struct sparseblock *blockptr;

  /*
   * A return code for the call to easy_sdp().
   */

  int ret;

  /*
   * variables for distance constraints
   */

  FILE* distance_constraints_file;
  int n_points;
  int n_labeled_points;
  int n_points_w;
  int n_dims;
  int n_dist_constraints;
  int i, j;
  double sq_dist;
  int** constraint_point_pairs;
  double* constraint_sq_distances;
  int constraint_num;
  int n_constraints;
  int point_num;
  int label;
  double* labels;
  FILE* labels_file;
  
  double lambda_k = 0.001;
  double lambda_s = 1;

  /*
   * Read in the labels.
   */

  labels_file = fopen("../../write_cvxmod_program/2_class_swiss_roll_no_labels.csv", "r");
  fscanf(labels_file, "%d", &n_labeled_points);
  labels = (double*) malloc(n_labeled_points * sizeof(double));
  point_num = 0;
  while(EOF != fscanf(labels_file, "%d", &label)) {
    labels[point_num] = label;
    point_num++;
  }
  fclose(labels_file);

  if(point_num != n_labeled_points) {
    printf("Number of labeled points specified at beginning of labels file does not match number of labels contained in that file!\n");
    exit(1);
  }


  /*
   * Read in the distance constraints.
   */

  distance_constraints_file =
    fopen("../../write_cvxmod_program/upper_triangularized_distance_constraints.csv", "r");

  n_points = 0;
  fscanf(distance_constraints_file, "%d %d", &n_points, &n_dist_constraints);

  n_points_w = n_points + 1;
  n_constraints = n_dist_constraints + n_labeled_points;

  constraint_point_pairs = (int**) malloc(2 * sizeof(int*));
  for(i = 0; i < 2; i++) {
    constraint_point_pairs[i] = (int*) malloc(n_dist_constraints * sizeof(int));
  }
  
  constraint_sq_distances = (double*) malloc(n_dist_constraints * sizeof(double));

  constraint_num = 0;
  while(EOF != fscanf(distance_constraints_file,
		      "%d %d %lf",
		      &i, &j, &sq_dist)) {
    constraint_point_pairs[0][constraint_num] = i;
    constraint_point_pairs[1][constraint_num] = j;
    constraint_sq_distances[constraint_num] = sq_dist;
    constraint_num++;
  }
  fclose(distance_constraints_file);


    
    
  



  /*
   * The first major task is to setup the C matrix and right hand side b.
   */

  /*
   * First, allocate storage for the C matrix.  We have three blocks, but
   * because C starts arrays with index 0, we have to allocate space for
   * four blocks- we'll waste the 0th block.  Notice that we check to 
   * make sure that the malloc succeeded.
   */

  C.nblocks = 2;
  C.blocks =
    (struct blockrec *) malloc((C.nblocks + 1) * sizeof(struct blockrec));
  if(C.blocks == NULL) {
    printf("Couldn't allocate storage for C!\n");
    exit(1);
  }

  /*
   * Setup the first block.  Note that we have to allocate space for 
   * n_points_w + 1 entries as C starts array indexing with 0 rather than 1.
   */
  
  C.blocks[1].blockcategory = MATRIX;
  C.blocks[1].blocksize = n_points_w;
  C.blocks[1].data.mat =
    (double*) malloc(n_points_w * n_points_w * sizeof(double));
  if(C.blocks[1].data.mat == NULL) {
    printf("Couldn't allocate storage for C!\n");
    exit(1);
  }

  /*
   * Put the entries into the first block.
   */
  
  for(i = 1; i <= n_points; i++) {
    for(j = 1; j <= n_points; j++) {
      if(i == j) {
	C.blocks[1].data.mat[ijtok(i, j, n_points_w)] = -lambda_k;
      }
      else {
	C.blocks[1].data.mat[ijtok(i, j, n_points_w)] = 0.0;
      }
    }
  }
  C.blocks[1].data.mat[ijtok(n_points_w, n_points_w, n_points_w)] = -1;

  
  C.blocks[2].blockcategory = DIAG;
  C.blocks[2].blocksize = 2 * n_labeled_points;
  C.blocks[2].data.vec =
    (double*) malloc((2 * n_labeled_points + 1) * sizeof(double));
  if(C.blocks[2].data.vec == NULL) {
    printf("Couldn't allocate storage for C!\n");
    exit(1);
  }
  
  /*
   * Put the entries into the second block.
   */
  
  for(i = 1; i <= n_labeled_points; i++) {
    C.blocks[2].data.vec[i] = -lambda_s;
  }
    

  /*
   * Allocate storage for the right hand side, b.
   */

  b = (double*) malloc((n_constraints + 1) * sizeof(double));
  if (b == NULL) {
    printf("Failed to allocate storage for a!\n");
    exit(1);
  }

  /*
   * Fill in the entries in b.
   */

  for(i = 1; i <= n_dist_constraints; i++) {
    b[i] = constraint_sq_distances[i - 1];
  }

  /*
   * Note: We multiply LHS and RHS of soft-margin constraints by -1
   *       so that we can use nonnegative slack variables for the inequality
   */
  for(i = n_dist_constraints + 1; i <= n_constraints; i++) {
    b[i] = -1;
  }

  /*
   * The next major step is to setup the constraint matrices Ai.
   * Again, because C indexing starts with 0, we have to allocate space for
   * one more constraint.  constraints[0] is not used.
   */
  
  constraints =
    (struct constraintmatrix*)
    malloc((n_constraints + 1) * sizeof(struct constraintmatrix));
  if (constraints == NULL) {
    printf("Failed to allocate storage for constraints!\n");
    exit(1);
  }

  /*
   * Setup the A1 matrix.  Note that we start with block 2 of Ai and then
   * do block 1 of Ai.  We do this in this order because the blocks will
   * be inserted into the linked list of A1 blocks in reverse order.  
   */

  /*
   * Terminate the linked list with a NULL pointer.
   */

  for(constraint_num = 1;
      constraint_num <= n_dist_constraints; constraint_num++) {
    constraints[constraint_num].blocks = NULL;

    blockptr = (struct sparseblock*) malloc(sizeof(struct sparseblock));
    if (blockptr == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }


    blockptr->blocknum = 1;
    blockptr->blocksize = n_points_w;
    blockptr->constraintnum = constraint_num;
    blockptr->next = NULL;
    blockptr->nextbyblock = NULL;
    blockptr->entries = (double*) malloc((3 + 1) * sizeof(double));
    if(blockptr->entries == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }
    blockptr->iindices=(int*) malloc((3 + 1) * sizeof(int));
    if(blockptr->iindices == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }
    blockptr->jindices = (int*) malloc((3 + 1) * sizeof(int));
    if(blockptr->jindices == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }

    /*
     * We have 3 nonzero entries in the upper triangle of block 1 of Ai.
     */

    blockptr->numentries = 3;

    blockptr->iindices[1] = constraint_point_pairs[0][constraint_num - 1] + 1;
    blockptr->jindices[1] = constraint_point_pairs[0][constraint_num - 1] + 1;
    blockptr->entries[1] = 1.0;

    /* this one is mirrored in the lower triangle for -2.0 cumulatively */
    blockptr->iindices[2] = constraint_point_pairs[0][constraint_num - 1] + 1;
    blockptr->jindices[2] = constraint_point_pairs[1][constraint_num - 1] + 1;
    blockptr->entries[2] = -1.0;

    blockptr->iindices[3] = constraint_point_pairs[1][constraint_num - 1] + 1;
    blockptr->jindices[3] = constraint_point_pairs[1][constraint_num - 1] + 1;
    blockptr->entries[3] = 1.0;


    /*
     * Insert block 1 into the linked list of A1 blocks.  
     */

    blockptr->next = constraints[constraint_num].blocks;
    constraints[constraint_num].blocks = blockptr;
    
  }

  /********/

  point_num = 1;

  for(constraint_num = n_dist_constraints + 1;
      constraint_num <= n_constraints; constraint_num++) {

    constraints[constraint_num].blocks = NULL;

    blockptr = (struct sparseblock*) malloc(sizeof(struct sparseblock));
    if (blockptr == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }

    /*
     * Initialize block 2.
     */

    blockptr->blocknum = 2;
    blockptr->blocksize = 2 * n_labeled_points;
    blockptr->constraintnum = constraint_num;
    blockptr->next = NULL;
    blockptr->nextbyblock = NULL;
    blockptr->entries = (double*) malloc((2 + 1) * sizeof(double));
    if(blockptr->entries == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }
    blockptr->iindices=(int*) malloc((2 + 1) * sizeof(int));
    if(blockptr->iindices == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }
    blockptr->jindices = (int*) malloc((2 + 1) * sizeof(int));
    if(blockptr->jindices == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }

    /*
     * We have 2 nonzero entries in this diagonal block of Ai.
     */

    blockptr->numentries = 2;

    blockptr->iindices[1] = point_num;
    blockptr->jindices[1] = point_num;
    blockptr->entries[1] = -1.0;

    blockptr->iindices[2] = n_labeled_points + point_num;
    blockptr->jindices[2] = n_labeled_points + point_num;
    blockptr->entries[2] = 1.0;

    /*
     * Insert block 2 into the linked list of Ai blocks.  
     */
    
    blockptr->next = constraints[constraint_num].blocks;
    constraints[constraint_num].blocks = blockptr;
    
    blockptr=(struct sparseblock*) malloc(sizeof(struct sparseblock));
    if(blockptr == NULL){
	printf("Allocation of constraint block failed!\n");
	exit(1);
      };
    
    /*
     * Initialize block 1.
     */
    
    blockptr->blocknum = 1;
    blockptr->blocksize = n_points_w;
    blockptr->constraintnum = constraint_num;
    blockptr->next = NULL;
    blockptr->nextbyblock = NULL;
    blockptr->entries = (double*) malloc((1 + 1) * sizeof(double));
    if(blockptr->entries == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }
    blockptr->iindices = (int*) malloc((1 + 1) * sizeof(int));
    if(blockptr->iindices == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }
    blockptr->jindices=(int*) malloc((1 + 1) * sizeof(int));
    if(blockptr->jindices == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }
    

    /*
     * We have 1 nonzero entry in the upper triangle of block 1 of A1.
     */
    
    blockptr->numentries=1;
    
    blockptr->iindices[1] = point_num;
    blockptr->jindices[1] = n_points_w;
    blockptr->entries[1] = -1 * labels[point_num - 1];
    
    
    /*
     * Insert block 1 into the linked list of Ai blocks.  
     */
    
    blockptr->next = constraints[constraint_num].blocks;
    constraints[constraint_num].blocks = blockptr;
    
    point_num++;  
  }
  

  

  /*********/



  /*
   * At this point, we have all of the problem data setup.
   */

  /*
   * Write the problem out in SDPA sparse format.
   */

  n_dims = n_points_w + 2 * n_labeled_points;

  write_prob("prob.dat-s", n_dims, n_constraints, C, b, constraints);

  /*
   * Create an initial solution.  This allocates space for X, y, and Z,
   * and sets initial values.
   */

  initsoln(n_dims, n_constraints, C, b, constraints, &X, &y, &Z);

  /*
   * Solve the problem.
   */

  ret = easy_sdp(n_dims, n_constraints, C, b, constraints, 0.0, &X, &y, &Z, &pobj, &dobj);

  if (ret == 0) {
    printf("The objective value is %.7e \n",(dobj + pobj) / 2);
  }
  else {
    printf("SDP failed.\n");
  }

  /*
   * Write out the problem solution.
   */

  write_sol("prob.sol", n_dims, n_constraints, X, y, Z);

  /*
   * Free storage allocated for the problem and return.
   */

  free_prob(n_dims, n_constraints, C, b, constraints, X, y, Z);
  exit(0);
  
}
