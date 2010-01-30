/*
   An example showing how to call the easy_sdp() interface to CSDP.  In
   this example, we solve the problem
 
      max tr(C*X)
          tr(A1*X)=1
          tr(A2*X)=2
          X >= 0       (X is PSD)
 
   where 
 
    C=[2 1
       1 2
           3 0 1
           0 2 0
           1 0 3
                 0
                   0]

   A1=[3 1
       1 3
           0 0 0
           0 0 0
           0 0 0
                 1
                   0] 

   A2=[0 0
       0 0
           3 0 1
           0 4 0
           1 0 5
                 0
                   1] 

  Notice that all of the matrices have block diagonal structure.  The first
  block is of size 2x2.  The second block is of size 3x3.  The third block
  is a diagonal block of size 2.  

 */

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
  int n_dist_constraints;
  int i, j, counter;
  double sq_dist;
  int** constraint_point_pairs;
  double* constraint_sq_distances;
  int constraint_num;
  int n_upper_triangle;
  int n_constraints;



  /*
   * Read in the distance constraints.
   */

  distance_constraints_file =
    fopen("upper_triangularized_distance_constraints.csv", "r");

  n_points = 0;
  fscanf(distance_constraints_file, "%d %d", &n_points, &n_dist_constraints);
  
  n_constraints = n_dist_constraints + 1;

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

  C.nblocks = 1;
  C.blocks=(struct blockrec *)malloc((1 + 1) * sizeof(struct blockrec));
  if (C.blocks == NULL) {
    printf("Couldn't allocate storage for C!\n");
    exit(1);
  }

  /*
   * Setup the first block.  Note that we have to allocate space for 
   * n_points + 1 entries because C starts array indexing with 0 rather than 1.
   */
  
  C.blocks[1].blockcategory = MATRIX;
  C.blocks[1].blocksize = n_points;
  C.blocks[1].data.mat =
    (double*) malloc(n_points * n_points * sizeof(double));
  if (C.blocks[1].data.mat == NULL) {
    printf("Couldn't allocate storage for C!\n");
    exit(1);
  }

  /*
   * Put the entries into the first block.
   */
  
  for(i = 1; i <= n_points; i++) {
    for(j = 1; j <= n_points; j++) {
      if(i == j) {
	C.blocks[1].data.mat[ijtok(i, j, n_points)] = 1.0;
      }
      else {
	C.blocks[1].data.mat[ijtok(i, j, n_points)] = 0.0;
      }
    }
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

  for(i = 0; i < n_dist_constraints; i++) {
    b[i + 1] = constraint_sq_distances[i];
  }
  b[i + 1] = 0;

  /*
   * The next major step is to setup the constraint matrices Ai.
   * Again, because C indexing starts with 0, we have to allocate space for
   * one more constraint.  constraints[0] is not used.
   */
  
  constraints = (struct constraintmatrix*) malloc((n_constraints + 1) * sizeof(struct constraintmatrix));
  if (constraints == NULL) {
    printf("Failed to allocate storage for constraints!\n");
    exit(1);
  }

  /*
   * Setup the A1 matrix.  Note that we start with block 3 of A1 and then
   * do block 1 of A1.  We do this in this order because the blocks will
   * be inserted into the linked list of A1 blocks in reverse order.  
   */

  /*
   * Terminate the linked list with a NULL pointer.
   */

  for(constraint_num = 1; constraint_num <= n_dist_constraints; constraint_num++) {
    constraints[constraint_num].blocks = NULL;

    blockptr = (struct sparseblock*) malloc(sizeof(struct sparseblock));
    if (blockptr == NULL) {
      printf("Allocation of constraint block failed!\n");
      exit(1);
    }


    blockptr->blocknum = 1;
    blockptr->blocksize = n_points;
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
     * We have 3 nonzero entries in the upper triangle of block 1 of A1.
     */

    blockptr->numentries = 3;

    /*
     * The entry in the 1,1 position of block 1 of A1 is 3.0
     */

    blockptr->iindices[1] = constraint_point_pairs[0][constraint_num - 1] + 1;
    blockptr->jindices[1] = constraint_point_pairs[0][constraint_num - 1] + 1;
    blockptr->entries[1] = 1.0;

    /*
     * The entry in the 1,2 position of block 1 of A1 is 1.0
     */

    blockptr->iindices[2] = constraint_point_pairs[0][constraint_num - 1] + 1;
    blockptr->jindices[2] = constraint_point_pairs[1][constraint_num - 1] + 1;
    blockptr->entries[2] = -1.0; /* should be mirrored to provide -2.0 cumulatively */

    /*
     * The entry in the 2,2 position of block 1 of A1 is 3.0
     */

    blockptr->iindices[3] = constraint_point_pairs[1][constraint_num - 1] + 1;
    blockptr->jindices[3] = constraint_point_pairs[1][constraint_num - 1] + 1;
    blockptr->entries[3] = 1.0;

    /*
     * Note that we don't have to store the 2,1 entry- this is assumed to be
     * equal to the 1,2 entry.
     */

    /*
     * Insert block 1 into the linked list of A1 blocks.  
     */

    blockptr->next = constraints[constraint_num].blocks; /* equivalent to 'set to NULL' */
    constraints[constraint_num].blocks = blockptr;
    
  }

  /********/
  
  constraints[constraint_num].blocks = NULL;
  
  blockptr = (struct sparseblock*) malloc(sizeof(struct sparseblock));
  if (blockptr == NULL) {
    printf("Allocation of constraint block failed!\n");
    exit(1);
  }

  n_upper_triangle= n_points + ((n_points * (n_points - 1)) / 2);


  blockptr->blocknum = 1;
  blockptr->blocksize = n_points;
  blockptr->constraintnum = constraint_num;
  blockptr->next = NULL;
  blockptr->nextbyblock = NULL;
  blockptr->entries = (double*) malloc((n_upper_triangle + 1) * sizeof(double));
  if(blockptr->entries == NULL) {
    printf("Allocation of constraint block failed!\n");
    exit(1);
  }
  blockptr->iindices=(int*) malloc((n_upper_triangle + 1) * sizeof(int));
  if(blockptr->iindices == NULL) {
    printf("Allocation of constraint block failed!\n");
    exit(1);
  }
  blockptr->jindices = (int*) malloc((n_upper_triangle + 1) * sizeof(int));
  if(blockptr->jindices == NULL) {
    printf("Allocation of constraint block failed!\n");
    exit(1);
  }

  blockptr->numentries = n_upper_triangle;

  counter = 1;
  for(i = 1; i <= n_points; i++) {
    for(j = i; j <= n_points; j++) {
      blockptr->iindices[counter] = i;
      blockptr->jindices[counter] = j;
      blockptr->entries[counter] = 1.0;
      counter++;
    }
  }
  
  
  blockptr->next = NULL;
  constraints[constraint_num].blocks = blockptr;

  /*********/



  /*
   * At this point, we have all of the problem data setup.
   */

  /*
   * Write the problem out in SDPA sparse format.
   */



  write_prob("prob.dat-s", n_points, n_constraints, C, b, constraints);

  /*
   * Create an initial solution.  This allocates space for X, y, and Z,
   * and sets initial values.
   */

  initsoln(n_points, n_constraints, C, b, constraints, &X, &y, &Z);

  /*
   * Solve the problem.
   */

  ret = easy_sdp(n_points, n_constraints, C, b, constraints, 0.0, &X, &y, &Z, &pobj, &dobj);

  if (ret == 0) {
    printf("The objective value is %.7e \n",(dobj + pobj) / 2);
  }
  else {
    printf("SDP failed.\n");
  }

  /*
   * Write out the problem solution.
   */

  write_sol("prob.sol", n_points, n_constraints, X, y, Z);

  /*
   * Free storage allocated for the problem and return.
   */

  free_prob(n_points, n_constraints, C, b, constraints, X, y, Z);
  exit(0);
  
}
