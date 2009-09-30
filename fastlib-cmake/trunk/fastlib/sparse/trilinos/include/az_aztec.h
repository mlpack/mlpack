/*@HEADER
// ***********************************************************************
// 
//        AztecOO: An Object-Oriented Aztec Linear Solver Package 
//                 Copyright (2002) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ***********************************************************************
//@HEADER
*/

/*====================================================================
 * ------------------------
 * | CVS File Information |
 * ------------------------
 *
 * $RCSfile: az_aztec.h,v $
 *
 * $Author: william $
 *
 * $Date: 2006/11/13 18:05:47 $
 *
 * $Revision: 1.24 $
 *
 * $Name: trilinos-release-8-0-branch $
 *====================================================================*/

/*
 * Include file for inclusion in any routine which will call the solver
 * library. Contains necessary constants and prototypes.
 *
 * Author:  Scott A. Hutchinson, SNL
 *          John  N. Shadid,     SNL
 *          Ray   S. Tuminaro,   SNL
 */
#ifndef __AZTECH__

/* Set variable to indicate that this file has already been included */

#define __AZTECH__

#include "az_aztec_defs.h"
#include "az_f77func.h"
/* Some older codes use AZ_MPI to set MPI mode for AztecOO/Aztec.
 * Check to see if AZ_MPI is defined, and define AZTEC_MPI if
 * it is not already defined.
 */

#if defined(AZ_MPI) && !defined(AZTEC_MPI)
#define AZTEC_MPI
#endif

/* Force AZTEC_MPI to be defined if ML_MPI is defined */

#if defined(ML_MPI) && !defined(AZTEC_MPI)
#define AZTEC_MPI
#endif

/* The definition of MPI_AZRequest and MPI_AZComm depend on
 * whether or not we are using MPI.
 * NOTE:  This technique can cause problems if az_aztec.h (this file)
 *        is included with AZTEC_MPI undefined in some files and with
 *        AZTEC_MPI defined in other files.  Therefore, users must
 *        make sure that AZTEC_MPI is either defined or undefined for
 *        all files that are compiled and including az_aztec.h.
 */

#ifdef AZTEC_MPI
#include <mpi.h>
#define MPI_AZRequest MPI_Request
#define MPI_AZComm    MPI_Comm
#else
#define MPI_AZRequest int
#define MPI_AZComm    int
#endif


/*structure definitions*/


struct AZ_MATRIX_STRUCT {

  /* Used to represent matrices. In particular, two structures are  */
  /* passed into AZ_iterate:                                      */
  /*    AZ_iterate(..., AZ_MATRIX *Amat, AZ_MATRIX *Precond, ...) */
  /* corresponding to matrix-vector products and preconditioners. */
  /*                                                              */
  /* For matrix-vector products, a subroutine Amat.'user_function'*/
  /* can be supplied. 'Amat' is be passed to this routine and thus*/
  /* relevant data can be placed in this structure. If a matrix-  */
  /* vector product is not supplied, either an MSR or VBR matrix  */
  /* must be used by specifying the arrays bindx,indx,rpntr,cpntr,*/
  /* bpntr, and val. In this case, Aztec supplies the matrix-     */
  /* vector product.                                              */
  /*                                                              */
  /* NOTE: Fortran users never explicitly see this structure but  */
  /* instead pass matrix-vector product and preconditioning       */
  /* information through parameters which Aztec copies to this    */
  /* structure.                                                   */

  int              matrix_type;  /* Indicates whether the matrix is MSR,   */
  /* VBR, or user-supplied.                 */
  /*                                        */
  int              N_local,      /* Number of local and ghost unknowns     */
    N_ghost;      /*                                        */
  /*                                        */
  int           mat_create_called;/* =1 indicates that AZ_matrix_create()   */
                                  /* was invoked.                           */
  int          must_free_data_org;/* =1 indicates that data_org was created */
                                  /* via the matrix_free set functions and  */
                                  /* needs to be freed during destroy oper. */
                                  /*                                        */
  int              *rpntr,*cpntr,/* arrays to support MSR & VBR formats    */
    *bpntr,*bindx;/*                                        */
  int              *indx;        /*                                        */
  double *val;                   /*                                        */
  int              *data_org;    /* array to support matvec communication  */
  /*                                        */
  /* Begin Aztec 2.1 mheroux mod */
  int              N_update;     /* Number of nodes updated on this proc   */
  /*                                        */
  int              *update;      /* array containing global indices map    */
  /*                                        */
  int       has_global_indices;  /* true/false for say bindx has global    */
  /*                                        */
  /* End Aztec 2.1 mheroux mod */
  void (*matvec)(double *,       /* function ptr to user-defined routine   */
                 double *, struct AZ_MATRIX_STRUCT *, int *);
  double (*matnorminf)(struct AZ_MATRIX_STRUCT *); /* function ptr to user-routine   */
  /*********************************************************************/
  /*********************************************************************/
  int (*getrow)(int columns[], double values[], int row_lengths[],
                struct AZ_MATRIX_STRUCT *Amat, int N_requested,
                int requested_rows[], int allocated_space);
  /* function ptr to user-defined routine   */
  /* Get some matrix rows ( requested_rows[0 ... N_requested_rows-1] ) */
  /* from the user's matrix and return this information  in            */
  /* 'row_lengths, columns, values'.  If there is not enough space to  */
  /* complete this operation, return 0.  Otherwise, return 1.          */
  /*                                                                   */
  /* Parameters                                                        */
  /* ==========                                                        */
  /* data             On input, points to user's data containing       */
  /*                  matrix values.                                   */
  /* N_requested_rows On input, number of rows for which nonzero are   */
  /*                  to be returned.                                  */
  /* requested_rows   On input, requested_rows[0...N_requested_rows-1] */
  /*                  give the row indices of the rows for which       */
  /*                  nonzero values are returned.                     */
  /* row_lengths      On output, row_lengths[i] is the number of       */
  /*                  nonzeros in the row 'requested_rows[i]'          */
  /*                  ( 0 <= i < N_requested_rows). NOTE: this         */
  /*                  array is of size 'N_requested_rows'.             */
  /* columns,values   On output, columns[k] and values[k] contains the */
  /*                  column number and value of a matrix nonzero where*/
  /*                  all the nonzeros for requested_rows[0] appear    */
  /*                  first followed by the nonzeros for               */
  /*                  requested_rows[1], etc. NOTE: these arrays are   */
  /*                  of size 'allocated_space'.                       */
  /* allocated_space  On input, indicates the space available in       */
  /*                  'columns' and 'values' for storing nonzeros. If  */
  /*                  more space is needed, return 0.                  */
  /*********************************************************************/
  /*********************************************************************/
  int  (*user_comm)(double *, struct AZ_MATRIX_STRUCT *);
  /* user communication routine before */
  /* doing matvecs. Only used when doing    */
  /* matrix-free.                           */

  double matrix_norm;            /* norm of the matrix A used in the case  */
  /* of least square preconditioning if the */
  /* matrix A is of type AZ_USER_MATRIX */
  /*                                        */
  /*                                        */
  int              **aux_ival;   /* integer, double precision, function,   */
  double           **aux_dval;   /* generic, and matrix pointers at the    */
  void              *aux_ptr;    /* product routine: 'matvec()'.           */
  void              *matvec_data;
  void              *getrow_data;
  struct AZ_MATRIX_STRUCT
  **aux_matrix;
  int              N_nz, max_per_row, /* Total number of nonzeros, maximum */
    largest_band;      /* nonzeros per row, and bandwidth.  */
  /* ONLY used for matrix-free         */

  struct AZ_CONVERGE_STRUCT * conv_info;
  char *print_string; /* Description of problem */
};

struct grid_level {
  int                     N;
  struct AZ_MATRIX_STRUCT *transfer_to_prev_grid;
  struct AZ_MATRIX_STRUCT *transfer_to_next_grid;
  struct AZ_MATRIX_STRUCT *discretization_op;
  struct AZ_PREC_STRUCT *smoother1;
  struct AZ_PREC_STRUCT *smoother2;
  void                    *mesh;
};

struct AZ_PREC_STRUCT {

  /* Used to represent preconditioners. In particular,            */
  /* two structures  are                                          */
  /* passed into AZ_iterate:                                      */
  /* AZ_iterate(..., AZ_MATRIX *Amat, AZ_PRECOND *Precond, ...) */
  /* corresponding to matrix and preconditioner descriptions. */
  /*                                                              */
  /* For matrix-vector products, a subroutine Amat.'matvec'*/
  /* can be supplied. 'Amat' is be passed to this routine and thus*/
  /* relevant data can be placed in this structure. If a matrix-  */
  /* vector product is not supplied, either an MSR or VBR matrix  */
  /* must be used by specifying the arrays bindx,indx,rpntr,cpntr,*/
  /* bpntr, and val. In this case, Aztec supplies the matrix-     */
  /* vector product as well as a number of preconditioners.       */
  /*                                                              */
  /* Likewise, a preconditioner can be supplied via the routine   */
  /* 'Precond.prec_function'. In this case options[AZ_precond]    */
  /* must be set to "AZ_user_precond". Otherwise                  */
  /* options[AZ_precond] must be set to one of the preconditioners*/
  /* supplied by Aztec and the matrix must be a MSR or VBR format */
  /* The matrix used as preconditionner is descibed in a AZ_MATRIX*/
  /* structure which could be either the same as Amat             */
  /* (precond.Pmat = Amat) or  a different matrix described       */
  /* by the arrays bindx,indx,rpntr,cpntr, bpntr, and val.        */
  /*                                                              */
  /* NOTE: Fortran users never explicitly see these structures but*/
  /* instead pass matrix and preconditioning  information through */
  /* parameters which Aztec copies to this  structure.            */
  /*                                                              */

  struct AZ_MATRIX_STRUCT *Pmat;     /* matrix used by the precondtioner  */
  /* when not using multilevel stuff   */
  /*                                   */
  int           prec_create_called;/* =1 indicates that AZ_precond_create() */
  /* was invoked.                           */

  void    (*prec_function)(double *, /* function ptr to user-defined      */
                           int *, int *, double *,/* preconditioning routine           */
                           struct AZ_MATRIX_STRUCT  *,
                           struct AZ_PREC_STRUCT *);

  int                   *options;    /* used to determine preconditioner  */
  double                *params;     /* when options[AZ_precond] is set   */
  struct AZ_PREC_STRUCT *next_prec;  /* to AZ_multilevel. The series of   */
  /* preconditioners is done in a      */
  /* multiplicative fashion.           */
  struct context        *context;
  struct grid_level     grid_levels[10]; /* multilevel stuff                 */
  void *ml_ptr;         /* MLDIFF */
  double timing[2];     /* preconditioner timing array */
  void *precond_data;
  char *print_string;
};


typedef struct AZ_MATRIX_STRUCT AZ_MATRIX;
typedef struct AZ_PREC_STRUCT   AZ_PRECOND;

struct AZ_CONVERGE_STRUCT {
  double r0_norm, A_norm, b_norm;
  int    total_N;
  int    not_initialized;
  struct AZ_SCALING *scaling;
  double epsilon;
  int isnan;
  int converged;
  int iteration;
  int print_info;
  int sol_updated;
  void * res_vec_object; /* This points to an already constructed Epetra_Vector used to pass the residual vector */
  void * conv_object; /* This will contain the pointer to the AztecOO_StatusTest object */
  void (*conv_function)(void * conv_test_obj,/* pointer to AztecOO_StatusTest object */
			void * res_vector_obj, /* pointer to Epetra_Vector that will hold res_vector */
                        int iteration,       /* current iteration */
                        double * res_vector, /* current natural residual vector */
                        int print_info,      /* no info print if 0, else  print info */
                        int sol_updated,      /* solution not updated if = 0, else it is
                                                 and is consistent with res_vector */
                        int * converged,     /* = 0 on return if not converged, otherwise converged */
                        int * isnan,         /* = 0 on return if not NaN, otherwise NaNs detected */
                        double * rnorm,     /* = current norm on return */
			int * r_avail);     /* If set to AZ_TRUE on return, the residual vector is needed
					       by this convergence on subsequent calls and it should be 
					       supplied by the calling routine */
};






struct aztec_choices {
  int *options;
  double *params;
};
struct context {                       /* This structure is used to  */
  int      *iu, *iflag, *ha, *ipvt;   /* hold variables specific to */
  int      *dblock, *space_holder;    /* the preconditioner */
  int      extra_fact_nz_per_row;
  int      N_large_int_arrays, N_large_dbl_arrays;
  int      N_nz_factors,N_nz_matrix, N_blk_rows, max_row;
  double   *pivot;
  struct   AZ_MATRIX_STRUCT     *A_overlapped;
  struct   aztec_choices        *aztec_choices;
  double   *x_pad, *ext_vals, *x_reord;
  int      *padded_data_org, *map, *ordering, *inv_ordering;
  int      N, N_unpadded, N_nz, N_nz_allocated;
  char     *tag;
  int      *proc_config;
  int      Pmat_computed;                 /* indicates that the has    */
  /* been called at least once */
  /* before with this context. */
  /* Begin Aztec 2.1 mheroux mod */
  void    *precon;
  /* End Aztec 2.1 mheroux mod */
};


/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

struct AZ_SCALING {  /* Left and right matrices to scale    */
                     /* the problem                         */
  int    action;
  double A_norm;
  int    mat_name;
  int    scaling_opt;

  /* Define a function pointer that can be called to perform
     scalings for user-defined matrices that can't be scaled
     by Aztec's AZ_scale_f function.
  */
  int (*scale)(int action,
               AZ_MATRIX* Amat,
               int options[],
               double b[],
               double x[],
               int proc_config[],
               struct AZ_SCALING* scaling);

  /* A void pointer that can be used to store data for later reuse.
  */
  void* scaling_data;
};

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

struct grid {                      /* used to define a grid. Still under */
                                   /* construction                       */
  int    *element_vertex_lists;
  int    *Nvertices_per_element;
  int    Nelements;
  int    Nvertices;
  double *vertices;
};

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/


/* Aztec's previous AZ_solve() is renamed to AZ_oldsolve() with 3 new  */
/* parameters appended to it: Amat, precond, scaling.                  */
/* This routine is never called directly by an application. It is only */
/* used internally by Aztec.                                           */

#ifdef __cplusplus
extern "C" {
#endif
  extern void AZ_oldsolve(double x[], double b[], int options[], double params[],
                          double status[], int proc_config[], AZ_MATRIX *Amat,
                          AZ_PRECOND *precond, struct AZ_SCALING *scaling);
#ifdef __cplusplus
}
#endif




/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/


/* This is the new Aztec interface. This routine calls AZ_oldsolve() passing */
/* in for example Amat->indx for the indx[] parameter in AZ_oldsolve().      */
/*                                                                           */
/* NOTE: User's can still invoke AZ_solve() in the old Aztec way. AZ_solve   */
/*       also calls AZ_oldsolve(). However, matrix-free and coarse grid      */
/*       capabilities are not available via AZ_solve().                      */

#ifdef __cplusplus
extern "C" {
#endif
  extern void AZ_iterate(double x[], double b[], int options[], double params[],
                         double status[], int proc_config[],
                         AZ_MATRIX *Amat, AZ_PRECOND *precond, struct AZ_SCALING *scaling);
#ifdef __cplusplus
}
#endif

#ifdef next_release
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

/* This is the new fortran interface to Aztec. There is a wrapper so that  */
/* fortran user's can invoke AZ_iterate (like 'C' users), however the      */
/* parameters are different as the information that is captured in the     */
/* structures Amat, precond, and scaling must now be passed as parameters. */
/*                                                                         */
/* Note: Multilevel stuff is currently not provided to Fortran users.      */

extern void AZ_fortransolve(double x[], double b[], int options[],
                            double params[], int data_org[], double status[], int proc_config[],

                            int indx[],  int bindx[],       /* VBR & MSR arrays. Note: all of these   */
                            int rpntr[], int cpntr[],       /* arrays are passed to 'user_Avec' and   */
                            int bpntr[], double val[],      /* to 'user_precond' if supplied.         */
                            /*                                        */
                            AZ_FUNCTION_PTR user_Avec,      /* user's matrix-free matvec              */
                            /* If doing matrix-free, the following    */
                            /* arrays and functions are passed to the */
                            /* user's matvec:                         */
                            int    A_ival0[],       int    A_ival1[],
                            int    A_ival2[],       int    A_ival3[],
                            double A_dval0[],       double A_dval1[],
                            double A_dval2[],       double A_dval3[],
                            AZ_FUNCTION_PTR A_fun0, AZ_FUNCTION_PTR A_fun1,
                            AZ_FUNCTION_PTR A_fun2, AZ_FUNCTION_PTR A_fun3,

                            AZ_FUNCTION_PTR user_precond,   /* user's preconditioning routine         */
                            /*                                        */
                            /* The following arrays and functions are */
                            /* passed to the user's preconditioner:   */
                            int    M_ival0[],       int    M_ival1[],
                            int    M_ival2[],       int    M_ival3[],
                            double M_dval0[],       double M_dval1[],
                            double M_dval2[],       double M_dval3[],
                            AZ_FUNCTION_PTR M_fun0, AZ_FUNCTION_PTR M_fun1,
                            AZ_FUNCTION_PTR M_fun2, AZ_FUNCTION_PTR M_fun3,

                            /* Vectors used to scale the problem.     */

                            double left_scale[], double right_scale[]);
#endif

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************

 The C user's matrix-vector product must look like this:

   'Amat.user_function'(double x[], double b[], int options[], double params[],
                        AZ_MATRIX *Amat, int proc_config[])

 where on output b = A * x.  The user can put what he wants in 'Amat'
 when he calls AZ_iterate() so that he can use it inside this function.

 The C user's preconditioner must look like this:

   'Prec.user_precond'(double x[], int options[], double params[],
                       AZ_MATRIX *Prec, int proc_config[])

 where on output x = M * x.  The user can put what he wants in 'Prec'
 when he calls AZ_iterate() so that he can use it inside this function.



 The Fortran user's matrix-vector product must look like this:

     'user_matvec'(x, b, options, params, data_org, proc_config,
       indx, bindx, rpntr, cpntr, bpntr,  val,
       A_ival0, A_ivals1, A_ival2, A_ival3,
       A_dval0, A_dvals1, A_dval2, A_dval3,
       A_fun0,  A_fun1,   A_fun2,  A_fun3)

  The user can put what he wants in the integer arrays (A_ival*),
  double precision arrays (A_dval*) and functions (A_fun*) when invoking
  Aztec so that he can use them inside this function.

  NOTE: Additionally, if not using MSR or VBR matrices, the arrays
  indx,bindx,rpntr,cpntr,bpntr, and val can also be used to pass information.


 The Fortran user's preconditioner must look like this:

     'user_precond'(x, options, params, data_org, proc_config,
       indx, bindx, rpntr, cpntr, bpntr,  val,
       M_ival0, M_ivals1, M_ival2, M_ival3,
       M_dval0, M_dvals1, M_dval2, M_dval3,
       M_fun0,  M_fun1,   M_fun2,  M_fun3)

  The user can put what he wants in the integer arrays (M_ival*),
  double precision arrays (M_dval*) and functions (M_fun*) when invoking
  Aztec so that he can use them inside this function.

  NOTE: Additionally, if not using MSR or VBR matrices, the arrays
  indx,bindx,rpntr,cpntr,bpntr, and val can also be used to pass information.

*/


/* Finally, some crude code that defines a function mymatvec() and
   myprecond() each of which use the function myfun() as well as the
   vbr arrays to generate a result.

   C:
   Amat.bindx         = bindx;
   Amat.indx          = indx;
   Amat.rpntr         = rpntr;
   Amat.cpntr         = cpntr;
   Amat.bpntr         = bpntr;
   Amat.val           = val;
   Amat.user_function = mymatvec;
   Amat.aux_funs      = (AZ_FUNCTION_PTR *) calloc(1,sizeof(AZ_FUNCTION_PTR));
   Amat.aux_funs[0]   = myfun;

   Prec.bindx         = bindx;
   Prec.indx          = indx;
   Prec.rpntr         = rpntr;
   Prec.cpntr         = cpntr;
   Prec.bpntr         = bpntr;
   Prec.val           = val;
   Prec.user_function = myprecond;
   Prec.aux_funs      = (AZ_FUNCTION_PTR *) calloc(1,sizeof(AZ_FUNCTION_PTR));
   Prec.aux_funs[0]   = myfun;

   AZ_iterate(x, ax, options, params, status, proc_config,
   &Amat, &Prec, &Scaling);

   Fortran:

   call AZ_iterate(x,b, options, params, data_org, status,
   $           proc_config, NULL,bindx,NULL,NULL,NULL, val,
   $           mymatvec,  NULL, NULL, NULL, NULL, NULL, NULL,
   $           NULL, NULL, myfun, NULL, NULL, NULL,
   $           myprecond, NULL, NULL, NULL, NULL, NULL, NULL,
   $           NULL, NULL, myfun, NULL, NULL, NULL,
   $           NULL, NULL)

*/
/* constants */


/* function definitions */

#ifndef AZ_MAX
#define AZ_MAX(x,y) (( (x) > (y) ) ?  (x) : (y))     /* max function  */
#endif
#ifndef AZ_MIN
#define AZ_MIN(x,y) (( (x) < (y) ) ?  (x) : (y))     /* min function  */
#endif
#ifndef AZ_SGN
#define AZ_SGN(x)   (( (x) < 0.0 ) ? -1.0 : 1.0)  /* sign function */
#endif

/*
 * There are different conventions for external names for fortran subroutines.
 * In addition, different compilers return differing caluse for a fortran
 * subroutine call. In this section we take these into account.
 */

#   define AZ_FNROOT_F77                 F77_FUNC_(az_fnroot,AZ_FNROOT)
#   define MC64AD_F77                    F77_FUNC(mc64ad,MC64AD)
#   define AZ_RCM_F77                    F77_FUNC_(az_rcm,AZ_RCM)
#   define AZ_BROADCAST_F77              F77_FUNC_(az_broadcast,AZ_BROADCAST)
#   define AZ_CHECK_INPUT_F77            F77_FUNC_(az_check_input,AZ_CHECK_INPUT)
#   define AZ_CHECK_MSR_F77              F77_FUNC_(az_check_msr,AZ_CHECK_MSR)
#   define AZ_CHECK_VBR_F77              F77_FUNC_(az_check_vbr,AZ_CHECK_VBR)
#   define AZ_DEFAULTS_F77               F77_FUNC_(az_defaults,AZ_DEFAULTS)
#   define AZ_EXCHANGE_BDRY_F77          F77_FUNC_(az_exchange_bdry,AZ_EXCHANGE_BDRY)
#   define AZ_FIND_INDEX_F77             F77_FUNC_(az_find_index,AZ_FIND_INDEX)
#   define AZ_FIND_LOCAL_INDICES_F77     F77_FUNC_(az_find_local_indices,AZ_FIND_LOCAL_INDICES)
#   define AZ_FIND_PROCS_FOR_EXTERNS_F77 F77_FUNC_(az_find_procs_for_externs,AZ_FIND_PROCS_FOR_EXTERNS)
#   define AZ_FREE_MEMORY_F77            F77_FUNC_(az_free_memory,AZ_FREE_MEMORY)
#   define AZ_GAVG_DOUBLE_F77            F77_FUNC_(az_gavg_double,AZ_GAVG_DOUBLE)
#   define AZ_GDOT_F77                   F77_FUNC_(az_gdot,AZ_GDOT)
#   define AZ_GMAX_DOUBLE_F77            F77_FUNC_(az_gmax_double,AZ_GMAX_DOUBLE)
#   define AZ_GMAX_INT_F77               F77_FUNC_(az_gmax_int,AZ_GMAX_INT)
#   define AZ_GMAX_MATRIX_NORM_F77       F77_FUNC_(az_gmax_matrix_norm,AZ_GMAX_MATRIX_NORM)
#   define AZ_GMAX_VEC_F77               F77_FUNC_(az_gmax_vec,AZ_GMAX_VEC)
#   define AZ_GMIN_DOUBLE_F77            F77_FUNC_(az_gmin_double,AZ_GMIN_DOUBLE)
#   define AZ_GMIN_INT_F77               F77_FUNC_(az_gmin_int,AZ_GMIN_INT)
#   define AZ_GSUM_DOUBLE_F77            F77_FUNC_(az_gsum_double,AZ_GSUM_DOUBLE)
#   define AZ_GSUM_INT_F77               F77_FUNC_(az_gsum_int,AZ_GSUM_INT)
#   define AZ_GSUM_VEC_F77               F77_FUNC_(az_gsum_vec,AZ_GSUM_VEC)
#   define AZ_GVECTOR_NORM_F77           F77_FUNC_(az_gvector_norm,AZ_GVECTOR_NORM)
#   define AZ_INIT_QUICK_FIND_F77        F77_FUNC_(az_init_quick_find,AZ_INIT_QUICK_FIND)
#   define AZ_INVORDER_VEC_F77           F77_FUNC_(az_invorder_vec,AZ_INVORDER_VEC)
#   define AZ_MATVEC_MULT_F77            F77_FUNC_(az_matvec_mult,AZ_MATVEC_MULT)
#   define AZ_VBR_MATVEC_MULT_F77        F77_FUNC_(az_VBR_matvec_mult,AZ_VBR_MATVEC_MULT)
#   define AZ_MSR_MATVEC_MULT_F77        F77_FUNC_(az_MSR_matvec_mult,AZ_MSR_MATVEC_MULT)
#   define AZ_MSR2VBR_F77                F77_FUNC_(az_msr2vbr,AZ_MSR2VBR)
#   define AZ_ORDER_ELE_F77              F77_FUNC_(az_order_ele,AZ_ORDER_ELE)
#   define AZ_PR_ERROR_F77               F77_FUNC_(az_pr_error,AZ_PR_ERROR)
#   define AZ_PRINT_OUT_F77              F77_FUNC_(az_print_out,AZ_PRINT_OUT)
#   define AZ_PROCESSOR_INFO_F77         F77_FUNC_(az_processor_info,AZ_PROCESSOR_INFO)
#   define AZ_QUICK_FIND_F77             F77_FUNC_(az_quick_find,AZ_QUICK_FIND)
#   define AZ_READ_MSR_MATRIX_F77        F77_FUNC_(az_read_msr_matrix,AZ_READ_MSR_MATRIX)
#   define AZ_READ_UPDATE_F77            F77_FUNC_(az_read_update,AZ_READ_UPDATE)
#   define AZ_REORDER_MATRIX_F77         F77_FUNC_(az_reorder_matrix,AZ_REORDER_MATRIX)
#   define AZ_REORDER_VEC_F77            F77_FUNC_(az_reorder_vec,AZ_REORDER_VEC)
#   define AZ_GET_COMM_F77               F77_FUNC_(az_get_comm,AZ_GET_COMM)
#   define AZ_SET_COMM_F77               F77_FUNC_(az_set_comm,AZ_SET_COMM)
#   define AZ_SET_MESSAGE_INFO_F77       F77_FUNC_(az_set_message_info,AZ_SET_MESSAGE_INFO)
#   define AZ_SET_PROC_CONFIG_F77        F77_FUNC_(az_set_proc_config,AZ_SET_PROC_CONFIG)
#   define AZ_SORT_F77                   F77_FUNC_(az_sort,AZ_SORT)
#   define AZ_SOLVE_F77                  F77_FUNC_(az_solve,AZ_SOLVE)
#   define AZ_TRANSFORM_F77              F77_FUNC_(az_transform,AZ_TRANSFORM)

#if defined(CRAY_T3X)

#define AZ_DLASWP_F77  F77_FUNC_(az_slaswp,AZ_SLASWP)
#define AZ_DLAIC1_F77  F77_FUNC_(az_slaic1,AZ_SLAIC1)

#else

#define AZ_DLASWP_F77  F77_FUNC_(az_dlaswp,AZ_DLASWP)
#define AZ_DLAIC1_F77  F77_FUNC_(az_dlaic1,AZ_DLAIC1)

#endif

#ifndef FSUB_TYPE
#define  FSUB_TYPE void
#endif

#ifdef __cplusplus
#include <stdio.h>
extern "C" {
#endif

void PREFIX AZ_DLASWP_F77(int *, double *, int *, int *, int *, int *, int *);

void PREFIX AZ_DLAIC1_F77(int * , int *, double *, double *, double *, double *,
			  double *, double *, double *);
void PREFIX AZ_SLASWP_F77(int *, float *, int *, int *, int *, int *, int *);

void PREFIX AZ_SLAIC1_F77(int * , int *, float *, float *, float *, float *,
			  float *, float *, float *);

  /* Aztec function prototypes that can be called by the user */

  extern void AZ_solve(
                       double x[],     /* On input 'x' contains the initial guess. On output*/
                       /* 'x' contains the solution to our linear system.   */
                       /* NOTE: THis vector must be of size >= N + NExt     */
                       double b[],     /* right hand side of linear system.                 */
                       /* NOTE: This vector must be of size >= N            */
                       int options[],
                       double params[],
                       int indx[],     /* The ith element of indx points to the location in */
                       /* val of the (0,0) entry of the ith block entry. The*/
                       /* last element is the number of nonzero entries of  */
                       /* matrix A plus one.                                */
                       int bindx[],    /* Contains the block column indices of the non-zero */
                       /* block entries.                                    */
                       int rpntr[],    /* The ith element of rpntr indicates the first point*/
                       /* row in the ith block row. The last element is the */
                       /* number of block rows plus one.                    */
                       int cpntr[],    /* The jth element of cpntr indicates the first point*/
                       /* column in the jth block column. The last element  */
                       /* is the number of block columns plus one.          */
                       int bpntr[],    /* The ith element of bpntr points to the first block*/
                       /* entry of the ith row in bindx. The last element is*/
                       /* the number of nonzero blocks of matrix A plus one.*/
                       double val[],   /* matrix A in sparse format (VBR)  .                */
                       /* Indicates current level of factorization          */
                       /* factor_flag =                                     */
                       /*      1: indicates first call to precond. routine  */
                       /*      that performs some type of factorization     */
                       /*      preprocessing such as an incomplete LU.      */
                       /*                                                   */
                       /*      2: use preprocessing info. from a previous   */
                       /*      call. Implies some further change in the     */
                       /*      the numerical entries rather than the sparse */
                       /*      pattern.                                     */
                       /*                                                   */
                       /*      3: use precondtioner from last level 1 or 2  */
                       /*      call to precond. (see specific precondioner  */
                       /*      for more info)                               */
                       int data_org[], double status[], int proc_config[]);

  extern int AZ_initialize(double x[], double b[], int options[],
                           double params[], double status[], int proc_config[], AZ_MATRIX *Amat,
                           AZ_PRECOND *precond, int save_old_values[], struct AZ_SCALING *);

  extern void AZ_finalize(double x[], double b[], int options[], int
                          proc_config[], AZ_MATRIX *Amat, AZ_PRECOND *precond, int save_old_values[],
                          struct AZ_SCALING *scaling);

  extern void AZ_iterate_setup(int options[], double params[], int proc_config[],
                               AZ_MATRIX *Amat, AZ_PRECOND *precond);

  extern void AZ_iterate_finish(int options[], AZ_MATRIX *Amat,
                                AZ_PRECOND *precond);

  extern int AZ_oldsolve_setup(double x[], double b[], int options[],
                               double params[], double status[], int proc_config[], AZ_MATRIX *Amat,
                               AZ_PRECOND *precond, int save_old_values[], struct AZ_SCALING *);

  extern void AZ_oldsolve_finish(double x[], double b[], int options[],
                                 int proc_config[], AZ_MATRIX *Amat, int save_old_values[],
                                 struct AZ_SCALING *);

  extern void AZ_abs_matvec_mult (double *b, double *c,AZ_MATRIX *Amat,int proc_config[]);

  extern void   AZ_add_new_ele(int cnptr[], int col, int blk_row, int bindx[],
                               int bnptr[], int indx[], double val[], int therow,
                               double new_ele, int maxcols, int blk_space,
                               int nz_space, int blk_type);

  extern void   AZ_add_new_row(int therow, int *nz_ptr, int *current, double
                               **val, int **bindx, char *input,FILE *dfp,
                               int *msr_len, int *column0);

  extern int AZ_adjust_N_nz_to_fit_memory(int N, int , int);

  extern char *AZ_allocate(unsigned int iii);

  extern char *AZ_allocate_or_free(void *ptr, unsigned int size, int action);

  extern void   AZ_backsolve(double newa[], double pivot[], double x[], int snr[],
                             int ha[], int iflag[], int *ifail, int *nn, int *n, int *iha);

  extern void AZ_block_diagonal_scaling(int action, AZ_MATRIX *Amat, double val[],
                                        int indx[], int bindx[], int rpntr[], int cpntr[], int bpntr[],
                                        int data_org[], double b[], int options[], int proc_config[],
                                        struct AZ_SCALING *scaling);


  extern int    AZ_breakdown_f(int N, double v[], double w[], double inner,
                               int proc_config[]);

  extern void AZ_broadcast(char *ptr, int length, int proc_config[], int action);

  extern unsigned int AZ_broadcast_info(char buffer[], int proc_config[],
                                        unsigned int length);

  extern void   AZ_calc_blk_diag_inv(double *val, int *indx, int *bindx,
                                     int *rpntr, int *cpntr, int *bpntr,
                                     double *d_inv, int *d_indx, int *d_bindx,
                                     int *d_rpntr, int *d_bpntr, int data_org[]);

  extern void AZ_calc_blk_diag_LU(double *val, int *indx, int *bindx, int *rpntr,
                                  int *cpntr, int *bpntr, double *d_inv, int *d_indx,
                                  int *d_bindx, int *d_rpntr, int *d_bpntr,
                                  int *data_org, int *ipvt);

  extern double AZ_calc_iter_flops(int solver_flag, double inner_flops,
                                   double daxpy_flops, double matvec_flops,
                                   int total_its, double gnnz, double K);

  extern double AZ_calc_precond_flops(int solver_flag, int options[],
                                      double daxpy_flops, double matvec_flops,
                                      int total_its, int gn, double gnnz,
                                      int data_org[], int proc_config[]);

  extern double AZ_calc_solve_flops(int options[], int, double , int , double,
                                    int data_org[], int proc_config[]);

  extern void   AZ_change_it(int indx[], int length, int *first, int *total,
                             int b[]);

  extern void   AZ_change_sign(double *lambda_max, double val[], int indx[],
                               int bindx[], int rpntr[], int cpntr[], int bpntr[],
                               int data_org[]);
  extern void AZ_check_block_sizes(int bindx[], int cpntr[], int Nrows,
                                   int *new_block);

  extern int  AZ_check_input(int data_org[], int options[], double params[],
                             int proc_config[]);

  extern void AZ_check_msr(int *bindx, int N_update, int N_external,
                           int option, int *proc_config);

  extern int    AZ_check_options(int * , int ,int data_org[], int,double *,
                                 AZ_MATRIX *, AZ_PRECOND *);

  extern void AZ_check_vbr(int N_update, int N_external, int option,
                           int bindx[], int bnptr[], int cnptr[], int rnptr[],
                           int proc_config[]);

  extern void AZ_combine_overlapped_values(int sym_flag, int data_org[],
                                           int options[], double x[], int map[], double ext_vals[], int name,
                                           int proc_config[]);

  extern int AZ_compare_update_vs_soln(int N, double, double alpha, double p[],
                                       double x[],
                                       double update_reduction, int ouput_flag, int proc_config[], int *first_time);

  extern int AZ_compress_msr(int *ibindx[], double *ival[], int allocated,
                             int needed, int name, struct context *context);

  extern void   AZ_compute_global_scalars(AZ_MATRIX *Amat,
                                          double x[], double b[], double r[],
                                          double w[], double *r_norm,
                                          double *scaled_r_norm, int option_i[],
                                          int data_org[], int proc_config[],
                                          int *use_r, double v1[], double v2[],
                                          double *value,
                                          struct AZ_CONVERGE_STRUCT *);

  extern void AZ_compute_matrix_size(AZ_MATRIX *Amat, int options[],
                                     int N_nz_unpadded, int N_unpadded, int *N_nz_padded, int N_external,
                                     int *max_row, int *N, int *N_nz, double fill,int *extra_fact_nz_per_row,
                                     int Nb_unpadded, int *bandwidth);

  extern int AZ_compute_max_nz_per_row(AZ_MATRIX *Amat, int N, int Nb,
                                       int *largest_band);


  extern void   AZ_compute_residual( double b[], double u[], double r[],
                                     int proc_config[], AZ_MATRIX *);

  extern void   AZ_convert_ptrs_to_values(int array[], int length);

  extern void   AZ_convert_values_to_ptrs(int array[], int length, int start);

  extern struct AZ_CONVERGE_STRUCT *AZ_converge_create(void);
  extern void AZ_converge_destroy(struct AZ_CONVERGE_STRUCT **temp);

  extern AZ_MATRIX *AZ_create_matrix(int local, int additional, int matrix_type,
                                     int local_blks, int *not_using);

  extern void AZ_defaults(int options[], double params[]);

  extern void AZ_delete_matrix(AZ_MATRIX *ptr);

  extern void   AZ_dgemv2(int m, int n, double *a, double *x, double *y);

  extern void   AZ_dgemv3(int m, int n, double *a, double *x, double *y);

  extern void   AZ_direct_sort(int b[], int indx[], char buffer[], char a[],
                               int *start, int buf_len, int *ind_index,
                               int *the_first, int *real_lists, int *pre_mid);

  extern void   AZ_divide_block(int i, int j, double val[], int indx[],
                                int bindx[], int cpntr[], double *z,
                                double *blockj, double *blocki, int *ipvt);

  extern void   AZ_divide_block0(int i, int j, double val[], int indx[],
                                 int bindx[], int cpntr[], int *ipvt);

  extern void AZ_domain_decomp(double x[], AZ_MATRIX *Amat, int options[],
                               int proc_config[], double params[],
                               struct context *context);

  extern void   AZ_dtrans(int *, int *, double *);

  extern void AZ_equil_scaling(int action, AZ_MATRIX *Amat,
                               double b[],
                               double x[], int options[],
                               int proc_config[], struct AZ_SCALING *scaling);

  extern void AZ_exchange_bdry(double x[], int data_org[], int proc_config[]);

  extern void   AZ_exchange_local_info(int N_neighbors, int proc_num_neighbor[],
                                       char *message_send_add[],
                                       unsigned int message_send_length[],
                                       char *message_recv_add[],
                                       unsigned int message_recv_length[],
                                       int type, int proc_config[]);

  extern int AZ_exit(int input);

  extern int AZ_extract_comm_info(int **idata_org, int (*user_comm)(double *,
                                                                    AZ_MATRIX *), AZ_MATRIX *,
                                  int proc_config[], int N_cols, int Nghost);

  extern void AZ_fact_bilu(int new_blks, AZ_MATRIX *A_overlapped,
                           int *diag_block, int *pivot);

  extern void AZ_fact_chol(int bindx[], double val[], int N,
                           double rthresh, double athresh);

  extern void  AZ_fact_ilut( int *, AZ_MATRIX *, double *a, int *ja,
                             double drop, int extra_fact_nz_per_row, int shift,
                             int *iu, double *cr, double *unorm, int *ind,
                             int *nz_used, int *jnz,
                             double rthresh, double athresh);

  extern void   AZ_fact_lu(double x[], AZ_MATRIX *A_overlapped, double *aflag, double *pivot,
                           int *rnr, int *ha, int *iflag, int *, int*, int *, int *, int *);

  extern void AZ_fact_rilu(int N, int *nz_used, int *iu, int *iw,
                           AZ_MATRIX *A_overlapped, double omega,
                           double rthresh, double athresh);

  extern void AZ_factor_subdomain(struct context *context, int N,
                                  int N_nz, int *nz_used);

  extern int    AZ_fill_sparsity_pattern(struct context *context, int ifill,
                                         int bindx[], double val[], int N);

  extern int    AZ_find_block_col(int cnptr[], int column, int maxcols,
                                  int blk_type);

  extern int    AZ_find_block_in_row(int bindx[], int bnptr[], int i, int blk_col,
                                     int indx[], int, double val[], int blk_space,
                                     int nz_space);

  extern void AZ_find_MSR_ordering(int bindx2[],int **ordering,int N,
                                   int **inv_ordering, int name, struct context *);

  extern int    AZ_find_closest_not_larger(int key, int list[], int length);

  extern int  AZ_find_index(int key, int list[], int length);

  extern void AZ_find_local_indices(int N_update, int bindx[], int update[],
                                    int **external, int *N_external, int mat_type,
                                    int bpntr[]);

  extern void AZ_find_procs_for_externs(int N_update, int update[],
                                        int external[], int N_external,
                                        int proc_config[], int **extern_proc);

  extern int AZ_find_simple(int, int *, int, int *, int, int *, int *);

  extern void AZ_find_global_ordering(int proc_config[], AZ_MATRIX *Amat,
                                      int **global_bindx, int **update);

  extern void AZ_revert_to_global(int proc_config[], AZ_MATRIX *Amat,
                                  int **global_bindx, int **update);


  extern void AZ_fix_pt(double *, double *, double *, int *, double * , int * ,
                        double *, AZ_MATRIX *, AZ_PRECOND *, struct AZ_CONVERGE_STRUCT *);

  extern void AZ_flop_rates(int data_org[],int indx[],int bpntr[], int bindx[],
                            int options[], double status[], double total_time,
                            int proc_config[]);

  extern void AZ_free(void *ptr);

  extern void AZ_free_memory(int label);

  extern void AZ_free_space_holder(struct context *variables);

  extern void   AZ_gappend(int vals[], int *cur_length, int total_length,
                           int proc_config[]);

  extern double AZ_gavg_double(double var, int proc_config[]);

  extern double AZ_gdot(int N, double r[], double z[], int proc_config[]);

  extern void   AZ_gdot_vec(int N, double dots[], double dots2[],
                            int proc_config[]);

  extern int    AZ_get_block(int j, int k, int bindx[], int bpntr[], int *ptr_j);

  extern unsigned int AZ_get_sol_param_size(void);

  extern int    AZ_get_new_eps(double *epsilon, double, double, int options[],
                               int proc_config[]);

  extern void   AZ_get_poly_coefficients(int power, double b, double c[],
                                         int param_flag);

  extern int    AZ_get_sym_indx(int, int, int *, int *, int *);

  extern void   AZ_get_x_incr(int options[], int data_org[], int proc_config[],
                              double params[], int i, double **hh, double *rs,
                              double *trash, double **ss, AZ_MATRIX *,
                              AZ_PRECOND *, double *, int *, int *, int);

  extern double AZ_gmax_double(double, int proc_config[]);

  extern int    AZ_gmax_int(int val, int proc_config[]);

  extern double AZ_gmax_matrix_norm(double val[], int indx[], int bindx[],
                                    int rpntr[], int cpntr[], int bpntr[],
                                    int proc_config[], int data_org[]);

  extern double AZ_gmax_vec(int N, double vec[], int proc_config[]);

  extern double AZ_gmin_double(double var, int proc_config[]);

  extern int AZ_gmin_int(int val, int proc_config[]);

  extern double AZ_gsum_double(double , int proc_config[]);

  extern int    AZ_gsum_int(int totals, int proc_config[]);

  extern void   AZ_gsum_vec_int(int vals[], int vals2[], int length,
                                int proc_config[]);

  extern double AZ_gvector_norm(int n, int p, double *x, int *);

  extern void AZ_hold_space(struct context *context, int N);

  extern void   AZ_init_quick_find(int list[], int length, int *shift, int *bins);

  extern void   AZ_init_subdomain_solver(struct context *context);

  extern void   AZ_invorder_vec(double vec[], int data_org[], int update_index[],
                                int rpntr[],double newarray[]);

  extern void AZ_list_print(int ivec[] , int length, double dvec[], int length2);

  extern void AZ_loc_avg(AZ_MATRIX *Amat, double r[], double newr[], int N_fixed,
                         int fixed_pts[], int proc_config[]);

  extern void AZ_lower_triang_vbr_solve(int Nrows, int cpntr[], int bpntr[],
                                        int indx[], int bindx[], double val[], double b[]);

  extern void  AZ_lower_icc(int bindx[],double val[],int N, double rhs[]);

  extern void  AZ_lower_tsolve(double x[],int  , double l[], int il[],
                               int jl[],  double y[] );

  extern double *AZ_manage_memory(unsigned int size, int action, int type,
                                  char *name, int *status);

  extern struct AZ_MATRIX_STRUCT *AZ_matrix_create(int local);

  extern struct AZ_MATRIX_STRUCT *AZ_submatrix_create(AZ_MATRIX *Amat, int Nsub_rows,
                                                      int sub_rows[], int Nsub_cols, int sub_cols[], int proc_config[]);

  void AZ_submatrix_destroy(AZ_MATRIX **submat);

  extern struct AZ_MATRIX_STRUCT *AZ_blockmatrix_create(AZ_MATRIX **submat_list, int Nsub_mats,
                                                        int **submat_locs, int Nblock_rows, int Nblock_cols, int Nsub_rows[], int **sub_rows,
                                                        int Nsub_cols[], int **sub_cols, int proc_config[]);

  void AZ_blockmatrix_destroy(AZ_MATRIX **blockmat);

  extern void AZ_matrix_init(AZ_MATRIX *Amat, int local);

  typedef void (*AZ_PREC_FUN)(double*, int*, int*, double*,
                              struct AZ_MATRIX_STRUCT*,
                              struct AZ_PREC_STRUCT*);

  extern struct AZ_PREC_STRUCT   *AZ_precond_create(struct AZ_MATRIX_STRUCT *Pmat,
                                                    AZ_PREC_FUN,
                                                    void *data);

  extern void AZ_matrix_destroy( struct AZ_MATRIX_STRUCT **Amat);
  extern void AZ_precond_destroy(struct AZ_PREC_STRUCT **precond);



  extern void AZ_matfree_Nnzs(AZ_MATRIX *Amat);

  extern void AZ_matfree_2_msr(AZ_MATRIX *Amat,double *val, int *bindx, int N_nz);

#ifdef AZ_COL_REORDER
  extern void AZ_mat_colperm(int N, int bindx2[], double val2[],
                             int **inv_ordering, int name, struct context *);
#endif

  extern void AZ_mat_reorder(int n, int bindx[], double val[], int perm[],
                             int invp[]);

  extern void   AZ_matvec_mult(double *val, int *indx, int *bindx, int *rpntr,
                               int *cpntr, int *bpntr, double *b, double *c,
                               int exchange_flag, int *data_org);

  extern void AZ_mk_context(int options[], double params[], int data_org[],
                            AZ_PRECOND *precond, int proc_config[]);

  extern void AZ_mk_identifier(double *params, int *options,
                               int *data_org, char *tag);

  extern void AZ_MSR_mult_patterns(int *bindx, int N, int *work1, int length,
                                   int *work2);

  extern void AZ__MPI_comm_space_ok(void);

  extern int  AZ_MSR_getrow(int columns[], double values[], int row_lengths[],
                            struct AZ_MATRIX_STRUCT *Amat, int N_requested_rows,
                            int requested_rows[], int allocated_space);

  extern int  AZ_VBR_getrow(int columns[], double values[], int row_lengths[],
                            struct AZ_MATRIX_STRUCT *Amat, int N_requested_rows,
                            int requested_rows[], int allocated_space);




  extern void   AZ_msr2lu(int oldN, AZ_MATRIX *A_overlapped, int *rnr);

  extern void   AZ_msr2vbr(double val[], int indx[], int rnptr[], int cnptr[],
                           int bnptr[], int bindx[], int msr_bindx[],
                           double msr_val[], int total_blk_rows,
                           int total_blk_cols, int blk_space, int nz_space,
                           int blk_type);

  extern void AZ_msr2vbr_mem_efficient(int N, int **ibindx, double **ival,
                                       int **icpntr, int **ibpntr, int **iindx,
                                       int *N_blk_rows, int name, char *label, int);

  extern void   AZ_order(int M, double *val_old, double *val_new, int *bindx,
                         int *indx_old, int *indx_new, int *bpntr,
                         int *diag_bloc);

  extern void   AZ_order_ele(int update_index[], int extern_index[],
                             int *internal, int *border, int N_update,
                             int msr_bindx[], int bindx[], int extern_proc[],
                             int N_external, int option, int m_type);

  extern void   AZ_p_error(char *str, int proc);

  extern void AZ_pad_matrix(struct context *context, int proc_config[],
                            int N_unpadded, int *N, int **map, int **padded_data_org,
                            int *N_nz, int estimated_requirements);

  extern void   AZ_pbicgstab(double *, double *, double *, int *, double *,
                             int *, double *, AZ_MATRIX *, AZ_PRECOND *, struct AZ_CONVERGE_STRUCT *);

  extern void   AZ_pcg_f(double *, double *, double *, int *, double * , int * ,
                         double *, AZ_MATRIX *, AZ_PRECOND *, struct AZ_CONVERGE_STRUCT *);

  extern void   AZ_pcgs(double *, double *, double *, int *, double * , int * ,
                        double *, AZ_MATRIX *, AZ_PRECOND *, struct AZ_CONVERGE_STRUCT *);

  extern void AZ_perror(char *string);

  extern void   AZ_pgmresr(double *, double *, double *, int *, double * , int * ,
                           double *, AZ_MATRIX *, AZ_PRECOND *, struct AZ_CONVERGE_STRUCT *);

  extern void   AZ_pgmres(double *, double *, double *, int *, double * , int * ,
                          double *, AZ_MATRIX *, AZ_PRECOND *, struct AZ_CONVERGE_STRUCT *);

  extern void   AZ_polynomial_expansion( double z[], int options[],
                                         int proc_config[], AZ_PRECOND *);

  extern int AZ_pos( int , int bindx[] , int position[], int inv_ordering[],
                     double , int );


  extern void   AZ_precondition(double x[], int options[], int proc_config[],
                                double params[], AZ_MATRIX *, AZ_PRECOND *);

  extern void   AZ_pqmrs(double *, double *, double *, int *, double * , int * ,
                         double *, AZ_MATRIX *, AZ_PRECOND *, struct AZ_CONVERGE_STRUCT *);

  extern void   AZ_print_call_iter_solve(int * , double *, int , int, AZ_MATRIX *, AZ_PRECOND *);

  extern void   AZ_print_error(int error_code);

  extern void AZ_print_header(int options[], int mem_overlapped,
                              int mem_orig, int mem_factor);

#ifdef AZ_ENABLE_CAPTURE_MATRIX
  extern void   AZ_capture_matrix(AZ_MATRIX * Amat,
                                  int proc_config[], int data_org[], double b[]);
#endif

  extern void AZ_print_out(int update_index[], int extern_index[], int update[],
                           int external[],
                           double val[], int indx[],  int
                           bindx[], int rpntr[], int cpntr[], int bpntr[], int
                           proc_config[], int choice, int matrix, int N_update,
                           int N_external, int off_set );

  extern void   AZ_print_sync_start(int proc,int do_print_line,int proc_config[]);

  extern void   AZ_print_sync_end(int proc_config[], int do_print_line);

  extern void   AZ_processor_info(int proc_config[]);

  extern void AZ_put_in_dbl_heap(int *row, double vals[], int heap[],
                                 int *length);
  extern void AZ_put_in_heap(int heap[], int *val,int *length);

  extern int    AZ_quick_find(int key, int list[],int length, int shift,
                              int bins[]);

  extern void   AZ_random_vector(double u[], int data_org[], int proc_config[]);

  extern void   AZ_read_msr_matrix(int update[], double **val, int **bindx,
                                   int N_update, int proc_config[]);

  extern void   AZ_read_update(int *N_update_blks, int *update_blks[],
                               int proc_config[], int bigN, int chunk,
                               int input_option);

  extern void   AZ_input_msr_matrix(char datafile[], int update[], double **val, int **bindx,
                                    int N_update, int proc_config[]);

  extern void   AZ_input_update(char datafile[], int *N_update_blks, int *update_blks[],
                                int proc_config[], int bigN, int chunk,
                                int input_option);

  extern char *AZ_realloc(void *ptr, unsigned int size);

  extern void AZ_recover_sol_params(int instance, int **sub_options,
                                    double **sub_params, double **sub_status, AZ_MATRIX **sub_matrix,
                                    AZ_PRECOND **sub_precond, struct AZ_SCALING **);

  extern void   AZ_reorder_matrix(int N_update, int bindx[], double val[],
                                  int update_index[], int extern_index[],
                                  int indx[], int rnptr[], int bnptr[],
                                  int N_external, int cnptr[], int option,
                                  int);

  extern void   AZ_reorder_vec(double vec[], int data_org[], int update_index[],
                               int rpntr[]);

  extern void AZ_global2local(int data_org[], int bindx[], int update[],
                              int update_index[], int externs[], int extern_index[]);

  extern void AZ_restore_unreordered_bindx(int bindx[], double val[], int update[],
                                           int update_index[], int external[],
                                           int extern_index[], int data_org[]);

  extern void   AZ_reverse_it(int indx[], int length, int first, int total,
                              int b[]);

  extern void AZ_rm_context(int options[], double params[], int data_org[]);

  extern void AZ_rm_dbl_heap_root(int heap[], double vals[], int *length);

  extern void AZ_rm_heap_root(int heap[], int *length);

  extern void AZ_rm_duplicates(int array[], int *N);

  extern void AZ_row_sum_scaling(int action, AZ_MATRIX *Amat,
                                 double b[], int options[],
                                 struct AZ_SCALING *scaling);

  extern void AZ_scale_f(int action, AZ_MATRIX *Amat, int options[], double b[],
                         double x[], int proc_config[], struct AZ_SCALING *scaling);

  extern struct AZ_SCALING *AZ_scale_matrix_only(AZ_MATRIX *Amat, int options[],
                                                 int proc_config[]);

  extern void AZ_scale_rhs_only(double b[], AZ_MATRIX *Amat, int options[],
                                int proc_config[], struct AZ_SCALING *scaling);

  extern void AZ_scale_sol_only(double x[], AZ_MATRIX *Amat, int options[],
                                int proc_config[], struct AZ_SCALING *scaling);

  extern void AZ_scale_rhs_sol_before_iterate(double x[], double b[],
                                              AZ_MATRIX *Amat, int options[], int proc_config[], struct AZ_SCALING *scaling);

  extern void AZ_unscale_after_iterate(double x[], double b[], AZ_MATRIX *Amat,
                                       int options[], int proc_config[],
                                       struct AZ_SCALING *scaling);

  extern void AZ_clean_scaling(struct AZ_SCALING **scaling);


  extern void   AZ_scale_true_residual(double x[], double b[], double v[],
                                       double w[], double *actual_residual,
                                       double *scaled_r_norm, int options[],
                                       int data_org[], int proc_config[],
                                       AZ_MATRIX *Amat,
                                       struct AZ_CONVERGE_STRUCT *);

  extern struct AZ_SCALING *AZ_scaling_create(void);

  extern void AZ_scaling_destroy(struct AZ_SCALING **temp);

  extern double AZ_second(void);

  extern MPI_AZComm *AZ_get_comm(int proc_config[]);

  extern void AZ_set_comm(int proc_config[], MPI_AZComm );

  extern void AZ_set_MATFREE_name(AZ_MATRIX *Amat, int name);

  extern void AZ_set_matrix_print_string(AZ_MATRIX *Amat,const char str[]);

  extern void AZ_set_MATFREE_matrix_norm(AZ_MATRIX *Amat, double mat_norm);

  extern void AZ_set_MATFREE(AZ_MATRIX *Amat, void *data,
                             void (*matvec)(double *, double *, struct AZ_MATRIX_STRUCT *, int *));

  extern void AZ_set_MATNORMINF(AZ_MATRIX *Amat, void *data,
                             double (*matnorminf)(struct AZ_MATRIX_STRUCT *));

  extern void AZ_set_MATFREE_getrow(AZ_MATRIX *Amat, void *data,
                                    int  (*getrow)(int *, double *, int *, struct AZ_MATRIX_STRUCT *, int ,
                                                   int *, int),
                                    int  (*user_comm)(double *, AZ_MATRIX *), int N_ghost, int proc_config[]);

  extern void AZ_set_MSR(AZ_MATRIX *Amat, int bindx[], double val[],
                         int data_org[], int N_update, int update[], int option);

  extern void AZ_set_VBR(AZ_MATRIX *Amat, int rpntr[], int cpntr[], int bpntr[],
                         int indx[], int bindx[], double val[], int data_org[],
                         int N_update, int update[], int option);


  extern void   AZ_set_message_info(int N_external, int extern_index[],
                                    int N_update, int external[],
                                    int extern_proc[], int update[],
                                    int update_index[], int proc_config[],
                                    int cnptr[], int *data_org[], int);

  extern void AZ_set_precond_print_string(struct AZ_PREC_STRUCT *precond,
                                          const char str[]);

  extern void AZ_set_proc_config(int proc_config[], MPI_AZComm );

  extern int AZ_set_solver_parameters(double *params, int *options, AZ_MATRIX *Amat,
                                      AZ_PRECOND *Pmat, struct AZ_SCALING *S);

  extern void AZ_setup_dd_olap_msr(int N_rows, int *New_N_rows, int *bindx,
                                   double *val, int olap_size, int *proc_config, int *data_org[], int **map3,
                                   int bindx_length,int name, int *prev_data_org,int estimated_requirements,
                                   struct context *context);

  extern double AZ_condest(int N, struct context *context);

  extern void AZ_solve_subdomain(double x[],int N, struct context *context);

  extern void   AZ_sort(int list[], int N, int list2[], double list3[]);

  extern void   AZ_sort_dble(char a[], int indx[], int start, int end, int b[],
                             int *mid, int real_lists, char buffer[], int buf_len,
                             int afirst, int );

  extern void   AZ_sort_ints(char a[], int indx[], int start, int end, int b[],
                             int *mid, int real_lists, char buffer[], int buf_len,
                             int afirst, int );

  extern void AZ_sort_msr(int bindx[], double val[], int N);

  extern void   AZ_sortqlists(char a[], int b[], int lists[], int length,
                              int type_length, int ind_length);

  extern void AZ_space_for_factors(double input_fill, int N_nz, int N,
                                   int *extra_factor_nonzeros, int options[],int bandwidth, int );

  extern void AZ_space_for_kvecs(int request, int **kvec_sizes, double ***saveme,
                                 double **ptap, int *options, int *data_org, char *suffix, int proc, double **);

  extern void AZ_space_for_padded_matrix(int overlap, int N_nonzeros, int N,
                                         int *extra_rows, int *extra_nonzeros, int N_external, int *largest);

  extern void   AZ_splitup_big_msg(int num_neighbors, char *buffer, char *buf2,
                                   unsigned int element_size,
                                   int *start_send_proc,
                                   int *actual_send_length,int *num_nonzeros_recv,
                                   int *proc_num_neighbor, int type,
                                   int *total_num_recv, int *proc_config);

  extern double AZ_srandom1(int *seed);

  extern void AZ_sum_bdry(double x[], int data_org[], int proc_config[]);


  extern void   AZ_sym_block_diagonal_scaling(double val[], int indx[],
                                              int bindx[], int rpntr[],
                                              int cpntr[], int bpntr[],
                                              double b[], int options[],
                                              int data_org[],
                                              int proc_config[]
                                              /* struct AZ_SCALING * */);

  extern void AZ_sym_diagonal_scaling(int action, AZ_MATRIX *Amat,
                                      double b[], double x[], int options[],
                                      int proc_config[], struct AZ_SCALING *scaling);


  extern void   AZ_sym_gauss_seidel(void);

  extern void   AZ_sym_gauss_seidel_sl(double val[], int bindx[], double x[],
                                       int data_org[], int options[], struct context *,
                                       int proc_config[]);

  extern void AZ_sym_reinvscale_sl(double x[], int data_org[], int options[],
                                   int proc_config[], struct AZ_SCALING *scaling);

  extern void   AZ_sym_rescale_sl(double x[], int data_org[], int options[],
                                  int proc_config[],struct AZ_SCALING * );

  extern void AZ_sym_row_sum_scaling(int action, AZ_MATRIX *Amat,
                                     double b[],
                                     double x[], int options[],
                                     int proc_config[], struct AZ_SCALING *scaling);

  extern void   AZ_sync(int proc_config[]);

  extern void   AZ_terminate_status_print(int situation, int iter,
                                          double status[], double rec_residual,
                                          double params[], double scaled_r_norm,
                                          double actual_residual, int options[],
                                          int proc_config[]);

  extern void   AZ_transform(int proc_config[], int *external[], int bindx[],
                             double val[], int update[], int *update_index[],
                             int *extern_index[], int *data_org[], int N_update,
                             int indx[], int bnptr[], int rnptr[], int *cnptr[],
                             int mat_type);

  extern void   AZ_update_block(int i, int k, int j, double val[], int indx[],
                                int bindx[], int cpntr[]);

  extern void  AZ_upper_icc( int bindx[],double val[],int N, double rhs[]);

  extern void AZ_upper_triang_vbr_solve(int Nrows, int cpntr[], int bpntr[],
                                        int indx[], int bindx[], double val[], double b[], int piv[], int dblock[]);

  extern void  AZ_upper_tsolve( double x[],int ,double u[],int iui[],
                                int ju[]);

  extern void   AZ_vb2msr(int m, double val[], int indx[], int bindx[],
                          int rpntr[], int cpntr[], int bpntr[], double msr_val[],
                          int msr_bindx[]);

  void AZ_zero_out_context(struct context *);

  void AZ_version(char string[]);

  extern void   AZ_MSR_matvec_mult(double x[], double b[], AZ_MATRIX *Amat,
                                   int proc_config[]);

  extern void   AZ_VBR_matvec_mult(double x[], double b[], AZ_MATRIX *Amat,
                                   int proc_config[]);

  extern void PAZ_compose_external(int, int*, int *, int *, int **);

  extern void PAZ_find_local_indices(int,int*,int*,int*,int,int*);

  extern void PAZ_order_ele(int*,int,int*, int, int*, int*, int);

  extern void PAZ_set_message_info(int, int, int*, int*, int*, int*,
                                   int **, int ,int,int ,struct context*);

  extern int  PAZ_sorted_search(int, int, int*);

  extern void AZ_pgmres_condnum(double b[], double x[], double weight[], int options[],
				double params[], int proc_config[],double status[],
				AZ_MATRIX *Amat, AZ_PRECOND *precond,
				struct AZ_CONVERGE_STRUCT *convergence_info );

  extern void AZ_pcg_f_condnum(double b[], double x[], double weight[], int options[],
			       double params[], int proc_config[],double status[],
			       AZ_MATRIX *Amat, AZ_PRECOND *precond,
			       struct AZ_CONVERGE_STRUCT *convergence_info );
  

  /*****************************************************************************/
  /*                    IFPACK interface routine                               */
  /*****************************************************************************/
#ifdef IFPACK
  extern void az2ifp_blockmatrix (void **bmat, AZ_MATRIX *Amat);
  extern void ifp_freebiluk( void *precon);
#endif

  /*****************************************************************************/
  /*                    Machine Dependent communication routines               */
  /*****************************************************************************/
  extern unsigned int md_wrap_iread(void *, unsigned int, int *, int *, MPI_AZRequest *);

  extern unsigned int md_wrap_iwrite(void *,unsigned int, int , int ,int *, MPI_AZRequest *);

extern unsigned int md_wrap_wait(void *, unsigned int, int *, int *,int *,MPI_AZRequest *);

extern unsigned int md_wrap_write(void *, unsigned int , int , int , int *);

extern unsigned int md_wrap_request_free(MPI_AZRequest *);

#define mdwrap_request_free(a)   md_wrap_request_free(a)
#ifdef AZTEC_MPI
#define mdwrap_wait(a,b,c,x,y,z)   md_mpi_wait(a,b,c,(x),(y),(z),proc_config)
#define mdwrap_iwrite(a,b,c,x,y,z) md_mpi_iwrite(a,b,c,(x),(y),(z),proc_config)
#define mdwrap_iread(a,b,c,x,y)   md_mpi_iread((a),(b),(c),(x),(y),proc_config)
#define mdwrap_write(a,b,c,x,y)   md_mpi_write((a),(b),(c),(x),(y),proc_config)

extern unsigned int md_mpi_iread(void *, unsigned int, int *, int *,
                MPI_AZRequest *, int *);

extern unsigned int md_mpi_iwrite(void *,unsigned int, int , int ,int *,
                MPI_AZRequest *, int *);

extern unsigned int md_mpi_wait(void *, unsigned int, int *, int *,int *,
                MPI_AZRequest *, int *);

extern unsigned int md_mpi_write(void *, unsigned int ,int , int , int *,int *);
#else
#define mdwrap_wait(a,b,c,x,y,z)   md_wrap_wait(a,b,c,(x),(y),(z))
#define mdwrap_iwrite(a,b,c,x,y,z) md_wrap_iwrite(a,b,c,(x),(y),(z))
#define mdwrap_iread(a,b,c,x,y)   md_wrap_iread((a),(b),(c),(x),(y))
#define mdwrap_write(a,b,c,x,y)   md_wrap_write((a),(b),(c),(x),(y))
#endif
/*****************************************************************************/
/*                    Auxilliary fortran rroutines needed by Aztec           */
/*****************************************************************************/

extern void AZ_FNROOT_F77(int *,int *,int *,int *, int *, int *, int *);

extern void MC64AD_F77(int *, int *, int *, int *, int *, double*,
                    int *, int *, int *, int *, int *, double*,
                    int *, int *);

extern void AZ_RCM_F77(int *, int *,int *, int *,int *, int *, int *);

/*****************************************************************************/
/*                    Auxilliary routines available to users                 */
/*****************************************************************************/

extern void AZ_check_update(int update[], int N_update, int proc_config[]);

extern void AZ_clear_solver_parameters(int handle);

extern void   AZ_mysleep(int i);

extern void   AZ_output_matrix(double val[], int indx[], int bindx[],
                               int rpntr[], int cpntr[], int bpntr[],
                               int proc_config[], int data_org[]);

extern void AZ_print_vbr_matrix(
        int matrix_flag, /* = 0 no matrix output, = 1 output matrix */
        int Proc,        /* Processor number                  */
        int itotal_nodes,/* Number of internal + border nodes */
        int ext_nodes,   /* Number of external nodes          */
        double  val[],   /* matrix A in sparse format (VBR)   */
        int  indx[],     /* The ith element of indx points to the location in */
                         /* val of the (0,0) entry of the ith block entry. The*/
                         /* last element is the number of nonzero entries of  */
                         /* matrix A plus one.                                */
        int bindx[],     /* Contains the block column indices of the non-zero */
                         /* block entries.                                    */
        int rpntr[],     /* The ith element of rpntr indicates the first point*/
                         /* row in the ith block row. The last element is the */
                         /* number of block rows plus one.                    */
        int bpntr[]      /* The ith element of bpntr points to the first block*/
                         /* entry of the ith row in bindx. The last element is*/
                         /* the number of nonzero blocks of matrix A plus one.*/
        );

extern double AZ_sync_timer(int proc_config[]);

/* most Aztec code calls the following functions for stdout/stderr output
  rather than the regular printf and fprintf(stderr,...) functions. This
  allows for the possibility that a C++ user (such as the AztecOO class)
  can specify arbitrary C++ std::ostreams to receive Aztec's output instead
  of having it go to stdout/stderr. The C++ functions for setting the
  ostreams are in AZOO_printf.[h,cpp].
*/
extern int AZ_printf_out(const char* format, ...);
extern int AZ_printf_err(const char* format, ...);

extern void AZ_flush_out();




/*****************************************************************************/
/*                    Routines just used locally at Sandia                   */
/*****************************************************************************/

#ifdef Sandia
extern void   AZ_dvbr_diag_sparax(int m, double *val, int *rpntr, int *bpntr,
                                  double *b, double *c);

extern void   AZ_transpose(int N, double l[], int ijl[], double lt[],
                           int ijlt[], int row_counter[]);

extern void   AZ_psymmlq(double *, double *, double *, int *, double *, int * ,
                       double *, AZ_MATRIX *, AZ_PRECOND *);



extern void   AZ_gather_mesg_info(double x[],int data_org[],char **, char **,
                                  int *, int *);

extern void   AZ_read_local_info(int data_org[], char *message_recv_add[],
                                 int message_recv_length[]);
extern void   AZ_write_local_info(int data_org[], char *message_recv_add[],
                                  char *message_send_add[],
                                  int message_recv_length[],
                                  int message_send_length[]);

#endif

/*****************************************************************************/
/*                    Timing Routine                                         */
/*****************************************************************************/

#ifdef TIME_VB
extern void   AZ_time_kernals(int , int , double , double *, int *, int *,
                              int *, int *, int *, double *, double *, int,
                              double *,AZ_MATRIX *);
#endif

#ifdef next_version
extern void   AZ_sym_rescale_vbr(double x[], int data_org[], int options[]);
#endif

/* When calling this fortran routine from C we need to include an extra     */
/* parameter on the end indicating the string length of the first parameter */

/* #ifdef AZ_PA_RISC */
/* extern void   dgemvnsqr_(int *, double *, double *, double *); */
/* extern void   vec_$dcopy(double *, double *, int *); */
/* extern void   blas_$dgemm(char *, char *, int *, int *, int *, double *, */
/*                          double *, int *, double *, int *, double *, double *, */
/*                          int *, int, int); */
/*#endif */



#ifdef __cplusplus
}
#endif

#endif
