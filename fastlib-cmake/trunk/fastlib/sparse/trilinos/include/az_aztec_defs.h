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
 * $RCSfile: az_aztec_defs.h,v $
 *
 * $Author: william $
 *
 * $Date: 2006/07/14 21:17:42 $
 *
 * $Revision: 1.22 $
 *
 * $Name: trilinos-release-8-0-branch $
 *====================================================================*/

#ifndef __AZTECDEFSH__

#define __AZTECDEFSH__

#ifdef HAVE_CONFIG_H

/*
 * The macros PACKAGE, PACKAGE_NAME, etc, get defined for each package and need to
 * be undef'd here to avoid warnings when this file is included from another package.
 * KL 11/25/02
 */
#ifdef PACKAGE
#undef PACKAGE
#endif

#ifdef PACKAGE_NAME
#undef PACKAGE_NAME
#endif

#ifdef PACKAGE_BUGREPORT
#undef PACKAGE_BUGREPORT
#endif

#ifdef PACKAGE_STRING
#undef PACKAGE_STRING
#endif

#ifdef PACKAGE_TARNAME
#undef PACKAGE_TARNAME
#endif

#ifdef PACKAGE_VERSION
#undef PACKAGE_VERSION
#endif

#ifdef VERSION
#undef VERSION
#endif

#include "AztecOO_config.h"
/*This file doesn't exist in old make and configure system*/

#ifdef HAVE_MPI

#ifndef AZTEC_MPI
#define AZTEC_MPI
#endif

#ifndef AZ_MPI
#define AZ_MPI
#endif

#ifndef EPETRA_MPI
#define EPETRA_MPI
#endif

#ifndef ML_MPI
#define ML_MPI
#endif

#endif /*HAVE_MPI*/

/*
#ifdef HAVE_CSTDLIB
#include <cstdlib>
#else
#include <stdlib.h>
#endif

#ifdef HAVE_CSTDIO
#include <cstdio>
#else
#include <stdio.h>
#endif

#ifdef HAVE_CASSERT
#include <cassert>
#else
#include <assert.h>
#endif

#ifdef HAVE_STRING
#include <string>
#else
#include <string.h>
#endif

#ifndef JANUS_STLPORT
#ifdef HAVE_CMATH
#include <cmath>
#else
#include <math.h>
#endif
#else
#include <math.h>
#endif

#ifdef HAVE_CFLOAT
#include <cfloat>
#else
#include <float.h>
#endif

#ifdef HAVE_CTIME
#include <ctime>
#else
#include <sys/time.h>
#endif

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#else HAVE_CONFIG_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <malloc.h> */
#endif /*HAVE_CONFIG_H*/


#ifndef __AZTECH__
#ifdef AZTEC_MPI
#include <mpi.h>
#define MPI_AZRequest MPI_Request
#define MPI_AZComm    MPI_Comm
#else
#define MPI_AZRequest int
#define MPI_AZComm    int
#endif
#endif

#define AZ_NOT_MPI 1

/******************************************************************************
 *
 *             version information
 *
 *****************************************************************************/
#define AZ_ver2_1_0_1
#define AZ_ver2_1_0_2
#define AZ_ver2_1_0_3
#define AZ_ver2_1_0_4
#define AZ_ver2_1_0_5
#define AZ_ver2_1_0_6
#define AZ_ver2_1_0_7
#define AZ_ver2_1_0_8
#define AZ_ver2_1_0_9


/*******************************************************************************
 *
 *              message types used in communication
 *
 ******************************************************************************/


/* message space used corresponds to AZ_MSG_TYPE --> AZ_MSG_TYPE + NUM_MSGS */

/* In general, AZTEC uses message types that lie between the values: AZ_MSG_TYPE
   and AZ_MSG_TYPE + AZ_NUM_MSGS. Each time we reach a code segment which every
   processor executes, the current message type to be used (the global variable
   AZ_sys_msg_type) is incremeted (modulo NUM_MSGS) to determine the new message
   type to use. */

#define AZ_MSG_TYPE      1234
#define AZ_NUM_MSGS        20

/*******************************************************************************
 *
 *              various constants
 *
 ******************************************************************************/

#ifndef AZ_MAX_MEMORY_SIZE
#define AZ_MAX_MEMORY_SIZE   16000000  /* maximum memory size used for the LU */
/* within domain decomposition.        */
#endif
#ifndef AZ_MAX_MSG_BUFF_SIZE
#define AZ_MAX_MSG_BUFF_SIZE 100000    /* max usable message buffer size      */
#endif
#define AZ_MAX_NEIGHBORS     250
#define AZ_MAX_MESSAGE_SIZE  (AZ_MAX_MSG_BUFF_SIZE / (2*AZ_MAX_NEIGHBORS))
#define AZ_FALSE               0
#define AZ_TRUE                1
#define AZ_MAX_POLY_ORDER     10 /* max order for polynomial preconditioners */
#define AZ_default           -10 /* options[i] = AZ_default ==>
                                    AZ_check_input() sets options[i] to
                                    its default value.
                                  */

/*******************************************************************************
 *
 *              constants for solver types
 *
 ******************************************************************************/

#define AZ_cg               0 /* preconditioned conjugate gradient method     */
#define AZ_gmres            1 /* preconditioned gmres method                  */
#define AZ_cgs              2 /* preconditioned cg squared method             */
#define AZ_tfqmr            3 /* preconditioned transpose-free qmr method     */
#define AZ_bicgstab         4 /* preconditioned stabilized bi-cg method       */
#define AZ_slu              5 /* super LU direct method.                      */
#define AZ_symmlq           6 /* indefinite symmetric like symmlq             */
#define AZ_GMRESR           7 /* recursive GMRES (not supported)              */
#define AZ_fixed_pt         8 /* fixed point iteration                        */
#define AZ_analyze          9 /* fixed point iteration                        */
#define AZ_lu              10 /* sparse LU direct method. Also used for a     */
#define AZ_cg_condnum      11
#define AZ_gmres_condnum   12
/* preconditioning option.  NOTE: this should   */
/* be the last solver so that AZ_check_input()  */
/* works properly.                              */

/*******************************************************************************
 *
 *              constants for scaling types
 *
 ******************************************************************************/

/* #define AZ_none          0    no scaling                                   */
/* #define AZ_Jacobi        1    Jacobi scaling                               */
#define AZ_BJacobi          2 /* block Jacobi scaling                         */
#define AZ_row_sum          3 /* point row-sum scaling                        */
#define AZ_sym_diag         4 /* symmetric diagonal scaling                   */
#define AZ_sym_row_sum      5 /* symmetric diagonal scaling                   */
#define AZ_equil            6 /* equilib scaling */
#define AZ_sym_BJacobi      7 /* symmetric block Jacobi scaling. NOTE: this   */
/* should be last so that AZ_check_input()      */
/* works properly.                              */

/*******************************************************************************
 *
 *              constants for preconditioner types
 *
 ******************************************************************************/

#define AZ_none             0 /* no preconditioning. Note: also used for      */
/* scaling, output, overlap options options     */
#define AZ_Jacobi           1 /* Jacobi preconditioning. Note: also used for  */
/* scaling options                              */
#define AZ_sym_GS           2 /* symmetric Gauss-Siedel preconditioning       */
#define AZ_Neumann          3 /* Neumann series polynomial preconditioning    */
#define AZ_ls               4 /* least-squares polynomial preconditioning     */
#define AZ_ilu              6 /* domain decomp with  ilu in subdomains        */
#define AZ_bilu             7 /* domain decomp with block ilu in subdomains   */
/* #define AZ_lu           10    domain decomp with   lu in subdomains        */
#define AZ_icc              8 /* domain decomp with incomp Choleski in domains*/
#define AZ_ilut             9 /* domain decomp with ilut in subdomains        */
#define AZ_rilu            11 /* domain decomp with rilu in subdomains        */
#define AZ_recursive       12 /* Recursive call to AZ_iterate()               */
#define AZ_smoother        13 /* Recursive call to AZ_iterate()               */
#define AZ_dom_decomp      14 /* Domain decomposition using subdomain solver  */
/* given by options[AZ_subdomain_solve]         */
#define AZ_multilevel      15 /* Do multiplicative domain decomp with coarse  */
/* grid (not supported).                        */
#define AZ_user_precond    16 /*  user's preconditioning */
/* Begin Aztec 2.1 mheroux mod */
#define AZ_bilu_ifp        17 /* dom decomp with bilu using ifpack in subdom  */
/* End Aztec 2.1 mheroux mod */


/*******************************************************************************
 *
 *              constants for convergence types
 *
 ******************************************************************************/
/*                                                                            */
/* DO NOT change these numbers as they are hard-wired in MPSalsa !!!!         */
/*                                                                            */
#define AZ_r0               0 /* ||r||_2 / ||r^{(0)}||_2                      */
#define AZ_rhs              1 /* ||r||_2 / ||b||_2                            */
#define AZ_Anorm            2 /* ||r||_2 / ||A||_infty                        */
#define AZ_sol              3 /* ||r||_infty/(||A||_infty ||x||_1+||b||_infty)*/
#define AZ_weighted         4 /* ||r||_WRMS                                   */
#define AZ_expected_values  5 /* ||r||_WRMS with weights taken as |A||x0|     */
#define AZ_noscaled         6 /* ||r||_2                                      */
#define AZTECOO_conv_test   7 /* Convergence test will be done via AztecOO    */
#define AZ_inf_noscaled     8 /* ||r||_infty                                  */
/* NOTE: AZ_inf_noscaled should be last         */
/* so that AZ_check_input() works properly.     */

/*******************************************************************************
 *
 *              constants for output types
 *
 ******************************************************************************/

#define AZ_all             -4 /* Print out everything including matrix        */
/* Must be lowest value so that AZ_check_input()*/
/* works properly.                              */
/* #define AZ_none          0    Print out no results (not even warnings)     */
#define AZ_last            -1 /* Print out final residual and warnings        */
#define AZ_summary         -2 /* Print out summary, final residual and warnings*/
#define AZ_warnings        -3 /* Print out only warning messages              */

/*******************************************************************************
 *
 *              constants for matrix output
 *
 ******************************************************************************/

#define AZ_input_form       0 /* Print out the matrix arrays as they appear   */
/* along with some additional information. The  */
/* idea here is to print out the information    */
/* that the user must supply as input to the    */
/* function AZ_transform()                      */
#define AZ_global_mat       1 /* Print out the matrix as a(i,j) where i and j */
/* are the global indices. This option must     */
/* be invoked only after AZ_transform() as the  */
/* array update_index[] is used.                */
/* NOTE: for VBR matrices the matrix is printed */
/* as a(I(i),J(j)) where I is the global block  */
/* row and J is the global block column and i   */
/* and j are the row and column indices within  */
/* the block.                                   */
#define AZ_explicit         2 /* Print out the matrix as a(i,j) where i and j */
/* are the local indices.                       */
/* NOTE: for VBR matrices the matrix is printed */
/* as a(I(i),J(j)) where I is the global block  */
/* row and J is the global block column and i   */
/* and j are the row and column indices within  */
/* the block.                                   */

/*******************************************************************************
 *
 *              constants for using factorization information
 *
 ******************************************************************************/

#define AZ_calc             1 /* use no previous information                  */
#define AZ_recalc           2 /* use last symbolic information                */
#define AZ_reuse            3 /* use a previous factorization to precondition */
#define AZ_sys_reuse        4 /* use last factorization to precondition       */
/* NOTE: AZ_sys_reuse should be last so that    */
/* AZ_check_input() works properly.             */

/*******************************************************************************
 *
 *              constants for domain decompositon overlap
 *
 ******************************************************************************/

/* #define AZ_none          0    No overlap                                   */
#define AZ_diag            -1 /* Use diagonal blocks for overlapping          */
#define AZ_full             1 /* Use external rows   for overlapping          */
/* Note: must be highest value so that          */
/*       AZ_check_input() works properly.       */

/*******************************************************************************
 *
 *              constants to determine if overlapped values are added
 *              (symmetric) or just taken from the closest processor.
 *
 ******************************************************************************/
#define AZ_standard         0
#define AZ_symmetric        1

/*******************************************************************************
 *
 *              constants for GMRES orthogonalization procedure
 *
 ******************************************************************************/

#define AZ_classic          0 /* Does double classic */
#define AZ_modified         1 /* Does single modified */
#define AZ_single_classic   2
#define AZ_single_modified  3
#define AZ_double_classic   4
#define AZ_double_modified  5

/*******************************************************************************
 *
 *              constants for determining rtilda (used in bicgstab, cgs, tfqmr)
 *
 ******************************************************************************/

#define AZ_resid            0
#define AZ_rand             1

/*******************************************************************************
 *
 *              constants indicating reason for iterative method termination
 *
 ******************************************************************************/

#define AZ_normal           0 /* normal termination                           */
#define AZ_param            1 /* requested option not implemented             */
#define AZ_breakdown        2 /* numerical breakdown during the computation   */
#define AZ_maxits           3 /* maximum iterations exceeded                  */
#define AZ_loss             4 /* loss of precision                            */
#define AZ_ill_cond         5 /* GMRES hessenberg is ill-conditioned          */

/*******************************************************************************
 *
 *              array indices into options array
 *
 ******************************************************************************/

#define AZ_solver              0
#define AZ_scaling             1
#define AZ_precond             2
#define AZ_conv                3
#define AZ_output              4
#define AZ_pre_calc            5
#define AZ_max_iter            6
#define AZ_poly_ord            7
#define AZ_overlap             8
#define AZ_type_overlap        9
#define AZ_kspace              10
#define AZ_orthog              11
#define AZ_aux_vec             12
#define AZ_reorder             13
#define AZ_keep_info           14
#define AZ_recursion_level     15
#define AZ_print_freq          16
#define AZ_graph_fill          17
#define AZ_subdomain_solve     18
#define AZ_init_guess          19
#define AZ_keep_kvecs          20
#define AZ_apply_kvecs         21
#define AZ_orth_kvecs          22
#define AZ_ignore_scaling      23
#define AZ_check_update_size   24
#define AZ_extreme             25
#define AZ_diagnostics         26

/*******************************************************************************
 *
 *              array indices into params array
 *
 ******************************************************************************/

#define AZ_tol                 0
#define AZ_drop                1
#define AZ_ilut_fill           2
#define AZ_omega               3
/* Begin Aztec 2.1 mheroux mod */
#define AZ_rthresh             4
#define AZ_athresh             5
#define AZ_update_reduction    6
#define AZ_temp                7
#define AZ_ill_cond_thresh     8
#define AZ_weights             9 /* this parameter should be the last one */
/* End Aztec 2.1 mheroux mod */


/*******************************************************************************
 *
 *              array indices into data_org array
 *
 ******************************************************************************/

#define AZ_matrix_type         0
#define AZ_N_internal          1
#define AZ_N_border            2
#define AZ_N_external          3
#define AZ_N_int_blk           4
#define AZ_N_bord_blk          5
#define AZ_N_ext_blk           6
#define AZ_N_neigh             7
#define AZ_total_send          8
#define AZ_name                9
#define AZ_internal_use        10
#define AZ_N_rows              11
#define AZ_neighbors           12
#define AZ_rec_length          (12 +   AZ_MAX_NEIGHBORS)
#define AZ_send_length         (12 + 2*AZ_MAX_NEIGHBORS)
#define AZ_send_list           (12 + 3*AZ_MAX_NEIGHBORS)

/*******************************************************************************
 *
 *              Array sizes for declarations (MUST APPEAR AFTER DATA_ORG)
 *
 ******************************************************************************/

#define SIZEOF_MPI_AZCOMM     20
/* MPI Communicators are kludged into Aztec's integer proc_config[] */
/* array. SIZEOF_MPI_AZComm must be greater or equal to             */
/*             sizeof(MPI_AZComm)/sizeof(int).                      */
/* If this is not true, an error message will be generated and you  */
/* will be asked to change this value.                              */

#define AZ_OPTIONS_SIZE       47
#define AZ_FIRST_USER_OPTION  27 /* User can define up to 20 options
                                    values starting at AZ_FIRST_USER_OPTION */
/* Begin Aztec 2.1 mheroux mod */
#define AZ_PARAMS_SIZE         30
#define AZ_FIRST_USER_PARAM    10 /* User can define up to 20 params
                                     values starting at AZ_FIRST_USER_PARAM  */
/* End Aztec 2.1 mheroux mod */
#define AZ_PROC_SIZE           (7+SIZEOF_MPI_AZCOMM)
#define AZ_STATUS_SIZE         11
#define AZ_COMM_SIZE          AZ_send_list
#define AZ_CONV_INFO_SIZE      8
#define AZ_COMMLESS_DATA_ORG_SIZE  AZ_neighbors

/*******************************************************************************
 *
 *              array indices into status array
 *
 ******************************************************************************/

#define AZ_its                 0
#define AZ_why                 1
#define AZ_r                   2
#define AZ_rec_r               3
#define AZ_scaled_r            4
#define AZ_first_precond       5     /* This is used to record the time for */
/* the first preconditioning step. The */
/* intention is time factorization     */
/* routines. Note: not mentioned in    */
/* manual                              */
#define AZ_solve_time          6     /* This is used to record the time for */
/* the entire solve.                   */
#define AZ_Aztec_version       7     /* This is used to record the current  */
/* version of Aztec.                   */
#define AZ_condnum             8
#define AZ_lambda_min          9
#define AZ_lambda_max          10

/*******************************************************************************
 *
 *              array indices into proc_config array
 *
 ******************************************************************************/

#define AZ_Comm_MPI            0
#define AZ_node                (SIZEOF_MPI_AZCOMM+1)
#define AZ_N_procs             (SIZEOF_MPI_AZCOMM+2)
#define AZ_dim                 (SIZEOF_MPI_AZCOMM+3)
#define AZ_MPI_Tag             (SIZEOF_MPI_AZCOMM+4)
#define AZ_Comm_Set            (SIZEOF_MPI_AZCOMM+5)

#define AZ_Done_by_User        7139

/*******************************************************************************
 *
 *              partitioning option choices
 *
 ******************************************************************************/

#define AZ_linear              0
#define AZ_file                1
#define AZ_box                 2

/*******************************************************************************
 *
 *              constants for memory management
 *
 ******************************************************************************/

#define AZ_ALLOC               0
#define AZ_CLEAR               1
#define AZ_REALLOC             2
#define AZ_SELECTIVE_CLEAR     3
#define AZ_SPEC_REALLOC        4
#define AZ_RESET_STRING        5
#define AZ_SUBSELECTIVE_CLEAR  6
#define AZ_EVERYBODY_BUT_CLEAR  7
#define AZ_EMPTY                8
#define AZ_LOOKFOR_PRINT        9
#define AZ_CLEAR_ALL           10
#define AZ_SYS                 -914901
#define AZ_OLD_ADDRESS         0
#define AZ_NEW_ADDRESS         1
#define AZ_SPECIAL             13
#define AZ_SOLVER_PARAMS       -100

/*******************************************************************************
 *
 *              constants for matrix types
 *
 ******************************************************************************/

#define AZ_MSR_MATRIX          1
#define AZ_VBR_MATRIX          2
#define AZ_USER_MATRIX         3

/*******************************************************************************
 *
 *              constants for scaling action
 *
 ******************************************************************************/

#define AZ_SCALE_MAT_RHS_SOL   0
#define AZ_SCALE_RHS           1
#define AZ_INVSCALE_RHS        2
#define AZ_SCALE_SOL           3
#define AZ_INVSCALE_SOL        4
#define AZ_DESTROY_SCALING_DATA 5

/*******************************************************************************
 *
 *              constants used for residual expresion calculations
 *              (performed by AZ_compute_global_scalars) within iterative methods
 *
 ******************************************************************************/

#define AZ_NOT_FIRST           0   /* not the first residual expression       */
/* request. Information should be available*/
/* from a previous request and the residual*/
/* may not be available.                   */
#define AZ_FIRST_TIME          1   /* first time that a residual expression   */
/* is requested for a particular iterative */
/* solve. This means that the true residual*/
/* is available and that certain invariant */
/* information (e.g. r_0, ||A||) must be   */
/* computed.                               */

/*******************************************************************************
 *
 *              constants (see AZ_get_new_eps) used to determine whether to
 *              continue or to quit the iterative method when the real
 *              residual does not match the updated residual
 *
 ******************************************************************************/

#define AZ_QUIT             5
#define AZ_CONTINUE         6

/*******************************************************************************
 *
 *              constants (see AZ_broadcast) used to determine whether to
 *              concatenate information to be broadcast or to send information
 *              already stored in an internal buffer
 *
 ******************************************************************************/

#define AZ_PACK             0
#define AZ_SEND             1

#define AZ_CONVERT_TO_LOCAL 0
#define AZ_CONVERT_BACK_TO_GLOBAL 1

#define AZ_NO_EXTRA_SPACE 0
#define AZ_NOT_USING_AZTEC_MATVEC (int *) NULL

#define AZ_ZERO     0
#define AZ_NOT_ZERO 1
#define AZ_Nspace 0
#define AZ_Nkept 1

/* #define AZ_none                  0     do nothing                */
#define AZ_left_scaling             1 /*  scaling on left           */
#define AZ_right_scaling            2 /*  scaling on right          */
#define AZ_left_and_right_scaling   3 /*  scaling on left and right */
#define AZ_call_scale_f             4 /*  use scaling subroutine    */
#define AZ_inv_scaling              5 /*  within scaling routine    */
/*  perform inverse operation */
#define AZ_low             0
#define AZ_high            1


/*******************************************************************************
 *
 *              software tool constants
 *
 ******************************************************************************/

#define AZ_TEST_ELE         3
#define AZ_ALL              1 /* All elements are reordered.                  */
#define AZ_EXTERNS          2 /* Only external elements are reordered.        */
#define AZ_GLOBAL           1 /* MSR entries correspond to global columns     */
#define AZ_LOCAL            2 /* MSR entries correspond to local columns      */

#define AZ_get_matvec_data(Amat) ((Amat)->matvec_data)
#define AZ_get_getrow_data(Amat) ((Amat)->getrow_data)
#define AZ_get_precond_data(precond) ((precond)->precond_data)

#endif
