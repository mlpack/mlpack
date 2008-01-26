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
      integer  AZ_NOT_MPI 
      integer  AZ_ver2_1_0_1 
      integer  AZ_ver2_1_0_2 
      integer  AZ_ver2_1_0_3 
      integer  AZ_ver2_1_0_4 
      integer  AZ_ver2_1_0_5 
      integer  AZ_ver2_1_0_6 
      integer  AZ_ver2_1_0_7 
      integer  AZ_ver2_1_0_8 
      integer  AZ_MSG_TYPE 
      integer  AZ_NUM_MSGS 
      integer  AZ_MAX_MEMORY_SIZE 
      integer  AZ_MAX_MSG_BUFF_SIZE 
      integer  AZ_MAX_NEIGHBORS 
      integer  AZ_FALSE 
      integer  AZ_TRUE 
      integer  AZ_MAX_POLY_ORDER 
      integer  AZ_default 
      integer  AZ_cg 
      integer  AZ_gmres 
      integer  AZ_cgs 
      integer  AZ_tfqmr 
      integer  AZ_bicgstab 
      integer  AZ_slu 
      integer  AZ_symmlq 
      integer  AZ_GMRESR 
      integer  AZ_fixed_pt 
      integer  AZ_analyze 
      integer  AZ_lu 
      integer  AZ_BJacobi 
      integer  AZ_row_sum 
      integer  AZ_sym_diag 
      integer  AZ_sym_row_sum 
      integer  AZ_equil
      integer  AZ_sym_BJacobi 
      integer  AZ_none 
      integer  AZ_Jacobi 
      integer  AZ_sym_GS 
      integer  AZ_Neumann 
      integer  AZ_ls 
      integer  AZ_ilu 
      integer  AZ_bilu 
      integer  AZ_icc 
      integer  AZ_ilut 
      integer  AZ_rilu 
      integer  AZ_recursive 
      integer  AZ_smoother 
      integer  AZ_dom_decomp 
      integer  AZ_multilevel 
      integer  AZ_user_precond 
      integer  AZ_bilu_ifp 
      integer  AZ_r0 
      integer  AZ_rhs 
      integer  AZ_Anorm 
      integer  AZ_sol 
      integer  AZ_weighted 
      integer  AZ_expected_values 
      integer  AZ_noscaled 
      integer  AZ_inf_noscaled 
      integer  AZ_all 
      integer  AZ_last 
      integer  AZ_warnings 
      integer  AZ_input_form 
      integer  AZ_global_mat 
      integer  AZ_explicit 
      integer  AZ_calc 
      integer  AZ_recalc 
      integer  AZ_reuse 
      integer  AZ_sys_reuse 
      integer  AZ_diag 
      integer  AZ_full 
      integer  AZ_standard 
      integer  AZ_symmetric 
      integer  AZ_classic 
      integer  AZ_modified 
      integer  AZ_resid 
      integer  AZ_rand 
      integer  AZ_normal 
      integer  AZ_param 
      integer  AZ_breakdown 
      integer  AZ_maxits 
      integer  AZ_loss 
      integer  AZ_ill_cond 
      integer  AZ_solver 
      integer  AZ_scaling 
      integer  AZ_precond 
      integer  AZ_conv 
      integer  AZ_output 
      integer  AZ_pre_calc 
      integer  AZ_max_iter 
      integer  AZ_poly_ord 
      integer  AZ_overlap 
      integer  AZ_type_overlap 
      integer  AZ_kspace 
      integer  AZ_orthog 
      integer  AZ_aux_vec 
      integer  AZ_reorder 
      integer  AZ_keep_info 
      integer  AZ_recursion_level 
      integer  AZ_print_freq 
      integer  AZ_graph_fill 
      integer  AZ_subdomain_solve 
      integer  AZ_init_guess 
      integer  AZ_keep_kvecs 
      integer  AZ_apply_kvecs 
      integer  AZ_orth_kvecs 
      integer  AZ_ignore_scaling 
      integer  AZ_check_update_size 
      integer  AZ_extreme 
      integer  AZ_tol 
      integer  AZ_drop 
      integer  AZ_ilut_fill 
      integer  AZ_omega 
      integer  AZ_rthresh 
      integer  AZ_athresh 
      integer  AZ_update_reduction 
      integer  AZ_temp 
      integer  AZ_weights 
      integer  AZ_matrix_type 
      integer  AZ_N_internal 
      integer  AZ_N_border 
      integer  AZ_N_external 
      integer  AZ_N_int_blk 
      integer  AZ_N_bord_blk 
      integer  AZ_N_ext_blk 
      integer  AZ_N_neigh 
      integer  AZ_total_send 
      integer  AZ_name 
      integer  AZ_internal_use 
      integer  AZ_N_rows 
      integer  AZ_neighbors 
      integer  AZ_rec_length 
      integer  AZ_send_length 
      integer  AZ_send_list 
      integer  SIZEOF_MPI_AZCOMM 
      integer  AZ_OPTIONS_SIZE 
      integer  AZ_PARAMS_SIZE 
      integer  AZ_PROC_SIZE 
      integer  AZ_STATUS_SIZE 
      integer  AZ_COMM_SIZE 
      integer  AZ_CONV_INFO_SIZE 
      integer  AZ_COMMLESS_DATA_ORG_SIZE 
      integer  AZ_its 
      integer  AZ_why 
      integer  AZ_r 
      integer  AZ_rec_r 
      integer  AZ_scaled_r 
      integer  AZ_first_precond 
      integer  AZ_solve_time 
      integer  AZ_Aztec_version 
      integer  AZ_Comm_MPI 
      integer  AZ_node 
      integer  AZ_N_procs 
      integer  AZ_dim 
      integer  AZ_Comm_Set 
      integer  AZ_Done_by_User 
      integer  AZ_linear 
      integer  AZ_file 
      integer  AZ_box 
      integer  AZ_ALLOC 
      integer  AZ_CLEAR 
      integer  AZ_REALLOC 
      integer  AZ_SELECTIVE_CLEAR 
      integer  AZ_SPEC_REALLOC 
      integer  AZ_RESET_STRING 
      integer  AZ_SUBSELECTIVE_CLEAR 
      integer  AZ_EVERYBODY_BUT_CLEAR 
      integer  AZ_EMPTY 
      integer  AZ_LOOKFOR_PRINT 
      integer  AZ_SYS 
      integer  AZ_OLD_ADDRESS 
      integer  AZ_NEW_ADDRESS 
      integer  AZ_SPECIAL 
      integer  AZ_SOLVER_PARAMS 
      integer  AZ_MSR_MATRIX 
      integer  AZ_VBR_MATRIX 
      integer  AZ_USER_MATRIX 
      integer  AZ_SCALE_MAT_RHS_SOL 
      integer  AZ_SCALE_RHS 
      integer  AZ_INVSCALE_RHS 
      integer  AZ_SCALE_SOL 
      integer  AZ_INVSCALE_SOL 
      integer  AZ_NOT_FIRST 
      integer  AZ_FIRST_TIME 
      integer  AZ_QUIT 
      integer  AZ_CONTINUE 
      integer  AZ_PACK 
      integer  AZ_SEND 
      integer  AZ_CONVERT_TO_LOCAL 
      integer  AZ_CONVERT_BACK_TO_GLOBAL 
      integer  AZ_NO_EXTRA_SPACE 
      integer  AZ_ZERO 
      integer  AZ_NOT_ZERO 
      integer  AZ_Nspace 
      integer  AZ_Nkept 
      integer  AZ_left_scaling 
      integer  AZ_right_scaling 
      integer  AZ_left_and_right_scaling 
      integer  AZ_call_scale_f 
      integer  AZ_inv_scaling 
      integer  AZ_low 
      integer  AZ_high 
      integer  AZ_TEST_ELE 
      integer  AZ_EXTERNS 
      integer  AZ_GLOBAL 
      integer  AZ_LOCAL 
      parameter (AZ_NOT_MPI = 1)
      parameter (AZ_MSG_TYPE = 1234)
      parameter (AZ_NUM_MSGS = 20)
      parameter (AZ_MAX_MEMORY_SIZE = 16000000)  
      parameter (AZ_MAX_MSG_BUFF_SIZE = 100000)    
      parameter (AZ_MAX_NEIGHBORS = 250)
      parameter (AZ_FALSE = 0)
      parameter (AZ_TRUE = 1)
      parameter (AZ_MAX_POLY_ORDER = 10) 
      parameter (AZ_default = -10) 
      parameter (AZ_cg = 0) 
      parameter (AZ_gmres = 1) 
      parameter (AZ_cgs = 2) 
      parameter (AZ_tfqmr = 3) 
      parameter (AZ_bicgstab = 4) 
      parameter (AZ_slu = 5) 
      parameter (AZ_symmlq = 6) 
      parameter (AZ_GMRESR = 7) 
      parameter (AZ_fixed_pt = 8) 
      parameter (AZ_analyze = 9) 
      parameter (AZ_lu = 10) 
      parameter (AZ_BJacobi = 2) 
      parameter (AZ_row_sum = 3) 
      parameter (AZ_sym_diag = 4) 
      parameter (AZ_sym_row_sum = 5) 
      parameter (AZ_equil = 6) 
      parameter (AZ_sym_BJacobi = 7) 
      parameter (AZ_none = 0) 
      parameter (AZ_Jacobi = 1) 
      parameter (AZ_sym_GS = 2) 
      parameter (AZ_Neumann = 3) 
      parameter (AZ_ls = 4) 
      parameter (AZ_ilu = 6) 
      parameter (AZ_bilu = 7) 
      parameter (AZ_icc = 8) 
      parameter (AZ_ilut = 9) 
      parameter (AZ_rilu = 11) 
      parameter (AZ_recursive = 12) 
      parameter (AZ_smoother = 13) 
      parameter (AZ_dom_decomp = 14) 
      parameter (AZ_multilevel = 15) 
      parameter (AZ_user_precond = 16) 
      parameter (AZ_bilu_ifp = 17) 
      parameter (AZ_r0 = 0) 
      parameter (AZ_rhs = 1) 
      parameter (AZ_Anorm = 2) 
      parameter (AZ_sol = 3) 
      parameter (AZ_weighted = 4) 
      parameter (AZ_expected_values = 5) 
      parameter (AZ_noscaled = 6) 
      parameter (AZ_inf_noscaled = 7) 
      parameter (AZ_all = -3) 
      parameter (AZ_last = -1) 
      parameter (AZ_warnings = -2) 
      parameter (AZ_input_form = 0) 
      parameter (AZ_global_mat = 1) 
      parameter (AZ_explicit = 2) 
      parameter (AZ_calc = 1) 
      parameter (AZ_recalc = 2) 
      parameter (AZ_reuse = 3) 
      parameter (AZ_sys_reuse = 4) 
      parameter (AZ_diag = -1) 
      parameter (AZ_full = 1) 
      parameter (AZ_standard = 0)
      parameter (AZ_symmetric = 1)
      parameter (AZ_classic = 0)
      parameter (AZ_modified = 1)
      parameter (AZ_resid = 0)
      parameter (AZ_rand = 1)
      parameter (AZ_normal = 0) 
      parameter (AZ_param = 1) 
      parameter (AZ_breakdown = 2) 
      parameter (AZ_maxits = 3) 
      parameter (AZ_loss = 4) 
      parameter (AZ_ill_cond = 5) 
      parameter (AZ_solver = 0)
      parameter (AZ_scaling = 1)
      parameter (AZ_precond = 2)
      parameter (AZ_conv = 3)
      parameter (AZ_output = 4)
      parameter (AZ_pre_calc = 5)
      parameter (AZ_max_iter = 6)
      parameter (AZ_poly_ord = 7)
      parameter (AZ_overlap = 8)
      parameter (AZ_type_overlap = 9)
      parameter (AZ_kspace = 10)
      parameter (AZ_orthog = 11)
      parameter (AZ_aux_vec = 12)
      parameter (AZ_reorder = 13)
      parameter (AZ_keep_info = 14)
      parameter (AZ_recursion_level = 15)
      parameter (AZ_print_freq = 16)
      parameter (AZ_graph_fill = 17)
      parameter (AZ_subdomain_solve = 18)
      parameter (AZ_init_guess = 19)
      parameter (AZ_keep_kvecs = 20)
      parameter (AZ_apply_kvecs = 21)
      parameter (AZ_orth_kvecs = 22)
      parameter (AZ_ignore_scaling = 23)
      parameter (AZ_check_update_size = 24)
      parameter (AZ_extreme = 25)
      parameter (AZ_tol = 0)
      parameter (AZ_drop = 1)
      parameter (AZ_ilut_fill = 2)
      parameter (AZ_omega = 3)
      parameter (AZ_rthresh = 4)
      parameter (AZ_athresh = 5)
      parameter (AZ_update_reduction = 6)
      parameter (AZ_temp = 7)
      parameter (AZ_ill_cond_thresh = 8) 
      parameter (AZ_weights = 9) 
      parameter (AZ_matrix_type = 0)
      parameter (AZ_N_internal = 1)
      parameter (AZ_N_border = 2)
      parameter (AZ_N_external = 3)
      parameter (AZ_N_int_blk = 4)
      parameter (AZ_N_bord_blk = 5)
      parameter (AZ_N_ext_blk = 6)
      parameter (AZ_N_neigh = 7)
      parameter (AZ_total_send = 8)
      parameter (AZ_name = 9)
      parameter (AZ_internal_use = 10)
      parameter (AZ_N_rows = 11)
      parameter (AZ_neighbors = 12)
      parameter (AZ_rec_length = (12) +   AZ_MAX_NEIGHBORS)
      parameter (AZ_send_length = (12) + 2*AZ_MAX_NEIGHBORS)
      parameter (AZ_send_list = (12) + 3*AZ_MAX_NEIGHBORS)
      parameter (SIZEOF_MPI_AZCOMM = 20)
      parameter (AZ_OPTIONS_SIZE = 47)
      parameter (AZ_FIRST_USER_OPTION = 27)
      parameter (AZ_PARAMS_SIZE = 30)
      parameter (AZ_FIRST_USER_PARAM = 10)
      parameter (AZ_PROC_SIZE = (6+SIZEOF_MPI_AZCOMM))
      parameter (AZ_STATUS_SIZE = 11)
      parameter (AZ_COMM_SIZE = AZ_send_list)
      parameter (AZ_CONV_INFO_SIZE = 8)
      parameter (AZ_COMMLESS_DATA_ORG_SIZE = AZ_neighbors)
      parameter (AZ_its = 0)
      parameter (AZ_why = 1)
      parameter (AZ_r = 2)
      parameter (AZ_rec_r = 3)
      parameter (AZ_scaled_r = 4)
      parameter (AZ_first_precond = 5)     
      parameter (AZ_solve_time = 6)     
      parameter (AZ_Aztec_version = 7)     
      parameter (AZ_Comm_MPI = 0)
      parameter (AZ_node = (SIZEOF_MPI_AZCOMM+1))
      parameter (AZ_N_procs = (SIZEOF_MPI_AZCOMM+2))
      parameter (AZ_dim = (SIZEOF_MPI_AZCOMM+3))
      parameter (AZ_Comm_Set = (SIZEOF_MPI_AZCOMM+4))
      parameter (AZ_Done_by_User = 7139)
      parameter (AZ_linear = 0)
      parameter (AZ_file = 1)
      parameter (AZ_box = 2)
      parameter (AZ_ALLOC = 0)
      parameter (AZ_CLEAR = 1)
      parameter (AZ_REALLOC = 2)
      parameter (AZ_SELECTIVE_CLEAR = 3)
      parameter (AZ_SPEC_REALLOC = 4)
      parameter (AZ_RESET_STRING = 5)
      parameter (AZ_SUBSELECTIVE_CLEAR = 6)
      parameter (AZ_EVERYBODY_BUT_CLEAR = 7)
      parameter (AZ_EMPTY = 8)
      parameter (AZ_LOOKFOR_PRINT = 9)
      parameter (AZ_SYS = -914901)
      parameter (AZ_OLD_ADDRESS = 0)
      parameter (AZ_NEW_ADDRESS = 1)
      parameter (AZ_SPECIAL = 13)
      parameter (AZ_SOLVER_PARAMS = -100)
      parameter (AZ_MSR_MATRIX = 1)
      parameter (AZ_VBR_MATRIX = 2)
      parameter (AZ_USER_MATRIX = 3)
      parameter (AZ_SCALE_MAT_RHS_SOL = 0)
      parameter (AZ_SCALE_RHS = 1)
      parameter (AZ_INVSCALE_RHS = 2)
      parameter (AZ_SCALE_SOL = 3)
      parameter (AZ_INVSCALE_SOL = 4)
      parameter (AZ_NOT_FIRST = 0)   
      parameter (AZ_FIRST_TIME = 1)   
      parameter (AZ_QUIT = 5)
      parameter (AZ_CONTINUE = 6)
      parameter (AZ_PACK = 0)
      parameter (AZ_SEND = 1)
      parameter (AZ_CONVERT_TO_LOCAL = 0)
      parameter (AZ_CONVERT_BACK_TO_GLOBAL = 1)
      parameter (AZ_NO_EXTRA_SPACE = 0)
      parameter (AZ_ZERO = 0)
      parameter (AZ_NOT_ZERO = 1)
      parameter (AZ_Nspace = 0)
      parameter (AZ_Nkept = 1)
      parameter (AZ_left_scaling = 1) 
      parameter (AZ_right_scaling = 2) 
      parameter (AZ_left_and_right_scaling = 3) 
      parameter (AZ_call_scale_f = 4) 
      parameter (AZ_inv_scaling = 5) 
      parameter (AZ_low = 0)
      parameter (AZ_high = 1)
      parameter (AZ_TEST_ELE = 3)
      parameter (AZ_EXTERNS = 2) 
      parameter (AZ_GLOBAL = 1) 
      parameter (AZ_LOCAL = 2) 
