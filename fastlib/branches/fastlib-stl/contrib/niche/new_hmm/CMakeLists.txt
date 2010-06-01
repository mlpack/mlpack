cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   diag_gaussian.h
   empirical_mmk.h
   empirical_mmk_impl.h
   fisher_kernel.h
   fisher_kernel_impl.h
   gaussian.h
   generative_mmk.h
   generative_mmk_impl.h
   hmm.h
   hmm_kernel_utils.h
   isotropic_gaussian.h
   kernel_pca.h
   latent_mmk.h
   latent_mmk_impl.h
   la_utils.h
   loghmm.h
#   logprob_matrixmult.h
   mixture.h
   multinomial.h
   utils.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

#add_executable(test_gmmk_cont_2
#   test_gmmk_cont.cc
#   test_engine.h
#   test_engine_impl.h
#)
#target_link_libraries(test_gmmk_cont_2
#   fastlib
#   mlpack
#   mlpack_contrib
#)

#add_executable(test_gmmk_2
#   test_gmmk.cc
#   test_engine.h
#   test_engine_impl.h
#)
#target_link_libraries(test_gmmk_2
#   fastlib
#   mlpack
#   mlpack_contrib
#)

add_executable(test_bayes_2
   EXCLUDE_FROM_ALL
   test_bayes.cc
   test_engine.h
   test_engine_impl.h
   test_utils.h
)
target_link_libraries(test_bayes_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(save_one_hmm_2
   EXCLUDE_FROM_ALL
   save_one_hmm.cc
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(save_one_hmm_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(save_one_fmri_hmm_2
   EXCLUDE_FROM_ALL
   save_one_fmri_hmm.cc
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(save_one_fmri_hmm_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_fmri_gmmk_2
   EXCLUDE_FROM_ALL
   test_fmri_gmmk.cc
#   test_fmri_utils.cc
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_fmri_gmmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_fmri_bayes_2
   EXCLUDE_FROM_ALL
   test_fmri_bayes.cc
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_fmri_bayes_2
   fastlib
   mlpack
   mlpack_contrib
)

#add_executable(test_load_2
#   test_load.cc
#)
#target_link_libraries(test_load_2
#   fastlib
#   mlpack
#   mlpack_contrib
#)

#add_executable(test_k_2
#   test_k.cc
#)
#target_link_libraries(test_k_2
#   fastlib
#   mlpack
#   mlpack_contrib
#)

add_executable(kernel_pca_2
   EXCLUDE_FROM_ALL
   kpca.cc
   kernel_pca.h
)
target_link_libraries(kernel_pca_2
   fastlib
   mlpack   
   mlpack_contrib
)

add_executable(test_inbio_kpca_2
   EXCLUDE_FROM_ALL
   test_inbio_kpca.cc
)
target_link_libraries(test_inbio_kpca_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_kill_duplicate_points_2
   EXCLUDE_FROM_ALL
   test_kill_duplicate_points.cc
)
target_link_libraries(test_kill_duplicate_points_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(save_one_dna_hmm_2
   EXCLUDE_FROM_ALL
   save_one_dna_hmm.cc
)
target_link_libraries(save_one_dna_hmm_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(save_one_synth_hmm_2
   EXCLUDE_FROM_ALL
   save_one_synth_hmm.cc
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(save_one_synth_hmm_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_dna_fisher_kernel_2
   EXCLUDE_FROM_ALL
   test_dna_fisher_kernel.cc
   test_dna_utils.h
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_dna_fisher_kernel_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_synth_fisher_kernel_2
   EXCLUDE_FROM_ALL
   test_synth_fisher_kernel.cc
   test_synth_utils.h
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_synth_fisher_kernel_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_dna_emmk_2
   EXCLUDE_FROM_ALL
   test_dna_emmk.cc
   test_dna_utils.h
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_dna_emmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_synth_emmk_2
   EXCLUDE_FROM_ALL
   test_synth_emmk.cc
   test_synth_utils.h
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_synth_emmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_dna_bayes_2
   EXCLUDE_FROM_ALL
   test_dna_bayes.cc
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_dna_bayes_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_synth_bayes_2
   EXCLUDE_FROM_ALL
   test_synth_bayes.cc
   test_synth_utils.h
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_synth_bayes_2
   fastlib
   mlpack   
   mlpack_contrib
)

add_executable(test_dna_lmmk_2
   EXCLUDE_FROM_ALL
   test_dna_lmmk.cc
   test_dna_utils.h
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_dna_lmmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_synth_lmmk_2
   EXCLUDE_FROM_ALL
   test_synth_lmmk.cc
   test_synth_utils.h
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_synth_lmmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_dna_lmmk2_2
   EXCLUDE_FROM_ALL
   test_dna_lmmk2.cc
   test_dna_utils.h
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_dna_lmmk2_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_synth_lmmk2_2
   EXCLUDE_FROM_ALL
   test_synth_lmmk2.cc
   test_synth_utils.h
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_synth_lmmk2_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_dna_gmmk_2
   EXCLUDE_FROM_ALL
   test_dna_gmmk.cc
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_dna_gmmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_synth_gmmk_2
   EXCLUDE_FROM_ALL
   test_synth_gmmk.cc
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_synth_gmmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(empirical_mmk_2
   EXCLUDE_FROM_ALL
   empirical_mmk.h
   empirical_mmk_impl.h
   empirical_mmk.cc
)
target_link_libraries(empirical_mmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_engine_2
   EXCLUDE_FROM_ALL
   test_engine.cc
   test_engine.h
   test_engine_impl.h
)
target_link_libraries(test_engine_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(get_optimal_kde_bandwidth_2
   EXCLUDE_FROM_ALL
   get_optimal_kde_bandwidth.cc
)
target_link_libraries(get_optimal_kde_bandwidth_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_isotropic_gaussian_2
   EXCLUDE_FROM_ALL
   test_isotropic_gaussian.cc
   isotropic_gaussian.h
)
target_link_libraries(test_isotropic_gaussian_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(generative_mmk_2
   EXCLUDE_FROM_ALL
   generative_mmk.cc
)
target_link_libraries(generative_mmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_latent_mmk_2
   EXCLUDE_FROM_ALL
   test_latent_mmk.cc
)
target_link_libraries(test_latent_mmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(latent_mmk_2
   EXCLUDE_FROM_ALL
   latent_mmk.cc
)
target_link_libraries(latent_mmk_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(loghmm_2
   EXCLUDE_FROM_ALL
   hmm.cc
   loghmm.h
)
target_link_libraries(loghmm_2
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(hmm_niche_2
   EXCLUDE_FROM_ALL
   hmm.cc
   hmm.h
)
target_link_libraries(hmm_niche_2
   fastlib
   mlpack
   mlpack_contrib
)

#add_executable(test_template_classes_2
#   test_template_classes.cc
#   test_template_classes.h
#)
#target_link_libraries(test_template_classes_2
#   fastlib
#   mlpack
#   mlpack_contrib
#)

#add_executable(test_matrix_2
#   testMatrix.cc
#)
#target_link_libraries(test_matrix_2
#   fastlib
#   mlpack
#   mlpack_contrib
#)

add_executable(otrav_mem_test_2
   EXCLUDE_FROM_ALL
   otrav_mem_test.cc
)
target_link_libraries(otrav_mem_test_2
   fastlib
   mlpack
   mlpack_contrib
)
