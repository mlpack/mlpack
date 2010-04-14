cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
#   bayes_net_kde    ## Huh?  This is all MATLAB?
#   dtw              ## Does not compile right: ComputeDTWAlignmentScore() mismatch in calling syntax
#   hmm              ## Does not actually DO anything
#   hshmm            ## does not compile: invalid use of incomplete type 'struct stat'
   kernel_kmeans
   kernel_pca
#   kernel_vector_mult  ## Does not compile: mismatch in calling of KernelVectorMult::Init()
#   kmeans_nonempty  ## Too few arguments to do_mcf
#   lds              ## No actual usable code here
#   log_newhmm   ## apparent namespace conflicts?
#   mmf  ## apparent namespace conflicts?
#   new_hmm  ## apparent namespace conflicts?
   regularized_ISM
#   spike            ## No actual code in here (except some LaTeX?)
#   spike_train_dependence  ## missing hsic.h
#   svm  ## does not compile, several mismatched function calls
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
