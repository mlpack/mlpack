librule(
	name = "svm",
	
	sources = ["svm_main.cc"],
	
        headers = ["opt_smo.h", "opt_sgd.h", "svm.h"],
	
	deplibs = ["fastlib:fastlib"],
	
)

binrule(
    # The name of the executable.
    name = "svm_main",
    
    sources = ["svm_main.cc"],
   
    headers = ["opt_smo.h", "opt_sgd.h", "svm.h"],

    deplibs = [":svm"]

    )

librule(
    name = "utils",                        # this line can be safely omitted
    sources = ["utils.cc"],                 # files that must be compiled
    headers=["utils.h"],
    deplibs =["fastlib:fastlib_int","mlpack/fastica:fastica_lib"]
    )


#librule(
 #   name = "libsdp",                        # this line can be safely omitted
  #  sources = ["add_mat.cc","addscaledmat.cc","allocmat.cc","calc_dobj.cc","calc_pobj.cc","chol.cc","copy_mat.cc","easysdp.cc","Fnorm.cc","freeprob.cc","initparams.cc","initsoln.cc","linesearch.cc","makefill.cc","make_i.cc","mat_mult.cc","mat_multsp.cc","matvec.cc","norms.cc","op_a.cc","op_at.cc","op_o.cc","packed.cc","psd_feas.cc","qreig.cc","readprob.cc","readsol.cc","sdp.cc","solvesys.cc","sortentries.cc","sym_mat.cc","trace_prod.cc","tweakgap.cc","user_exit.cc","writeprob.cc","writesol.cc","zero_mat.cc"],  
   # headers = ["blockmat.h","declarations.h","index.h","parameters.h"],
   # deplibs =["fastlib:fastlib_int"]
   # )

librule(
    name = "libsdp",                        # this line can be safely omitted
    sources = ["add_mat.cc","addscaledmat.cc","allocmat.cc","calc_dobj.cc","calc_pobj.cc","chol.cc","copy_mat.cc","easysdp.cc","Fnorm.cc","freeprob.cc","initparams.cc","initsoln.cc","linesearch.cc","makefill.cc","make_i.cc","mat_mult.cc","mat_multsp.cc","matvec.cc","norms.cc","op_a.cc","op_at.cc","op_o.cc","packed.cc","psd_feas.cc","qreig.cc","readprob.cc","readsol.cc","sdp.cc","solvesys.cc","sortentries.cc","sym_mat.cc","trace_prod.cc","tweakgap.cc","user_exit.cc","writeprob.cc","writesol.cc","zero_mat.cc"],  
    headers = ["blockmat.h","declarations.h","index.h","parameters.h"],
    deplibs =["fastlib:fastlib_int"]
    )
librule(
    name = "example",                        # this line can be safely omitted
    sources = ["example.cc"],                 # files that must be compiled
    headers=["new_ext_header.h"],
    deplibs =[":libsdp","fastlib:fastlib_int","mlpack/fastica:fastica_lib"]
    )

librule(
    name = "lkm_lib",                        # this line can be safely omitted
    sources = [],                            # files that must be compiled
   # headers = ["local_kernel_machines_def.h","utils_lkm.h","local_kernel_machines_impl.h","my_crossvalidation.h"],
    deplibs =["fastlib:fastlib_int","mlpack/fastica:fastica_lib","mlpack/svm:svm",":utils",":example"]
    )

binrule(
    name = "lkm",
    sources = ["local_kernel_machines_main.cc"],
    #headers = ["local_kernel_machines_def.h","utils_lkm.h","local_kernel_machines_impl.h","my_crossvalidation.h"],
    deplibs = [":lkm_lib","fastlib:fastlib_int",":libsdp"]
    )


# to build:
# 1. make sure have environment variables set up:
#    $ source /full/path/to/fastlib/script/fl-env /full/path/to/fastlib
#    (you might want to put this in bashrc)
# 2. fl-build main
#    - this automatically will assume --mode=check, the default
#    - type fl-build --help for help
# 3. ./main
#    - to build same target again, type: make
#    - to force recompilation, type: make clean
