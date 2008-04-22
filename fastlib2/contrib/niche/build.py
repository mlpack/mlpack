binrule(
    name = "solvelinsys",
    sources = ["solvelinsys_main.cc"],
    headers = ["solvelinsys.h","kernel_vector_mult.h"],
    deplibs = ["contrib/dongryel/regression:krylov_lpr",
               "fastlib:fastlib_int",
               "fastlib/sparse/trilinos:trilinos"]
    )

# A librule creates a library, or code that lacks a main function.
# You can define many librules in a single build.py.
librule(
    name = "kvm",
    headers = ["kernel_vector_mult.h"],
    deplibs = ["fastlib:fastlib"],
    )


binrule(
    name = "kvm_main",
    sources = ["kernel_vector_mult_main.cc"],
    deplibs = [":kvm"]
    )
