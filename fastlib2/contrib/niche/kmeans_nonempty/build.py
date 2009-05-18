binrule(
    name = "kmeans",
    sources = ["kmeans.cc"],
    headers = ["kmeans.h", "do_mcf.h", "mcf/readmin.c", "mcf/mcfutil.c","mcf/pbeampp2.c","mcf/pstart.c","mcf/pbla.c", "mcf/treeup.c", "mcf/pflowup.c", "mcf/psimplex.c","mcf/output.c"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "domcf",
    sources = ["do_mcf.cc"],
    headers = ["do_mcf.h", "mcf/readmin.c", "mcf/mcfutil.c","mcf/pbeampp2.c","mcf/pstart.c","mcf/pbla.c", "mcf/treeup.c", "mcf/pflowup.c", "mcf/psimplex.c","mcf/output.c"],
    deplibs = []
    )
