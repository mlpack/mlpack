librule(
    name = "kmeans_nonempty",
    sources = [],
    headers = ["kmeans_nonempty.h", "la_augment.h", "do_mcf.h", "mcf/readmin.c", "mcf/mcfutil.c","mcf/pbeampp2.c","mcf/pstart.c","mcf/pbla.c", "mcf/treeup.c", "mcf/pflowup.c", "mcf/psimplex.c","mcf/output.c"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "kmeans_nonempty_test",
    sources = ["kmeans_nonempty.cc"],
    headers = [],
    deplibs = [":kmeans_nonempty"]
    )

binrule(
    name = "domcf",
    sources = ["do_mcf.cc"],
    headers = ["do_mcf.h", "mcf/readmin.c", "mcf/mcfutil.c","mcf/pbeampp2.c","mcf/pstart.c","mcf/pbla.c", "mcf/treeup.c", "mcf/pflowup.c", "mcf/psimplex.c","mcf/output.c"],
    deplibs = []
    )
