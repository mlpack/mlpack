librule(
    name = "cover_tree",
    sources = [],
    headers = ["ctree.h", "ctree_impl.h", "cover_tree.h"],
    deplibs = ["fastlib:fastlib"],
    tests = ["cover_tree_test.cc"]
)

librule(
    name = "emst_cover",
    headers = ["dtb_cover.h", "emst_cover.h"],
    deplibs = ["fastlib:fastlib", "mlpack/emst:union_find", ":cover_tree"]
)

binrule (
    name = "emst_cover_main",
    sources = ["emst_cover_main.cc"],
    deplibs = [":emst_cover"]
)

binrule (
   name = "comparison_main",
   sources = ["comparison_main.cc"],
   deplibs = [":emst_cover", "mlpack/emst:dtb", "mlpack/emst:union_find"]
)
