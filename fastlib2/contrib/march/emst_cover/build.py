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

librule(
    name = "geomst2",
    headers = ["geomst2.h", "emst_cover.h"],
    deplibs = ["fastlib:fastlib", "mlpack/emst:union_find"]
)

librule(
    name = "single_fragment",
    headers = ["friedman_bentley.h", "emst_cover.h"],
    sources = ["friedman_bentley.cc"],
    deplibs = ["fastlib:fastlib", "mlpack/emst:union_find"]
)

librule(
    name = "multi_fragment",
    headers = ["multi_fragment.h", "emst_cover.h"],
    sources = ["multi_fragment.cc"],
    deplibs = ["fastlib:fastlib", "mlpack/emst:union_find"]
)

binrule (
    name = "emst_cover_main",
    sources = ["emst_cover_main.cc"],
    deplibs = [":emst_cover", "mlpack/emst:dtb", "mlpack/emst:union_find", ":geomst2",
               ":single_fragment", ":multi_fragment"]
)

binrule (
   name = "comparison_main",
   sources = ["comparison_main.cc"],
   headers = ["mst_comparison.h"],
   deplibs = [":emst_cover", "mlpack/emst:dtb", "mlpack/emst:union_find", ":geomst2",
              ":single_fragment", ":multi_fragment"]
)
