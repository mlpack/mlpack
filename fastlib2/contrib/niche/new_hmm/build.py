binrule(
    name = "test_bayes",
    sources = ["test_bayes.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "test_utils.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "loghmm.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "save_one_hmm",
    sources = ["save_one_hmm.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "utils.h",
               "diag_gaussian.h",
               "loghmm.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "save_one_fmri_hmm",
    sources = ["save_one_fmri_hmm.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "utils.h",
               "diag_gaussian.h",
               "loghmm.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_fmri_gmmk",
    sources = ["test_fmri_gmmk.cc"],
    headers = ["test_fmri_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "generative_mmk.h",
               "generative_mmk_impl.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "hmm.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_fmri_bayes",
    sources = ["test_fmri_bayes.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "loghmm.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "test_load",
    sources = ["test_load.cc"],
    headers = ["utils.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "test_k",
    sources = ["test_k.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib"]
)


binrule(
    name = "kpca",
    sources = ["kpca.cc"],
    headers = ["kernel_pca.h"],
    deplibs = ["fastlib:fastlib"]
    )


binrule(
    name = "test_inbio_kpca",
    sources = ["test_inbio_kpca.cc"],
    headers = ["generative_mmk.h",
               "generative_mmk_impl.h",
               "kernel_pca.h",
               "utils.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion"]
    )


binrule(
    name = "test_kill_duplicate_points",
    sources = ["test_kill_duplicate_points.cc"],
    headers = ["utils.h"],
    deplibs = ["fastlib:fastlib"]
    )


binrule(
    name = "save_one_dna_hmm",
    sources = ["save_one_dna_hmm.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "utils.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "save_one_synth_hmm",
    sources = ["save_one_synth_hmm.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "utils.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "test_dna_fisher_kernel",
    sources = ["test_dna_fisher_kernel.cc"],
    headers = ["test_dna_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "fisher_kernel.h",
               "fisher_kernel_impl.h",
               "hmm_kernel_utils.h",
               "utils.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_synth_fisher_kernel",
    sources = ["test_synth_fisher_kernel.cc"],
    headers = ["test_synth_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "fisher_kernel.h",
               "fisher_kernel_impl.h",
               "hmm_kernel_utils.h",
               "utils.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "test_dna_emmk",
    sources = ["test_dna_emmk.cc"],
    headers = ["test_dna_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "empirical_mmk.h",
               "empirical_mmk_impl.h",
               "utils.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_synth_emmk",
    sources = ["test_synth_emmk.cc"],
    headers = ["test_synth_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "empirical_mmk.h",
               "empirical_mmk_impl.h",
               "utils.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "test_dna_bayes",
    sources = ["test_dna_bayes.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "hmm.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_synth_bayes",
    sources = ["test_synth_bayes.cc"],
    headers = ["test_synth_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "hmm.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "test_dna_lmmk",
    sources = ["test_dna_lmmk.cc"],
    headers = ["test_dna_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "latent_mmk.h",
               "latent_mmk_impl.h",
               "hmm_kernel_utils.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "hmm.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_synth_lmmk",
    sources = ["test_synth_lmmk.cc"],
    headers = ["test_synth_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "latent_mmk.h",
               "latent_mmk_impl.h",
               "hmm_kernel_utils.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "hmm.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "test_dna_lmmk2",
    sources = ["test_dna_lmmk2.cc"],
    headers = ["test_dna_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "latent_mmk.h",
               "latent_mmk_impl.h",
               "hmm_kernel_utils.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "hmm.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_synth_lmmk2",
    sources = ["test_synth_lmmk2.cc"],
    headers = ["test_synth_utils.h",
               "test_engine.h",
               "test_engine_impl.h",
               "latent_mmk.h",
               "latent_mmk_impl.h",
               "hmm_kernel_utils.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "hmm.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "test_dna_gmmk",
    sources = ["test_dna_gmmk.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "generative_mmk.h",
               "generative_mmk_impl.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "hmm.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_synth_gmmk",
    sources = ["test_synth_gmmk.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "generative_mmk.h",
               "generative_mmk_impl.h",
               "utils.h",
               "multinomial.h",               
               "isotropic_gaussian.h",
               "diag_gaussian.h",
               "mixture.h",
               "hmm.h",
               "../svm/svm.h",
               "../svm/smo.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "empirical_mmk",
    sources = ["empirical_mmk.cc"],
    headers = ["empirical_mmk.h",
               "empirical_mmk_impl.h",
               "utils.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "test_engine",
    sources = ["test_engine.cc"],
    headers = ["test_engine.h",
               "test_engine_impl.h",
               "../svm/smo.h",
               "../svm/svm.h",
               "generative_mmk.h",
               "generative_mmk_impl.h",
               "utils.h",
               "isotropic_gaussian.h",
               "multinomial.h",               
               "hmm.h",
               "diag_gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

# The driver for the bandwidth cross-validator.
binrule(
    name = "get_optimal_kde_bandwidth",
    sources = ["get_optimal_kde_bandwidth.cc"],
    headers = [],
    deplibs = ["mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "fastlib:fastlib_int"]
    )


binrule(
    name = "test_isotropic_gaussian",
    sources = ["test_isotropic_gaussian.cc"],
    headers = ["isotropic_gaussian.h"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "generative_mmk",
    sources = ["generative_mmk.cc"],
    headers = ["generative_mmk.h",
               "generative_mmk_impl.h",
               "utils.h",
               "isotropic_gaussian.h",
               "multinomial.h",               
               "hmm.h",
               "diag_gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib",
               "mlpack/kde:dualtree_kde",
               "mlpack/series_expansion:series_expansion",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_latent_mmk",
    sources = ["test_latent_mmk.cc"],
    headers = ["latent_mmk.h",
               "latent_mmk_impl.h",
               "hmm_kernel_utils.h",
               "utils.h",
               "hmm.h",
               "multinomial.h",
               "diag_gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "latent_mmk",
    sources = ["latent_mmk.cc"],
    headers = ["latent_mmk.h",
               "latent_mmk_impl.h",
               "hmm_kernel_utils.h",
               "hmm.h",
               "multinomial.h",
               "diag_gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "loghmm",
    sources = ["hmm.cc"],
    headers = ["loghmm.h",
               "multinomial.h",
               "diag_gaussian.h",
               "mixture.h",
               "utils.h"],
    deplibs = ["fastlib:fastlib",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "hmm",
    sources = ["hmm.cc"],
    headers = ["hmm.h",
               "multinomial.h",
               "diag_gaussian.h",
               "mixture.h",
               "utils.h"],
    deplibs = ["fastlib:fastlib",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )


binrule(
    name = "tc",
    sources = ["test_template_classes.cc"],
    headers = ["test_template_classes.h",
               "multinomial.h",
               "gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "testMatrix",
    sources = ["testMatrix.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "otrav_mem_test",
    sources = ["otrav_mem_test.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib"]
    )
