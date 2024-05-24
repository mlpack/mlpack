inlineCxxPlugin <- function(...) {
    is_macos <- Sys.info()[["sysname"]] == "Darwin"
    openmp_flag <- if (is_macos) "" else "$(SHLIB_OPENMP_CFLAGS)"
    plugin <-
        Rcpp::Rcpp.plugin.maker(
            include.before = "#include <mlpack.h>",
            libs           = paste(openmp_flag, "$(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)"),
            LinkingTo      = c("RcppArmadillo", "Rcpp", "RcppEnsmallen", "mlpack"),
            package        = "mlpack"
        )
    settings <- plugin()
    settings$env$PKG_CPPFLAGS <- paste("-I../inst/include", openmp_flag)
    # C++17 is required
    if (getRversion() < "4.2.0") settings$env$USE_CXX17 <- "yes"

    settings
}
