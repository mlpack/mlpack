if (requireNamespace("tinytest2JUnit", quietly=TRUE)
    && requireNamespace("tinytest", quietly=TRUE)) {
    ## When mlpack runs in CI, it picks up ${LANGUAGE}_bindings.junit.xml files so we
    ## make sure we create one for it; in all other settings the file is ignored (but
    ## also cheap to produce, and tinytest2Junit is lightweight and low dependency).
    ## This also runs and escalates tests as usual via the invocation of tinytest.
    tinytest2JUnit::writeJUnit(tinytest::test_package("mlpack"), "r_bindings.junit.xml")
} else if (requireNamespace("tinytest", quietly=TRUE)) {
    ## Run only tinytest as usual
    tinytest::test_package("mlpack")
}
