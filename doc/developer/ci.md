# mlpack continuous integration (CI) systems

Every pull request submitted to mlpack goes through a number of automated checks
to make sure that all unit tests pass, all code matches the desired style guide,
documentation does not contain any broken links, and so on and so forth.

In general, all CI checks need to pass for PRs to be merged, but like any
complex project, there are occasionally spurious failures or other unrelated
problems.

 * [Basic compilation and test jobs](#basic-compilation-and-test-jobs)
 * [Documentation builds](#documentation-builds)
 * [Style checks](#style-checks)
 * [Cross-compilation checks](#cross-compilation-checks)
 * [Memory checks](#memory-checks)

Also you can see the [list of CI infrastructure](#list-of-ci-infrastructure).

## Basic compilation and test jobs

Basic compilation and testing is done on Github Actions.
We use Github Actions primarily because of the large number of resources that
an mlpack build takes; our own [internal resources](#list-of-ci-infrastructure)
are thus preserved for more specific usage.

Link: [***mlpack on Github Actions***](https://github.com/mlpack/mlpack/actions)

 * Builds and tests mlpack for Linux, macOS, and Windows.

 * Also builds and tests all bindings on Linux, and Python bindings on macOS.

 * Builds the R tarball that can be submitted to CRAN and publishes it as an
   artifact.

 * Configurations for these jobs can be found in the mlpack repository under the
   `.github/` directory.
   - The main file for the build pipeline is `.github/workflows/ci.yml`.

 * *These jobs are most of what's shown in the jobs in a PR.*

***If your build is failing on Github Actions:***

 * Take a look at the build summary and build logs to identify the issue.

 * If the failure is during `mlpack_test`, look through the test output to find
   where the actual failed test is.
   - If the failed test is related to your code, you probably have a bug to fix.
     :)
   - If the failed test does not seem related at all, it could be a spurious
     error in another test.
   - You can run the test locally with `bin/mlpack_test NameOfTest`.
   - If the test seems like a random failure, try different random seeds:
     `bin/mlpack_test --rng-seed=X NameOfTest`.

## Binding tests

Inside of `.github/workflows/ci.yml`, the bindings are configured and built
(depending on the parameters of the matrix build) using a local Github action.
Similarly, the binding tests are also run using a local Github action.

 * `.github/actions/binding_setup/main.yml` defines the steps required to set up
   the environment for building each type of binding.  Sometimes the steps are
   specific to an OS (e.g. Linux/macOS/Windows).

 * `.github/actions/binding_run_tests/main.yml` defines the steps to run tests
   for each binding type.  Note that because we want output as junit XML so we
   can parse it, sometimes we have to do strange things for some languages, and
   we can't use CTest directly.

## Documentation builds

The 'documentation link check' job and 'documentation snippet build' jobs build
and tests *all* documentation, checking:

 * that all Markdown pages build and render properly;
 * that all HTML is valid;
 * that all links referenced in the documentation are valid;
 * that all code examples compile and run.

All of the scripts to perform these builds are located in the `scripts/`
directory, so that they can be run locally.  On CI, these are defined as two
Jenkins jobs in `.jenkins/doc-link-check/Jenkinsfile` and
`.jenkins/doc-snippet-build/Jenkinsfile`.

 * `./scripts/build-docs.sh`
   - Builds all documentation in `doc/` with the output directory `doc/html/`.
   - If you browse to `doc/html/index.html` you can browse locally-built
     documentation.
   - Checks all HTML links and anchors.

 * `./scripts/test-docs.sh doc/`
 * `./scripts/test-docs.sh doc/path/to/file.md`
   - Extracts code blocks from documentation and compiles and runs them.
   - Can be run on either all the documentation (with `doc/` or directory
     argument), or a single file.
   - May require `CXX`, `CXXFLAGS`, and `LDFLAGS` environment variables to be
     set.  See the script itself for more details.
   - If run on an individual file, the output of each compiled code snippet will
     be printed.

When writing new documentation, be sure to test it locally---going back and
forth with the
[link checker job on Jenkins](http://ci.mlpack.org/job/mlpack%20documentation%20link%20check/)
and [snippet build job on Jenkins](http://ci.mlpack.org/job/mlpack%20documentation%20snippet%20build/)
can be very tedious.

## Style checks

The [style checker job](http://ci.mlpack.org/job/mlpack%20style%20checks/) runs
on Jenkins and is defined in `.jenkins/style-checks/Jenkinsfile`.

 * The [`lint.sh` script](https://github.com/mlpack/jenkins-conf/blob/master/linter/lint.sh) to check for C++ style issues.

 * If your job failed this check, look at the "Test Result" tab in the Jenkins
   job.  Style issues for each file will be displayed in an expandable block.

 * See also the
   [style guidelines for mlpack](https://github.com/mlpack/mlpack/wiki/DesignGuidelines).

## Cross-compilation checks

The [cross-compilation checks](http://ci.mlpack.org/job/mlpack%20cross-compile%20tests/)
run on Jenkins and test cross-compilation of mlpack to a number of low-resource
and embedded devices.  The job is defined in
`.jenkins/cross-compilation/Jenkinsfile`.

 * The job builds mlpack in a
   [cross-compilation environment](../embedded/supported_boards.md), targeting a
   number of architectures, and then running tests on actual embedded hardware.

 * Any failures seen here *that are not seen in other jobs* will probably be
   failures specific to the cross-compilation environment.

 * For the list of targeted devices, see the
   [list of CI infrastructure](#list-of-ci-infrastructure).

## Memory checks

The [memory checks](http://ci.mlpack.org/job/mlpack%20memory%20checks/) run
valgrind on any tests that were detected to be changed.  This detection is
performed via a heuristic and may not always be correct.  The job is defined in
`.jenkins/memory-checks/Jenkinsfile`.

 * The [`parse-test.py` script](https://github.com/mlpack/jenkins-conf/blob/master/memory/parse-tests.py)
   is used to find tests that are affected by the changes.

 * The [`memory-check.sh` script](https://github.com/mlpack/jenkins-conf/blob/master/memory/memory-check.sh)
   is used to actually run the tests.

If there are any memory issues with the code, this should be reported by a
failed memory check job.  If you encounter one of these, try compiling with
debugging symbols and running valgrind on the affected test, like this:

```
valgrind --leak-check=full --track-origins=yes bin/mlpack_test "TestName"
```

## List of CI infrastructure

Many physical systems are involved with testing mlpack and are hooked up to
Jenkins.

Link: [***Jenkins (`ci.mlpack.org`)***](http://ci.mlpack.org)

 * The 'specialized' build system.

 * Various Jenkins configuration related resources are found in the
   [`jenkins-conf` repository](https://github.com/mlpack/jenkins-conf/).

 * The list of workers (individual systems) can be found
   [here](http://ci.mlpack.org/computer/).

 * Adding or modifying jobs requires privileges; you can either ask an mlpack
   maintainer to make changes, or if you are on the Contributors team but still
   don't have access, ask somewhere and someone will give you access.  (Probably
   `#mlpack:matrix.org` is the best bet!)

 * A number of embedded devices are available to Jenkins and are used in the
   cross-compilation job.  Each system is named after a main ingredient in a
   good meal eaten just before receiving the embedded device.
    - `couscous.ratml.org`: [Raspberry Pi 5](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf),
      4GB RAM, 4-core ARM Cortex-A76
    - `tofu.ratml.org`: [Raspberry Pi 3 B+](https://datasheets.raspberrypi.com/rpi3/raspberry-pi-3-b-plus-product-brief.pdf),
      1GB RAM, 4-core ARM Cortex-A53
    - `chorizo.ratml.org`: [Jetson Orin Nano](https://nvdam.widen.net/s/zkfqjmtds2/jetson-orin-datasheet-nano-developer-kit-3575392-r2),
      8GB RAM, 6-core ARM Cortex-A78AE, NVIDIA Ampere GPU
    - `tatertot.ratml.org`: [BeagleBone AI-64](https://www.beagleboard.org/boards/beaglebone-ai-64),
      4GB RAM, 2-core ARM Cortex-A72
