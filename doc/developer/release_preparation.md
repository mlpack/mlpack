**mlpack** Release Preparation Guide
====================================

There are some things to be done before a release which are easy to forget but shouldn't be forgotten.  A simple guide is here:

1.  Run `release-mlpack.sh` with the new version number.
  - _notes from 3.3.2 release_: updating Doxyfile needed a bit of changing
  - _notes from 3.3.2 release_: needs to fail if there are any uncommitted changes
  - _notes from 3.3.2 release_: needs to commit finished history block only to tag
  - _notes from 3.3.2 release_: website did not update right; had to change versions in .md links with sed
  - _notes from 3.3.2 release_: soversion for library wasn't right---need to manually update src/mlpack/CMakeLists.txt
  - _notes from 3.3.2 release_: seems like src/mlpack/core/util/version.hpp did not get updated!
  - _notes from 3.4.0 release_: now we check out the master branch
  - _notes from 3.4.0 release_: all files now seem to be updating correctly
1.  Send announcement to mlpack list.
1.  Add release notes to Github page.

Push to distributions
---------------------

###### Fedora

1. Get git source: `fedpkg clone mlpack`.
2. Get new mlpack version: `wget http://www.mlpack.org/files/mlpack-x.y.z.tar.gz`.
3. Push tarball to lookaside cache: `fedpkg new-sources mlpack-x.y.z.tar.gz`.
4. Update spec file; build locally (`rpmbuild -ba mlpack.spec`) and make sure everything seems okay.  It may be useful to mock build for EPEL (since that can be picky sometimes).
5. Commit changes; `git commit -a`.
6. For each release of Fedora/EPEL, `fedpkg switch-branch <release>; git merge master; git push; fedpkg build; fedpkg update`.
7. When the waiting period is over for each update, push to stable (3-7 days for Fedora, 2 weeks for EPEL).

 _notes from 3.4.0 release_: this isn't automated

###### Python

See https://github.com/mlpack/mlpack-wheels, and update `BUILD_COMMIT` in `travis.yml` and `appveyor.yml`.

###### Julia

See https://github.com/mlpack/mlpack.jl, specifically the file `rel/deployment.md`.

###### R

TODO (the process is not fully devised yet)

###### Go

See https://github.com/mlpack/mlpack-go/, specifically the file `rel/deployment.md`.

###### Ubuntu/Debian

This isn't automated; it's handled downstream.

###### vcpkg

This isn't automated; it's handled downstream.

Push new benchmarks to website
------------------------------

We need to rebuild the benchmark reports so that the figure colors are right for the webpage.

1. Check out the benchmarking system to some directory.
2. Download massif.tar.gz and benchmark.db from the artifacts of the build server's [completed "benchmark - reports" job](http://big.cc.gt.atl.ga.us:8080/job/benchmark%20-%20reports/)
3. Put benchmark.db in the reports/ folder, and unpack the .mout files from massif.tar.gz into reports/etc/
4. Modify config.yaml; change chartColor to #000000, topChartColor to #000000, and textColor to #aaaaaa (the same color as the website text).
5. Run 'make reports' from the base benchmark directory.
6. Access the created file, reports/index.html, in a browser and take a look to make sure everything is okay.
7. Save the old benchmark.html on www.mlpack.org to some different file (for reference).
8. Copy index.html to www.mlpack.org:/var/www/www.mlpack.org/benchmark.html and all the support files too.
9. Use vimdiff to copy changes from benchmark-old.html to benchmark.html.  This should include loading another css stylesheet, adding the header and navigation menu to the top of the page, and removing the top graph (because it might be confusing to users who aren't sure what it means).
10. Change the version in the phrase "This page contains benchmarks for the various algorithms implemented in mlpack x.y.z."

Other notes
-----------

You can check which files have a license like this:

```
for i in `find src/ -iname '*.[hc]pp'`; do echo -n $i": "; cat $i | grep 'mlpack is free software;' | wc -l; done | grep -v ': 1' | grep -v 'arma_extend' | grep -v 'boost_backport' | grep -v 'arma_config.hpp' | grep -v 'gitversion.hpp'
```

You can get a list of all commits to master (doesn't print merge histories) like this:

```
# The first commit is the first commit since the last release.
git log --first-parent master --oneline ac73c69^..master
```

Then you can get the name of a branch to find its history:

```
# Get the name of the branch:
git show-branch | grep -A 2 'term to search for'
```

Or you can get the list of commits on Github.  In either case, then you can get a list of what to cherry-pick with (assuming branch commit `master~136^2` or similar):

```
git log --reverse --first-parent master~136^2 --oneline master~136^2~10..master~136^2
```

(where only 10 commits are shown here; you need to get the exact number from the PR or by inspecting the history) and then it is easy to take this and pipe the commits to `git cherry-pick`.