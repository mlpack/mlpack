# Community

mlpack is a community-led effort, and so the code is not possible without the
community.  Since mlpack is an open-source project, anyone is welcome to become
a part of the community and contribute.  There is no need to be a a machine
learning expert to participate; often, there are many tasks to be done that
don't require in-depth knowledge.

Over the past several years, mlpack has participated in
[Google Summer of Code](https://summerofcode.withgoogle.com/).  For more
information, see [this page](gsoc.md).

All mlpack development is done on [GitHub](https://github.com/mlpack/mlpack).
Commits and issue comments can be tracked via the
[mlpack-git](https://www.freelists.org/list/mlpack-git) list (graciously hosted
by [FreeLists](https://www.freelists.org).  Communication is generally either
via issues on GitHub, or via chat:

## Real-time chat

 * `#mlpack:matrix.org` on [Matrix](https://www.matrix.org/)
 * [mlpack Slack workspace](https://mlpack.slack.com/)
   - You will need to request an invite from the
     [auto-inviter](http://slack-inviter.mlpack.org:4000).
   - The Slack workspace is a bridged version of the Matrix room and it is
     generally a better experience to use Matrix directly.

## Video meetup

On the first Monday of every month, at ***1530 UTC***, we have *casual video
meetups* with no particular agenda.  Feel free to join up!  We often talk about
code changes that we are working on, issues that people are having with mlpack,
general design direction, and whatever else might be on our mind.

We use [this Zoom room](https://zoom.us/j/3820896170).  For security, we use a
password for the meeting to keep malicious bots out.  The password is simple:
it's just the name of the library (in all lowercase).

## Getting involved

Everyone is welcome to contribute to mlpack.  But before becoming a contributor,
it's often useful to understand mlpack as a user.  So, a good place to start is
to:

 - [download mlpack](https://www.mlpack.org/download.html)
 - use it in C++ (see [the documentation](../index.md))
 - use the bindings to other languages to perform machine learning tasks:
    [Python](../user/bindings/python.md), [Julia](../user/bindings/julia.md),
    [R](../user/bindings/r.md), [Go](../user/bindings/go.md), and the
    [command-line programs](../user/bindings/cli.md)

There is also the [examples repository](https://github.com/mlpack/examples) that
contains many examples you can build and play around with.

Once you have an idea of what's included in mlpack and how a user might use it,
then a good next step would be to set up a development environment.  Once you
have that set up, you can [build mlpack from source](../user/install.md), and
explore the codebase to see how it's organized.

Try making small changes to the code, or adding new tests to the test suite, and
then rebuild to see how your changes work.

Now you're set up to contribute!  There are lots of ways you can contribute.
Here are a couple ideas:

 * Help others figure out their mlpack issues and questions.
   [Here](https://github.com/mlpack/mlpack/issues?q=is%3Aopen+is%3Aissue+label%3A%22t%3A+question%22)
   is a list of GitHub issues tagged `question`.  Helping others figure out
   their problems is really one of the best ways to learn about the library.

 * Read through the [vision document](https://www.mlpack.org/papers/vision.pdf)
   to learn about the development goals of the mlpack community and see the
   high-level tasks that need to be done to accomplish that vision.

 * Find an issue that needs implementation help;
   [here](https://github.com/mlpack/mlpack/issues) is the list of issues.

 * Find an abandoned pull request;
   [here](https://github.com/mlpack/mlpack/pulls?q=is%3Aclosed+is%3Apr+label%3A%22s%3A+stale%22)
   is a list of pull requests that were closed for inactivity.  Often these have
   comments that need to be addressed, but the original author didn't have time
   to finish the work.  So, you can pick up where they left off!

 * Implement a new machine learning algorithm that mlpack doesn't currently
   have.

 * Take a look at the ideas on the Google Summer of Code
   [Ideas List](https://github.com/mlpack/mlpack/wiki/SummerOfCOdeIdeas) and see
   if you find any of them interesting or exciting.  Even if you're not planning
   to apply for Summer of Code, it's okay to take these ideas and implement them
   separately.
