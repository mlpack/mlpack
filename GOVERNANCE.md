# mlpack governance structure

Revised Oct. 21st, 2019.

## Introduction

mlpack has grown much since its initial inception as a small project out of a
university research lab.  Now that there are over 150 contributors, it's
important that we have a clearly defined process for making our decisions and
organizing ourselves.

This document aims to clarify the governance of mlpack.  This is a living
document: it may change over time.  The process for making these changes is
detailed in the "Governance Changes" section.

## Code of Conduct

mlpack aims to be an open and welcoming environment, and as such, we have a
code of conduct that helps foster this environment.  See
[here](https://github.com/mlpack/mlpack/blob/master/CODE_OF_CONDUCT.md) for
more information.

## Teams & Roles

To keep overhead minimal, mlpack's teams and roles are simple: there is only
the [Committers](https://github.com/orgs/mlpack/teams/contributors) team, and
the [NumFOCUS leadership team](TODO:link).

Members of the Committers team have commit access to all mlpack repositories
and help guide the development directions and goals of mlpack.  Committers
should be familiar with the [contribution
process](https://github.com/mlpack/mlpack/blob/master/CONTRIBUTING.md) and
follow it when merging code and reviewing pull requests; this is important for
the continued stability and quality of mlpack's codebase.  Responsibilities and
activities of Committers team members can include:

 * Welcoming new members to the community: helping support users and point
   potential contributors in the correct direction.

 * Reviewing pull requests and approving them when they are ready.

 * Merging pull requests after they have been approved by others for merge.

 * Communicating and coordinating with contributors to help get code merged and
   improve the software.

 * Helping map out mlpack's development directions and processes.

 * Maintaining mlpack infrastructure (build systems, continuous integration,
   etc.).

Membership on the Committers team does not expire.  Contributors who have
repeatedly shown that their code quality is high, demonstrated adherence to the
code of conduct, and shown that they have a strong interest in the project can
be added to the Committers team using the organizational decision process in
the next section.

The NumFOCUS leadership team is a subset of the Committers team whose
additional responsibilities are to coordinate with NumFOCUS and maintain this
governance document.  Membership in the NumFOCUS leadership team is limited to
five people, and does not confer any special voting power or decision rights.

## Voting and Organizational Decisions

Historically, mlpack organizational decisions have not been controversial and
this has allowed efficient decision making.  Therefore, a vote on a proposal is
not required unless there is any explicit disagreement or concern with the
proposal.  The topics of a proposal might be:

 * Adding/removing a new member to/from the Committers team.

 * Participating in a program such as Google Summer of Code or Outreachy.

 * A change to some part of the mlpack infrastructure or contribution process.

 * Refactoring or change of an important public part of the API.

 * Use of funds for a particular project.

That list is not inclusive.  Introducing a proposal or idea can be done
informally in a public place, such as the mlpack mailing list or on Github as
an issue.  It's a good idea (but not mandatory) to make the proposal discussion
fully public so that people who are not on the Committers team can also comment
and provide opinions---after all, this is a community-led project so we should
be sure to include the *entire* community whenever possible.

If there is any disagreement or concern with the proposal, the person who
introduced the proposal should work to try and find a resolution or compromise
if possible.  If that is not possible, then the proposal can be brought to a
vote.

For a proposal to pass, a simple majority vote suffices.  Each Committer has
one equal vote, and they may choose to abstain from voting if they do prefer.
Since some Committers may be inactive or busy, it is not required for every
Committer to participate in every vote; instead, someone who has a proposal
should make a good-faith effort to post the proposal in a public location so
that interested and active Committers can respond.  Voting for any proposal
should be open for at least five days to allow sufficient time.

If a proposal passes despite votes against it, it is generally a good idea for
the Committer who introduced the proposal to spend some time considering and
understanding the arguments that were presented against the proposal, or if
appropriate, for the Committer to try and find an acceptable compromise or
alternate strategy that addresses the given feedback.

## Governance Changes

The NumFOCUS leadership team is responsible for this governance document, and
thus any changes to this document, NumFOCUS membership, or the NumFOCUS
leadership team must be approved by that team, also by a simple majority vote.
Because every member of the NumFOCUS leadership team should be an active
Committer, efforts should be made to collect votes from all five members.
Voting for any proposal should be open for at least five days to allow
sufficient time. If necessary, every active member of the NumFOCUS leadership
team can ask for an extension of the voting deadline.
