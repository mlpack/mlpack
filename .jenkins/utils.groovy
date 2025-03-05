// A simple utility to mark the build as pending on Github.
def startBuild(String context)
{
  step([
      $class: "GitHubCommitStatusSetter",
      reposSource: [$class: "ManuallyEnteredRepositorySource",
                    url: "https://github.com/mlpack/mlpack"],
      contextSource: [$class: "ManuallyEnteredCommitContextSource",
                      context: context ],
      errorHandlers: [[$class: "ChangingBuildStatusErrorHandler",
                       result: "UNSTABLE"]],
      statusResultSource: [$class: "ConditionalStatusResultSource",
                           results: [[$class: "AnyBuildResult",
                                      message: "Building...",
                                      state: "PENDING"]]]
  ]);
}

// A simple utility to set the build status on Github for a commit.
def setBuildStatus(String currentResult,
                   String context,
                   String successMessage,
                   String unstableMessage,
                   String failureMessage)
{
  String message = "(unknown Jenkins build result)";
  String state = "FAILURE";
  if (currentResult == "FAILURE")
  {
    message = failureMessage;
  }
  else if (currentResult == "UNSTABLE")
  {
    message = unstableMessage;
    state = "UNSTABLE";
  }
  else if (currentResult == "SUCCESS")
  {
    message = successMessage;
    state = "SUCCESS";
  }

  step([
      $class: "GitHubCommitStatusSetter",
      reposSource: [$class: "ManuallyEnteredRepositorySource",
                    url: "https://github.com/mlpack/mlpack"],
      contextSource: [$class: "ManuallyEnteredCommitContextSource",
                      context: context ],
      errorHandlers: [[$class: "ChangingBuildStatusErrorHandler",
                       result: "UNSTABLE"]],
      statusResultSource: [$class: "ConditionalStatusResultSource",
                           results: [[$class: "AnyBuildResult",
                                      message: message,
                                      state: state]]]
  ]);
}

return this
