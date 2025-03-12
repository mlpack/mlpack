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
def setBuildStatus(Map paramsMap)
{
  // Extract arguments from the map.
  def result = paramsMap.result;
  def context = paramsMap.context;
  def successMessage = paramsMap.successMessage;
  def unstableMessage = paramsMap.unstableMessage;
  def failureMessage = paramsMap.failureMessage;

  def message = "(unknown Jenkins build result)";
  def state = "FAILURE";
  if (result == "FAILURE")
  {
    message = failureMessage;
  }
  else if (result == "UNSTABLE")
  {
    message = unstableMessage;
    state = "UNSTABLE";
  }
  else if (result == "SUCCESS")
  {
    message = successMessage;
    state = "SUCCESS";
  }
  else if (result == "ABORTED")
  {
    message = "Job aborted.";
    state = "ERROR";
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
