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

def startCheck(String name, String status)
{
  // Set module-level variables that we will retain as we build.
  this.name = name
  this.status = status
  this.time = currentBuild.timeInMillis

  publishChecks(name: name,
                status: 'IN_PROGRESS',
                title: status,
                text: status,
                detailsURL: currentBuild.absoluteUrl + 'console')
}

def updateCheckStatus(String status)
{
  def stepTime = (currentBuild.timeInMillis - this.time) / 1000.0
  this.status += '\n' + status + ' (' + stepTime.toString() + 's)'
  this.time = currentBuild.timeInMillis

  publishChecks(name: this.name,
                status: 'IN_PROGRESS',
                title: status,
                text: this.status,
                detailsURL: currentBuild.absoluteUrl + 'console')
}

def finishCheck(String status, boolean success)
{
  def stepTime = (currentBuild.timeInMillis - this.time) / 1000.0
  this.status += '\n' + status + ' (' + stepTime.toString() + 's)'
  this.time = currentBuild.timeInMillis

  publishChecks(name: this.name,
                status: 'COMPLETED',
                conclusion: success ? 'SUCCESS' : 'FAILURE',
                title: status,
                text: this.status,
                detailsURL: currentBuild.absoluteUrl + 'testReport')
}

return this
