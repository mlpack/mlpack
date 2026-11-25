// These variables are set to hold state between status check updates.
def name = ''
def status = ''
def time = 0
def matrixStatus = [:]
def matrixRunning = []

def startCheck(String name, String status)
{
  // Set module-level variables that we will retain as we build.
  this.name = name
  this.status = status
  this.time = currentBuild.duration

  publishChecks(name: name,
                status: 'IN_PROGRESS',
                title: status,
                text: status,
                detailsURL: currentBuild.absoluteUrl + 'console')
}

def updateCheckStatus(String status)
{
  def stepTime = (currentBuild.duration - this.time) / 1000.0
  this.status += ' (' + stepTime.toString() + 's)\n'
  this.status += status
  this.time = currentBuild.duration

  publishChecks(name: this.name,
                status: 'IN_PROGRESS',
                title: status,
                text: this.status,
                detailsURL: currentBuild.absoluteUrl + 'console')
}

def finishCheck(String status, boolean success)
{
  def stepTime = (currentBuild.duration - this.time) / 1000.0
  this.status += ' (' + stepTime.toString() + 's)\n'
  this.status += status

  if (!success)
  {
    this.status += '\n\n' +
        '<b>Click \'view more details\' below to see failure details...</b>';
  }

  publishChecks(name: this.name,
                status: 'COMPLETED',
                conclusion: success ? 'SUCCESS' : 'FAILURE',
                title: status,
                text: this.status,
                detailsURL: currentBuild.absoluteUrl + 'testReport')
}

def prepareMatrixJobStatusString()
{
  def local_status = ''
  for (entry in this.matrixStatus)
  {
    if (local_status.size() > 0)
    {
      local_status += '; '
    }

    local_status += entry.key + ": ["
    for (it in entry.value)
    {
      if (local_status[-1] != '[')
      {
        local_status += ", "
        first = false
      }

      local_status += it
    }
    local_status += "]"
  }
  local_status += "."
  return local_status
}

def initMatrixJobStatus(String name, List<String> statuses, List<String> targets)
{
  this.name = name
  this.matrixStatus = [:]
  for (it in statuses)
  {
    this.matrixStatus[it] = []
  }
  this.matrixStatus[statuses[0]] = targets
  this.matrixRunning = targets

  def local_status = prepareMatrixJobStatusString()

  publishChecks(name: this.name,
                status: 'IN_PROGRESS',
                title: local_status,
                text: local_status,
                detailsURL: currentBuild.absoluteUrl + 'stages')
  this.status = local_status
}

def updateMatrixJobStatus(String target, String status, boolean finished)
{
  // Find whatever status the target currently has and delete it.
  for (entry in this.matrixStatus)
  {
    def index = entry.value.indexOf(target)
    if (index != -1)
    {
      this.matrixStatus[entry.key] -= target
    }
  }

  this.matrixStatus[status] += target

  if (finished)
  {
    def index = this.matrixRunning.indexOf(target)
    if (index != -1)
    {
      this.matrixRunning -= target
    }
  }

  def local_status = prepareMatrixJobStatusString()
  def job_status = 'IN_PROGRESS'
  // Set the job status correctly if we are finished.
  if (this.matrixRunning.empty)
  {
    // Check if any of the jobs failed.
    if (this.matrixStatus.containsKey("failed") &&
        !this.matrixStatus["failed"].empty)
    {
      job_status = 'FAILURE'
    }
    else
    {
      job_status = 'SUCCESS'
    }
  }

  publishChecks(name: this.name,
                status: 'IN_PROGRESS',
                title: local_status,
                text: local_status,
                detailsURL: currentBuild.absoluteUrl + 'stages')
}

return this
