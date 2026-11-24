// These variables are set to hold state between status check updates.
def name = ''
def status = ''
def time = 0
def matrixStatus = [:]

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
  this.status = ''
  for (entry in this.matrixStatus)
  {
    if (this.status.size() > 0)
    {
      this.status += '; '
    }

    this.status += entry.key + ": ["
    for (it in entry.value)
    {
      if (this.status[-1] != '[')
      {
        this.status += ", "
        first = false
      }

      this.status += it
    }
    this.status += "]"
  }
  this.status += "."
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

  prepareMatrixJobStatusString()

  publishChecks(name: this.name,
                status: 'IN_PROGRESS',
                title: this.status,
                text: this.status,
                detailsURL: currentBuild.absoluteUrl + 'stages')
}

def updateMatrixJobStatus(String target, String status)
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

  prepareMatrixJobStatusString()

  // TODO: check if we are all finished
  publishChecks(name: this.name,
                status: 'IN_PROGRESS',
                title: this.status,
                text: this.status,
                detailsURL: currentBuild.absoluteURL + 'stages')
}

return this
