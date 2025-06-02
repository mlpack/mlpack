// These variables are set to hold state between status check updates.
def name = ''
def status = ''
def time = 0

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

return this
