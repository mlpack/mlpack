#include <QtNetwork>
#include "main_window.h"

MainWindow::MainWindow(QWidget *parent)
    : QWidget(parent)
{
  this->btExtractWordNetStructure_ = new QPushButton("Extract Word Net Structure");
  this->btConvertSynsetToText_ = new QPushButton("Convert binary links file to text");
  connect(this->btExtractWordNetStructure_, SIGNAL(clicked()), this, SLOT(btExtractWordNetStructureClicked()));
  connect(this->btConvertSynsetToText_, SIGNAL(clicked()), this, SLOT(btConvertSynsetToTextClicked()));

  manager_ = new QNetworkAccessManager(this);
  connect(manager_, SIGNAL(finished(QNetworkReply*)), this, SLOT(replyFinished(QNetworkReply*)));

  timer_ = new QTimer(this);
  connect(timer_, SIGNAL(timeout()), this, SLOT(timerTimeOut()));

  QVBoxLayout *layout = new QVBoxLayout;
  layout->addWidget(this->btExtractWordNetStructure_);
  layout->addWidget(this->btConvertSynsetToText_);
  this->setLayout(layout);
}

MainWindow::~MainWindow()
{

}

QStringList getWords(const QString& s)
{
  QStringList tmpList = s.split(QRegExp("[ ,]"), QString::SkipEmptyParts), sList;
  Q_FOREACH(const QString& word, tmpList)
  {
    if (!word.simplified().isEmpty()) sList << word.simplified();
  }
  return sList;
}

void MainWindow::btExtractWordNetStructureClicked()
{
  QTextStream cout(stdout);
  QString filename = QFileDialog::getOpenFileName(this, "Open Word-Net file ...", "/net/hg200/houyang/flickr", "*.txt");
  if (filename.isEmpty()) return;
  QFile file(filename);
  file.open(QIODevice::ReadOnly | QIODevice::Text);
  if (!file.isOpen())
  {
    cout << "Cannot open " << filename << endl;
    return;
  }
  QTextStream in(&file);
  synsetWords_.clear();
  wordTree_.clear();
  while (!in.atEnd())
  {
    QString line = in.readLine();
    cout << "line = " << line << endl;
    QString synsetID = line.left(9);
    QStringList lineWords = getWords(line.mid(10));
    synsetWords_[synsetID] = lineWords;
//    if (synsetWords_.size() >= 10) break;
  }
  iterator_ = synsetWords_.begin();
  n_replys_ = 0;
  total_request_ = 0;
  timer_->start(5);
}

void MainWindow::btConvertSynsetToTextClicked()
{
  loadData();
  saveText();
  QTextStream(stdout) << "done converting." << endl;
}

void MainWindow::replyFinished(QNetworkReply* reply)
{
  QTextStream cout(stdout);
  n_replys_++;
  QUrl url = reply->url();
  QString synsetId = url.toString().right(9);
  cout << " n_reply = " << n_replys_  << " total = " << total_request_
      << " id = " << synsetId << endl;
  if (reply->error())
  {
    cout << "Download of " << url.toEncoded().constData() << " failed: "
         << reply->errorString() << endl;
  }
  else
  {
    QTextStream in(reply);
    while (!in.atEnd())
    {
      QString line = in.readLine().simplified();
      if (line[0] != '-')
      {
        if (line != synsetId)
          cout << "  error line = " << line << " synsetId = " << synsetId << endl;
      }
      else
      {
        QString wordId = line.right(9);
        wordTree_.insert(wordId, synsetId);
        cout << "  " << synsetId << "  -->  " << wordId << endl;
      }
    }
  }

  if (total_request_ < 0 && n_replys_ == -total_request_)
  {
    cout << "Finished building word tree" << endl;
    saveData();
    loadData();
  }
  reply->deleteLater();
}

void MainWindow::saveData(const QString &filename)
{
  QFile file(filename);
  if (!file.open(QIODevice::WriteOnly))
  {
    QTextStream(stderr) << "Cannot open " << filename << endl;
    return;
  }
  QDataStream out(&file);
  out.setVersion(QDataStream::Qt_4_6);
  out << synsetWords_ << wordTree_;
  QTextStream(stdout) << "saved : " << synsetWords_.size() << " words with "
      << wordTree_.size() << " links" << endl;
}

void MainWindow::saveText(const QString &filename)
{
  QFile file(filename);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
  {
    QTextStream(stderr) << "Cannot open " << filename << endl;
    return;
  }
  QTextStream out(&file);
  QMultiMap<QString, QString>::const_iterator iter = wordTree_.begin();
  for (; iter != wordTree_.end(); iter++)
  {
    out << iter.key() << " " << iter.value() << endl;
  }
}

void MainWindow::loadData(const QString &filename)
{
  QFile file(filename);
  if (!file.open(QIODevice::ReadOnly))
  {
    QTextStream(stderr) << "Cannot open " << filename << endl;
    return;
  }
  QDataStream in(&file);
  in.setVersion(QDataStream::Qt_4_6);
  synsetWords_.clear();
  wordTree_.clear();
  QTextStream(stdout) << "begin loading: " << synsetWords_.size() << " words with "
      << wordTree_.size() << " links" << endl;
  in >> synsetWords_ >> wordTree_;
  QTextStream(stdout) << "loaded : " << synsetWords_.size() << " words with "
      << wordTree_.size() << " links" << endl;
}

void MainWindow::timerTimeOut()
{
//  if (manager_->)
  if (iterator_ == synsetWords_.end()) return;

  QString synsetID = iterator_.key();
  // http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=n00001740
  QUrl url("http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid="+synsetID);
  QNetworkRequest request(url);
  manager_->get(request);
  total_request_++;

  iterator_++;
  if (iterator_ == synsetWords_.end())
  {
    total_request_ = -total_request_;
    timer_->stop();
  }
}
