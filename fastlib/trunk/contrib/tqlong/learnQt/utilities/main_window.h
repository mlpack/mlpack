#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtGui>
#include <QtNetwork>

class MainWindow : public QWidget
{
  Q_OBJECT

  QPushButton *btExtractWordNetStructure_, *btConvertSynsetToText_;
  QNetworkAccessManager *manager_;
  QMultiMap<QString, QString> wordTree_;
  QMap<QString, QStringList> synsetWords_;
  QMap<QString, QStringList>::const_iterator iterator_;
  QTimer *timer_;
  int n_replys_, total_request_;
public slots:
  void btExtractWordNetStructureClicked();
  void btConvertSynsetToTextClicked();
  void replyFinished(QNetworkReply*);
  void timerTimeOut();
public:
  void saveData(const QString& filename = "synset");
  void loadData(const QString& filename = "synset");
  void saveText(const QString& filename = "tree.txt");
public:
  MainWindow(QWidget *parent = 0);
  ~MainWindow();
};

#endif // MAIN_WINDOW_H
