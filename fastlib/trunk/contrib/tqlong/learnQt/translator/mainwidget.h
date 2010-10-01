#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QtGui>
#include "browser_translate.h"

class FindTextDialog;

class MainWidget : public QWidget
{
  Q_OBJECT
public slots:
  void btTranslateClicked();
  void btOKClicked();
  void btBookmarkClicked();
  void btNewTabClicked();
  void pageUrl(QUrl);
  void tabChanged(int);
  void tabClosed(int);
  void autoCheckBoxChanged(bool);
  void findTextReceived(QString);
  void findAllTextReceived(QString);
  void pageLoadProgress(int progress);
  void loadBookmarks();
public:
  MainWidget(QWidget *parent = 0);
  ~MainWidget();
  BrowserTranslate* addPage();
private:
  QPushButton *btTranslate, *btOK, *btNewTab;
  QLineEdit *addressEdit;
  QCheckBox *autoTranslateCheck_;
  QToolBar *bookmarks_;
  QTabWidget *tabs_;
  QSet<BrowserTranslate*> browsers_;
  FindTextDialog* findDialog_;
//  BrowserTranslate* browser;
protected:
  void closeEvent(QCloseEvent* event);
  void keyPressEvent(QKeyEvent* event);
  void addBookmark(const QString& title, const QUrl& url);
};

class FindTextDialog : public QWidget
{
  Q_OBJECT
public slots:
  void btFindClicked()
  {
    if (!textEdit_->text().isEmpty())
      emit findText(textEdit_->text());
  }
signals:
  void findText(QString);
  void findAllText(QString);
public:
  explicit FindTextDialog(QWidget* parent = 0) : QWidget(parent)
  {
    textEdit_ = new QLineEdit;
    btFind_ = new QPushButton("Find");
    btCancel_ = new QPushButton("Cancel");

    connect(textEdit_, SIGNAL(textChanged(QString)), this, SIGNAL(findAllText(QString)));
    connect(textEdit_, SIGNAL(returnPressed()), this, SLOT(btFindClicked()));
    connect(btFind_, SIGNAL(clicked()), this, SLOT(btFindClicked()));
    connect(btCancel_, SIGNAL(clicked()), this, SLOT(hide()));

    QHBoxLayout* layout = new QHBoxLayout;
    layout->addWidget(textEdit_);
    layout->addWidget(btFind_);
    layout->addWidget(btCancel_);

    setLayout(layout);
  }
private:
  QLineEdit* textEdit_;
  QPushButton *btFind_, *btCancel_;
};

class BookmarkButton : public QPushButton
{
  QUrl url_;
public:
  explicit BookmarkButton(const QString& text = QString(), const QUrl& url = QUrl(), QWidget* parent = 0)
    : QPushButton(text.left(30), parent), url_(url)
  {
  }
  const QUrl& getUrl() const { return url_; }
};

#endif // MAINWIDGET_H
