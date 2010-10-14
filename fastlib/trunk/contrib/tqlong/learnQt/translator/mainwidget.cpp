#include <QtGui>
#include "mainwidget.h"

MainWidget::MainWidget(QWidget *parent)
    : QWidget(parent)
{
  Translator::loadDictionary();

  btTranslate = new QPushButton("Translate");
  btOK = new QPushButton("Go");
  btNewTab = new QPushButton("+");
  addressEdit = new QLineEdit;
  autoTranslateCheck_ = new QCheckBox("Auto translate");
  bookmarks_ = new QToolBar("Bookmarks");
  bookmarks_->hide();
  loadBookmarks();
  tabs_ = new QTabWidget;
  tabs_->setTabsClosable(true);
  sizeSlider_ = new SizeSlider;
  sizeSlider_->setRange(0, 50);
  BrowserTranslate* first = addPage();
  first->load(QUrl::fromUserInput("http://www.google.com"));
  findDialog_ = new FindTextDialog();
  findDialog_->setWindowTitle("Find ...");
//  findDialog_->show();

  connect(btTranslate, SIGNAL(clicked()), this, SLOT(btTranslateClicked()));
  connect(btOK, SIGNAL(clicked()), this, SLOT(btOKClicked()));
  connect(btNewTab, SIGNAL(clicked()), this, SLOT(btNewTabClicked()));
  connect(addressEdit, SIGNAL(returnPressed()), this, SLOT(btOKClicked()));
  connect(tabs_, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));
  connect(tabs_, SIGNAL(tabCloseRequested(int)), this, SLOT(tabClosed(int)));
  connect(autoTranslateCheck_, SIGNAL(toggled(bool)), this, SLOT(autoCheckBoxChanged(bool)));
  connect(findDialog_, SIGNAL(findText(QString)), this, SLOT(findTextReceived(QString)));
  connect(findDialog_, SIGNAL(findAllText(QString)), this, SLOT(findAllTextReceived(QString)));
  connect(sizeSlider_, SIGNAL(valueChanged(int)), this, SLOT(changeSize(int)));
  connect(this, SIGNAL(changeFontSize(int)), sizeSlider_, SLOT(needChanged(int)));

  QHBoxLayout *toolbar = new QHBoxLayout;
  toolbar->addWidget(btNewTab);
  toolbar->addWidget(addressEdit);
  toolbar->addWidget(btOK);
  toolbar->addWidget(btTranslate);
  toolbar->addWidget(autoTranslateCheck_);
  QVBoxLayout *mainLayout = new QVBoxLayout;
  mainLayout->addLayout(toolbar);
  mainLayout->addWidget(bookmarks_);
  mainLayout->addWidget(tabs_);
  QHBoxLayout *mainLayout2 = new QHBoxLayout;
  mainLayout2->addWidget(sizeSlider_);
  mainLayout2->addLayout(mainLayout);
  setLayout(mainLayout2);
  this->setMinimumWidth(1024);
}

void MainWidget::btTranslateClicked()
{
  BrowserTranslate* browser = (BrowserTranslate*) tabs_->currentWidget();
  if (browser)
    browser->translateToVietnamese();
}

MainWidget::~MainWidget()
{
}

void MainWidget::btNewTabClicked()
{
  BrowserTranslate* browser = addPage();
  browser->load(QUrl::fromUserInput("www.google.com"));
  tabs_->setCurrentWidget(browser);
}

void MainWidget::btOKClicked()
{
  BrowserTranslate* browser = (BrowserTranslate*) tabs_->currentWidget();
  QTextStream(stdout) << "OK clicked\n";
  QUrl url = QUrl::fromUserInput(addressEdit->text());
  if (!url.isValid())
  {
    QTextStream(stdout) << "invalid URL:" << addressEdit->text() << " --> " << url.toString() << endl;
    return;
  }
  QTextStream(stdout) << "valid URL:" << addressEdit->text() << " --> " << url.toString() << endl;
  addressEdit->setText(url.toString());
  if (!browser)
    browser = addPage();
  browser->load(url);
}

void MainWidget::pageUrl(QUrl url)
{
  BrowserTranslate* browser = (BrowserTranslate*) tabs_->currentWidget();
  if (browser && browser == sender())
  {
    QTextStream(stdout) << "url changed: current = " << browser->page()->mainFrame()->baseUrl().toString() << endl;
    addressEdit->setText(url.toString());
  }
}

BrowserTranslate* MainWidget::addPage()
{
  BrowserTranslate* browser = new BrowserTranslate(this, this);
  connect(browser, SIGNAL(urlChanged(QUrl)), this, SLOT(pageUrl(QUrl)));
  connect(browser, SIGNAL(loadProgress(int)), this, SLOT(pageLoadProgress(int)));
  browser->setAutoTranslate(autoTranslateCheck_->isChecked());
//  browser->load(url);//, autoTranslateCheck_->isChecked());
  browsers_ << browser;
  tabs_->addTab(browser, "translator");
  return browser;
}

void MainWidget::tabChanged(int)
{
  BrowserTranslate* browser = (BrowserTranslate*)tabs_->currentWidget();
  addressEdit->setText(browser ? browser->page()->mainFrame()->baseUrl().toString() : "");
  emit changeFontSize(browser->page()->settings()->fontSize(QWebSettings::MinimumFontSize));
}

void MainWidget::pageLoadProgress(int progress)
{
  BrowserTranslate* browser = (BrowserTranslate*) sender();
  int index = tabs_->indexOf(browser);
  tabs_->setTabText(index, QString::number(progress) + " translator");
//  addressEdit->setText(browser ? browser->page()->mainFrame()->baseUrl().toString() : "");
}

void MainWidget::autoCheckBoxChanged(bool checked)
{
  QTextStream(stdout) << "set auto translate to " << (checked?"true":"false") << endl;
  Q_FOREACH(BrowserTranslate* browser, browsers_)
  {
    browser->setAutoTranslate(checked);
  }
}

void MainWidget::tabClosed(int index)
{
  BrowserTranslate* browser = (BrowserTranslate*) tabs_->widget(index);
  if (!browser) return;
  browsers_.remove(browser);
  browser->markForDelete();
}

void MainWidget::findTextReceived(QString s)
{
  BrowserTranslate* browser = (BrowserTranslate*) tabs_->currentWidget();
  if (browser)
    browser->findText(s, QWebPage::FindWrapsAroundDocument);
}

void MainWidget::findAllTextReceived(QString s)
{
  BrowserTranslate* browser = (BrowserTranslate*) tabs_->currentWidget();
  if (browser)
  {
    browser->findText("", QWebPage::HighlightAllOccurrences);
    browser->findText(s, QWebPage::HighlightAllOccurrences);
  }
}

void MainWidget::closeEvent(QCloseEvent* event)
{
  findDialog_->close();
  event->accept();
}

void MainWidget::keyPressEvent(QKeyEvent* event)
{
  if (event->key() == Qt::Key_F && event->modifiers() == Qt::ControlModifier)
    findDialog_->show();
  if (event->key() == Qt::Key_D && event->modifiers() == Qt::AltModifier)
  {
    addressEdit->setFocus();
    addressEdit->setSelection(0, addressEdit->text().length()-1);
  }
  int index = tabs_->currentIndex();
  BrowserTranslate* browser = (BrowserTranslate*) tabs_->currentWidget();
  if (event->key() == Qt::Key_F5 && event->modifiers() == Qt::NoModifier)
  {
    if (browser)
      browser->reload();
  }
  if (event->key() == Qt::Key_Left && event->modifiers() == Qt::AltModifier)
  {
    if (browser)
      browser->back();
  }
  if (event->key() == Qt::Key_Right && event->modifiers() == Qt::AltModifier)
  {
    if (browser)
      browser->forward();
  }
  if (event->key() == Qt::Key_D && event->modifiers() == Qt::ControlModifier)
  {
    if (!browser) return;
    QUrl url = browser->page()->mainFrame()->baseUrl();
    QString title = Translator::translate(browser->title());
    addBookmark(title, url);

    QFile file("bookmarks.txt");
    file.open(QIODevice::Text | QIODevice::Append);
    QTextStream out(&file);
    out << title << endl << url.toEncoded() << endl;
  }
  if (event->key() == Qt::Key_W && event->modifiers() == Qt::ControlModifier)
  {
    tabs_->removeTab(index);
    if (!browser) return;
    browsers_.remove(browser);
    browser->markForDelete();
  }
  if (event->key() == Qt::Key_T && event->modifiers() == Qt::ControlModifier)
  {
    this->btNewTabClicked();
  }
  if ((event->key() == Qt::Key_Plus||event->key() == Qt::Key_Equal) && event->modifiers() == Qt::ControlModifier)
  {
    if (browser)
    {
      int size = browser->page()->settings()->fontSize(QWebSettings::MinimumFontSize);
      changeSize(size+1);
    }
  }
  if ((event->key() == Qt::Key_Minus||event->key()==Qt::Key_Underscore) && event->modifiers() == Qt::ControlModifier)
  {
    if (browser)
    {
      int size = browser->page()->settings()->fontSize(QWebSettings::MinimumFontSize);
      size = size > 0 ? size : 1;
      changeSize(size-1);
    }
  }
  if (event->key() == Qt::Key_PageUp && event->modifiers() == Qt::ControlModifier)
  {
    tabs_->setCurrentIndex((tabs_->currentIndex()+1)%tabs_->count());
  }
  if (event->key() == Qt::Key_PageDown && event->modifiers() == Qt::ControlModifier)
  {
    int index = (tabs_->currentIndex()-1);
    if (index < 0) index = tabs_->count()-1;
    tabs_->setCurrentIndex(index);
  }
}

void MainWidget::changeSize(int size)
{
  BrowserTranslate* browser = (BrowserTranslate*) tabs_->currentWidget();
  if (size < 0 || size > 50 || !browser) return;
  browser->page()->settings()->setFontSize(QWebSettings::MinimumFontSize, size);
  QTextStream(stdout) << "minimum font size = " << size << endl;
  emit changeFontSize(size);
}


void MainWidget::btBookmarkClicked()
{
  BookmarkButton* button = (BookmarkButton*) sender();
  BrowserTranslate* browser = (BrowserTranslate*) tabs_->currentWidget();
  if (!browser)
    browser = addPage();
  browser->load(button->getUrl());
}

void MainWidget::loadBookmarks()
{
  QFile file("bookmarks.txt");
  file.open(QIODevice::Text | QIODevice::ReadOnly);
  if (!file.isOpen()) return;
  QTextStream in(&file);
  while (!in.atEnd())
  {
    QString title = in.readLine();
    QString urlStr = in.readLine();
    QTextStream(stdout) << "bookmark: title = " << title << " url = " << urlStr << endl;
    addBookmark(title, QUrl(urlStr));
  }
}

void MainWidget::addBookmark(const QString& title, const QUrl& url)
{
  BookmarkButton* button = new BookmarkButton(title, url, this);
  bookmarks_->addAction(title.left(20), button, SIGNAL(clicked()));
  //bookmarks_->addWidget(button);
  connect(button, SIGNAL(clicked()), this, SLOT(btBookmarkClicked()));
  bookmarks_->show();
}
