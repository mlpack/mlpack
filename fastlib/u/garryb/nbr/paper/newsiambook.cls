%% Revised 2001/10/10 to correct algorithms: xreferencing,numbering
%% Revised 2002/02/19 to correct optional title in algorithms
%% Revised 2002/05/01 to force footnotes to number consecutively throughout book
%% Revised 2003/10/21 added example environment, like theorem
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{newsiambook}
              [2003/10/22 v1.9
 Customized LaTeX document class]
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright
\newif\if@mainmatter \@mainmattertrue
\if@compatibility\else
\newif\if@onethmnum
\@onethmnumfalse
%_%_%_%_% remove theorem numbering from class???
\newif\if@mytheorems
\@mytheoremsfalse


%%%%add formatted authors to chapter opener...
\newenvironment{authorline}{\let\thanks\footnote\renewcommand{\thefootnote}{\fnsymbol{footnote}}%
\Large\bfseries\itshape}{\vskip36pt\setcounter{footnote}{0}}

%%%%add author names to table of contents...
\newcommand{\authortoc}[1]{\addtocontents{toc}{\hspace*{-.5em}{\bfseries\itshape #1}}}

%_%_%_%default "plain" chapter opener...no graphic
\DeclareOption{plain-opener}{%
\def\@makechapterhead#1{%
   {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter     
%
\noindent\begin{picture}(360,144)
\put(0,22.5){\setlength{\fboxsep}{0pt}%\fbox
{\parbox[b][99pt][c]{300pt}{\raggedright\sffamily\Large\bfseries%
\@chapapp\space \thechapter
        \vskip3pt\par\nobreak
    \Huge \bfseries #1\par}}}%%end of parbox and put
\end{picture}
 \fi
    \fi
    \interlinepenalty\@M}   \vskip 40\p@
  }%
\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\hbox to 20pt{\thepart}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\centering
     \interlinepenalty \@M
     \normalfont
    %_%_%_%_%add part background graphic...
 \ifnum \c@secnumdepth >-2\relax
       \sffamily\huge\bfseries \partname~\thepart
       \par
       \vskip 20\p@
     \fi
     \sffamily\Huge \bfseries #2\par}%
    \@endpart}}
\DeclareOption{opener-d}{%
%_%_%_%_% new makechapterhead
%_%_%_%_% type "D"
%_%_%_%_%
\def\@makechapterhead#1{%
   {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter     
%
\noindent\begin{picture}(360,144)
\put(0,11){\includegraphics[height=120pt]{macroa-1-gray.eps}}
\put(72,22.5){\setlength{\fboxsep}{0pt}%\fbox
{\parbox[b][99pt][c]{300pt}{\raggedright\sffamily\Large\bfseries%
\@chapapp\space \thechapter
        \vskip3pt\par\nobreak
    \Huge \bfseries #1\par}}}%%end of parbox and put
\end{picture}
 \fi
    \fi
    \interlinepenalty\@M}   \vskip 40\p@
  }%
\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\hbox to 20pt{\thepart}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
         \interlinepenalty \@M
     \normalfont
    %_%_%_%_%add part background graphic...
 \ifnum \c@secnumdepth >-2\relax
\noindent\begin{picture}(360,144)
\put(0,11){\includegraphics[height=120pt]{macroa-1-gray.eps}}
\put(72,22.5){\setlength{\fboxsep}{0pt}%\fbox
{\parbox[b][99pt][c]{300pt}{\parindent0pt\sffamily\Large\bfseries%
\partname\space \thepart
        \vskip3pt\par\nobreak
    \Huge \bfseries #2\par}}}%%end of parbox and put
\end{picture}
     \fi
    \@endpart}}
\DeclareOption{opener-c}{%

%_%_%_%_% new makechapterhead
%_%_%_%_% type "C"
%_%_%_%_%
\def\@makechapterhead#1{%
   {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter     
%
\noindent\begin{picture}(360,144)
\put(0,0){\includegraphics[width=114pt]{macroa2-gray.eps}}
\put(118,16){\vbox to 99pt{\hsize247pt\raggedright\sffamily\Large\bfseries%
\vfill\@chapapp\space \thechapter
        \vskip3pt\par\nobreak
    \huge \bfseries #1\par\vfill}}%%end of parbox and put
\end{picture}
 \fi
    \fi
    \interlinepenalty\@M}   \vskip 40\p@
  }%
\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\hbox to 20pt{\thepart}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
         \interlinepenalty \@M
     \normalfont
    %_%_%_%_%add part background graphic...
 \ifnum \c@secnumdepth >-2\relax
\noindent\begin{picture}(360,144)
\put(0,0){\includegraphics[width=114pt]{macroa2-gray.eps}}
\put(118,16){\vbox to 99pt{\hsize247pt\parindent0pt\sffamily\Large\bfseries%
\vfill\partname\space \thepart
        \vskip3pt\par\nobreak
    \huge \bfseries #2\par\vfill}}%%end of parbox and put
\end{picture}
     \fi
    \@endpart}}
\DeclareOption{opener-b}{%
%_%_%_%_% new makechapterhead
%_%_%_%_% type "B"
%_%_%_%_%
\def\@makechapterhead#1{%
   {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter     
%
\noindent\begin{picture}(360,144)
\put(0,0){\includegraphics[width=114pt]{macroa2-gray.eps}}
\put(118,16){\vbox to 99pt{\hsize247pt\raggedright\sffamily\Large\bfseries%
\@chapapp\space \thechapter
        \vskip3pt\par\vfill\nobreak
    \huge \bfseries #1\par}}%%end of parbox and put
\end{picture}
 \fi
    \fi
    \interlinepenalty\@M}   \vskip 40\p@
  }%
\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\hbox to 20pt{\thepart}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
         \interlinepenalty \@M
     \normalfont
    %_%_%_%_%add part background graphic...
 \ifnum \c@secnumdepth >-2\relax
\noindent\begin{picture}(360,144)
\put(0,0){\includegraphics[width=114pt]{macroa2-gray.eps}}
\put(118,16){\vbox to 99pt{\parindent0pt\hsize247pt\sffamily\Large\bfseries%
\partname\space \thepart
        \vskip3pt\par\vfill\nobreak
    \huge \bfseries #2\par}}%%end of parbox and put
\end{picture}
     \fi
    \@endpart}}

\DeclareOption{opener-a}{%
%_%_%_%_% new makechapterhead...
%_%_%_%_% type "A"
%_%_%_%_%
\def\@makechapterhead#1{%
   {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter     
%
\noindent\begin{picture}(360,144)
\put(0,-1){\includegraphics[height=102pt]{macroa-gray.eps}}
\put(118,0){\vbox to 100pt{\hsize247pt\raggedright\sffamily\Large\bfseries%
\@chapapp\space \thechapter
        \vfill\par\nobreak
    \huge \bfseries #1\par}}%%end of parbox and put
\end{picture}
 \fi
    \fi
    \interlinepenalty\@M}   \vskip 40\p@
  }%
\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{10em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
         \interlinepenalty \@M
     \normalfont
    %_%_%_%_%add part background graphic...
 \ifnum \c@secnumdepth >-2\relax
\noindent\begin{picture}(360,144)
\put(0,-1){\includegraphics[height=102pt]{macroa-gray.eps}}
\put(118,0){\vbox to 100pt{\parindent0pt\hsize247pt\sffamily\Large\bfseries%
\partname\space \thepart
        \vfill\par\nobreak
    \huge \bfseries #2\par}}%%end of parbox and put
\end{picture}
     \fi
    \@endpart}}
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility
\@openrighttrue
\else
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
\DeclareOption{onethmnum}{\@onethmnumtrue} % added 7.29.96
\DeclareOption{mytheorems}{\@mytheoremstrue} %_%_%_%_% remove theorem defs from class????
\ExecuteOptions{plain-opener,letterpaper,10pt,twoside,onecolumn,final,openright}
\ProcessOptions
\input{newsiambk1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.9}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.1}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.1}
\renewcommand\floatpagefraction{.91}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{{\rule[-5pt]{\textwidth}{1pt}}\hspace*{-\textwidth}\sffamily\thepage\hfil\leftmark}%
      \def\@oddhead{{\rule[-5pt]{\textwidth}{1pt}}\hspace*{-\textwidth}\sffamily{\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markboth {{%%%%%was \MakeUppercase
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}{}}%
    \def\sectionmark##1{%
      \markright {{%%%%%was \MakeUppercase
        \ifnum \c@secnumdepth >\z@
          \thesection. \ %
        \fi
        ##1}}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\sffamily\rightmark}\hfil\thepage}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}}}
\fi
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\sffamily\thepage\hfil\sffamily\leftmark}%
    \def\@oddhead{{\sffamily\rightmark}\hfil\sffamily\thepage}%
    \let\@mkboth\@gobbletwo
    \let\chaptermark\@gobble
    \let\sectionmark\@gobble
    }
\def\ps@plain{\let\@mkboth\@gobbletwo
     \let\@oddhead\@empty\def\@oddfoot{\reset@font\hfil\sffamily\thepage
     \hfil}\let\@evenhead\@empty\let\@evenfoot\@oddfoot}
  \if@titlepage
  \newcommand\maketitle{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \vskip 60\p@
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 3em%
    {\large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
      \vskip 1.5em%
    {\large \@date \par}%       % Set date in \large size.
  \end{center}\par
  \@thanks
  \vfil\null
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\else
\newcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}
\fi
\newcommand*\chaptermark[1]{}
\setcounter{secnumdepth}{2}
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thechapter {\@arabic\c@chapter}
\renewcommand \thesection {\thechapter.\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\@chapapp{\chaptername}
\newcommand\frontmatter{%
    \cleardoublepage
  \@mainmatterfalse
  \pagenumbering{roman}}
\newcommand\mainmatter{%
    \cleardoublepage
  \@mainmattertrue
  \pagenumbering{arabic}}
\newcommand\backmatter{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \@mainmatterfalse}
\newcommand\part{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \thispagestyle{plain}%
  \if@twocolumn
    \onecolumn
    \@tempswatrue
  \else
    \@tempswafalse
  \fi
  \null%%%%%\vfil
\vspace*{72pt}
  \secdef\@part\@spart}



\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \sffamily\Huge \bfseries #1\par}%
    \@endpart}
\def\@endpart{\vfil\newpage
              \if@twoside
                \null
                \thispagestyle{empty}%
                \newpage
              \fi
              \if@tempswa
                \twocolumn
              \fi}
\newcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
%_%_%_%_%
%_%_%_%_% add third parameter to \chapter to include background graphic
%_%_%_%_%
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
%_%_%_%_%
%_%_%_%_% new definition for chapter-page quote
%_%_%_%_% optional argument moves environment right, in order to center
\newenvironment{chapterquote}[1][0pt]{\normalsize\normalfont\itshape%
\advance\leftskip#1}{\par}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M\centering
    \Huge \sffamily\bfseries  #1\par\nobreak
    \vskip 40\p@
  }}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {1.5ex \@plus.0ex}%
                                   {\raggedright\normalfont\fontsize{14}{16pt}\selectfont\sffamily\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\raggedright\normalfont\large\sffamily\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\sffamily\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\sffamily\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\sffamily\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\if@compatibility
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
\newcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \renewcommand\@chapapp{\appendixname}%
  \renewcommand\thechapter{\@Alph\c@chapter}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset {equation}{chapter}
\renewcommand\theequation
  {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@equation}
\newcounter{figure}[chapter]
\renewcommand \thefigure
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\def\xtable{table}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\textbf{#1.} \itshape#2}%
  \ifdim \wd\@tempboxa >\hsize
    \hspace*{3pc}{\textbf{#1.} }\itshape#2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
\ifx\@captype\xtable
\vskip6pt
\else
\fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{15pt}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{2}
\newcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{%
           \contentsname}{\contentsname}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \normalsize \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}
\newcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{2.5em}%was 1.5em
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}


\newcommand*\l@section{\@dottedtocline{1}{2.5em}{3.3em}}
\newcommand*\l@subsection{\@dottedtocline{2}{5.8em}{5.2em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{10em}{5em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{12em}{6em}}
\newcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\listfigurename}%
              {\listfigurename}}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }
%_%_%_%_%
%_%_%_%_%make list of algorithms
%_%_%_%_%
\newcommand*\l@algorithm{\@dottedtocline{1}{0pt}{84pt}}
\newcommand\listofalgorithms{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{List of Algorithms
      \@mkboth{{List of Algorithms}}%
              {{List of Algorithms}}}%
    \@starttoc{loa}%
    \if@restonecol\twocolumn\fi
    }

\newcounter{algorithm}[chapter]
\renewcommand{\thealgorithm}{\arabic{chapter}.\arabic{algorithm}}
\newenvironment{algorithm}[1][\relax]{\addtocounter{algorithm}{1}%
\addcontentsline{loa}{algorithm}%
    {\protect\numberline{Algorithm~\thealgorithm}{\ignorespaces#1}}%
\par\vspace{1\baselineskip}%
\expandafter\ifx#1\relax
\parindent0pt {\scshape\bfseries Algorithm~\thealgorithm.}\\%%%
\else
\parindent0pt {\scshape\bfseries Algorithm~\thealgorithm.}\enspace{\bfseries#1.}\\%%%
\fi}
{\vspace{1\baselineskip}\par}
%_%_%_%_%
%_%_%_%_%
%_%_%_%_%
%%%\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{3.3em}}
\newcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{%
          \listtablename}{\listtablename}}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\chapter*{\bibname
        \@mkboth{\bibname}{\bibname}}%
\addcontentsline{toc}{chapter}{\bibname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\@makeschapterhead{\indexname}]%
		\addcontentsline{toc}{chapter}{\indexname}%
                \@mkboth{\indexname}%
                        {\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width6pc
  \kern2.6\p@}
%%%% 05/01/02 deleted to change numbering \@addtoreset{footnote}{chapter}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent\hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\bibname{Bibliography}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\chaptername{Chapter}
\newcommand\appendixname{Appendix}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{headings}
\pagenumbering{arabic}
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi


%_%_%_%_%
%_%_%_%_% borrowed from siamltex.cls...
%_%_%_%_%

\def\@begintheorem#1#2{\vskip-\lastskip\par\vskip12pt\par%
\bgroup\noindent{\bfseries #1\ #2. }\it\ignorespaces}
%
\def\@opargbegintheorem#1#2#3{\vskip-\lastskip\par\vskip12pt\par\bgroup%
   \noindent{\bfseries #1\ #2\ ({\upshape #3}). }\it\ignorespaces}
%
\def\@endtheorem{\egroup\vskip12pt}
%


%%% create theorems, etc. with upright font, no italic
\def\newtheoremup#1{%
  \@ifnextchar[{\@othmup{#1}}{\@nthmup{#1}}}
\def\@nthmup#1#2{%
  \@ifnextchar[{\@xnthmup{#1}{#2}}{\@ynthmup{#1}{#2}}}
\def\@xnthmup#1#2[#3]{%
  \expandafter\@ifdefinable\csname #1\endcsname
    {\@definecounter{#1}\@newctr{#1}[#3]%
     \expandafter\xdef\csname the#1\endcsname{%
       \expandafter\noexpand\csname the#3\endcsname \@thmcountersep
          \@thmcounterup{#1}}%
     \global\@namedef{#1}{\@thmup{#1}{#2}}%
     \global\@namedef{end#1}{\@endtheoremup}}}
\def\@ynthmup#1#2{%
  \expandafter\@ifdefinable\csname #1\endcsname
    {\@definecounter{#1}%
     \expandafter\xdef\csname the#1\endcsname{\@thmcounterup{#1}}%
     \global\@namedef{#1}{\@thmup{#1}{#2}}%
     \global\@namedef{end#1}{\@endtheoremup}}}
\def\@othmup#1[#2]#3{%
  \@ifundefined{c@#2}{\@nocounterr{#2}}%
    {\expandafter\@ifdefinable\csname #1\endcsname
    {\global\@namedef{the#1}{\@nameuse{the#2}}%
  \global\@namedef{#1}{\@thmup{#2}{#3}}%
  \global\@namedef{end#1}{\@endtheoremup}}}}
\def\@thmup#1#2{%
  \refstepcounter{#1}%
  \@ifnextchar[{\@ythmup{#1}{#2}}{\@xthmup{#1}{#2}}}
\def\@xthmup#1#2{%
  \@begintheoremup{#2}{\csname the#1\endcsname}\ignorespaces}
\def\@ythmup#1#2[#3]{%
  \@opargbegintheoremup{#2}{\csname the#1\endcsname}{#3}\ignorespaces}
\def\@thmcounterup#1{\noexpand\arabic{#1}}
\def\@thmcountersep{.}
\def\@begintheoremup#1#2{\trivlist
   \item[\hskip \labelsep{\bfseries #1\ #2}]\upshape}
\def\@opargbegintheoremup#1#2#3{\trivlist
      \item[\hskip \labelsep{\bfseries #1\ #2\ (#3)}]\upshape}
\def\@endtheoremup{\endtrivlist}


\newlength{\proofboxwd}
\setlength{\proofboxwd}{1pt}

\def\proofbox{\hspace{12\proofboxwd}\vbox{\hrule height0.6\proofboxwd\hbox{%
   \vrule height1.3ex width0.6\proofboxwd\hskip0.8ex
   \vrule width0.6\proofboxwd}\hrule height0.6\proofboxwd
  }}

\def\examplebox{\hspace{12\proofboxwd}\rule{5.36pt}{7.91pt}}


\def\tempproofbox{\hspace{12\proofboxwd}\vbox{\hrule height0.6\proofboxwd\hbox{%
   \vrule height1.3ex width0.6\proofboxwd\hskip0.8ex
   \vrule width0.6\proofboxwd}\hrule height0.6pt
  }}


\def\myproofbox{\tempproofbox\global\setlength{\proofboxwd}{0pt}}

\newenvironment{proof}{%
\vskip-\lastskip\par
\vskip12pt\par\noindent{\bfseries\itshape Proof.} \ignorespaces}%
{\proofbox\vspace{1\baselineskip}\global\setlength{\proofboxwd}{1pt}}


\if@mytheorems
\else
\if@onethmnum
  \newtheorem{theorem}{Theorem}
  \newtheorem{lemma}[theorem]{Lemma}
  \newtheorem{corollary}[theorem]{Corollary}
  \newtheorem{proposition}[theorem]{Proposition}
  \newtheorem{definition}[theorem]{Definition}
  \newtheoremup{example}[theorem]{Example}
\else
  \newtheorem{theorem}{Theorem}[chapter]
  \newtheorem{lemma}[theorem]{Lemma}
  \newtheorem{corollary}[theorem]{Corollary}
  \newtheorem{proposition}[theorem]{Proposition}
  \newtheorem{definition}[theorem]{Definition}
  \newtheoremup{example}[theorem]{Example}
\fi
\let\tempendexample\@endtheoremup
\def\endexample{\examplebox\tempendexample}
\fi


\newcounter{rmnum}
\newenvironment{romannum}
               {\begin{list}{{\hfill\upshape (\roman{rmnum})}}{\usecounter{rmnum}
                \setlength{\leftmargin}{24pt}\labelwidth24pt\labelsep.5em%
		\itemsep2pt\parsep0pt
                \setlength{\itemindent}{0pt}}}{\end{list}}
\newcounter{muni}
\newenvironment{remunerate}
               {\begin{list}{{\hfill\upshape \arabic{muni}.}}{\usecounter{muni}
                \setlength{\leftmargin}{24pt}\labelwidth24pt\labelsep.5em%
		\itemsep2pt\parsep0pt
                \setlength{\itemindent}{0pt}}}{\end{list}}

\newenvironment{bulletlist}
{\begin{list}{{\hfill\raisebox{1.12pt}{$\bullet$}}}{%
                \setlength{\leftmargin}{24pt}\labelwidth24pt\labelsep.5em%
		\itemsep2pt\parsep0pt
                \setlength{\itemindent}{0pt}}}{\end{list}}


\newenvironment{alphlist}
               {\begin{list}{{\hfill\upshape (\alph{muni})}}{\usecounter{muni}
                \setlength{\leftmargin}{24pt}\labelwidth24pt\labelsep.5em%
		\itemsep2pt\parsep0pt
                \setlength{\itemindent}{0pt}}}{\end{list}}


\newcommand\sameauthor{\leavevmode\vrule height 2pt depth -1.6pt width 23pt}


%_%_%_%_% fix eqnarray spacing
\def\@tempb{%
   \stepcounter{equation}%
   \def\@currentlabel{\p@equation\theequation}%
   \global\@eqnswtrue
   \m@th
   \global\@eqcnt\z@
   \tabskip\@centering
   \let\\\@eqncr
   $$\everycr{}\halign to\displaywidth\bgroup
       \hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
      &\global\@eqcnt\@ne\hskip \tw@\arraycolsep \hfil${##}$\hfil
      &\global\@eqcnt\tw@ \hskip \tw@\arraycolsep
         $\displaystyle{##}$\hfil\tabskip\@centering
      &\global\@eqcnt\thr@@ \hb@xt@\z@\bgroup\hss##\egroup
         \tabskip\z@skip
      \cr
}
\ifx\eqnarray\@tempb    % Try the default eqnarray environment
  \def\eqnarray{%
   \stepcounter{equation}%
   \def\@currentlabel{\p@equation\theequation}%
   \global\@eqnswtrue
   \m@th
   \global\@eqcnt\z@
   \tabskip\@centering
   \let\\\@eqncr
   $$\everycr{}\halign to\displaywidth\bgroup
       \hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
      &\global\@eqcnt\@ne \hfil$\displaystyle{{}##{}}$\hfil
      &\global\@eqcnt\tw@ $\displaystyle{##}$\hfil\tabskip\@centering
      &\global\@eqcnt\thr@@ \hb@xt@\z@\bgroup\hss##\egroup
         \tabskip\z@skip
      \cr
}
\else    \typeout{Warning: Unable to fix unknown version of \string\eqnarray.}
\fi
 
\def\@tempb{}


%_%_%_%_%
%_%_%_%_% for crops and other stuff...
%_%_%_%_%
\setlength{\paperheight}{10in}
\setlength{\paperwidth}{7in}

\setlength{\oddsidemargin}{.75in}
\setlength{\evensidemargin}{.766in}

%_%_%_%_%
%_%_%_%_% Problems/exercises
%_%_%_%_%
\newcounter{prob}
\newenvironment{problems}
               {%
\section*{\rule[6pt]{\textwidth}{1pt}\newline\nobreak Problems}%
\addcontentsline{toc}{section}{Problems}%
\markright{Problems}
\begin{list}{{\hfill\upshape \arabic{prob}.}}{\usecounter{prob}
                \setlength{\leftmargin}{24pt}\labelwidth24pt\labelsep.5em%
		\itemsep2pt\parsep0pt
                \setlength{\itemindent}{0pt}}}{\end{list}}

\newenvironment{exercises}
               {%
\section*{\rule[6pt]{\textwidth}{1pt}\newline\nobreak Exercises}%
\addcontentsline{toc}{section}{Exercises}%
\markright{Exercises}
\begin{list}{{\hfill\upshape \arabic{chapter}.\arabic{prob}.}}{\usecounter{prob}
                \setlength{\leftmargin}{28pt}\labelwidth28pt\labelsep.5em%
		\itemsep2pt\parsep0pt
                \setlength{\itemindent}{0pt}}}{\end{list}}

%_%_%_%_%
%_%_%_%_% front matter, etc.
%_%_%_%_%
\newenvironment{thepreface}{\if@openright\cleardoublepage\else\clearpage\fi
\@makeschapterhead{Preface}%
\addcontentsline{toc}{chapter}{Preface}%
                \@mkboth{Preface}%
                        {Preface}%
                \thispagestyle{plain}}
               {\clearpage}

%_%_%_%_%
%_%_%_%_%
%_%_%_%_%

%_%_%_%_% #1 is the contributor's name, #2 is the affiliation
\newcommand{\contributor}[2]{\noindent\vtop{\hsize14pc#1\\\itshape#2}\par}

\newenvironment{contributors}{\if@openright\cleardoublepage\else\clearpage\fi
\@makeschapterhead{List of Contributors}%
%\addcontentsline{toc}{chapter}{List of Contributors}%
                \@mkboth{List of Contributors}%
                        {List of Contributors}%
                \thispagestyle{plain}\begin{multicols}{2}\parindent0pt%
\parskip6pt plus2pt minus1pt%
\widowpenalty10000\clubpenalty10000}
               {\end{multicols}\clearpage}

\setlength{\parskip}{1\parskip}


\parskip1\parskip
\hfuzz362pt
\endinput
%%
%% End of file `book.cls'.



