\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{proposal}[1999/07/02 NSF proposal style v0.1 SGLS]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass[12pt,letterpaper]{article}
\RequirePackage{calc}
\RequirePackage{epsfig}

%\def\@biblabel#1{} % Suppress labels in bibliography

\addtolength{\voffset}{-1in}
\addtolength{\hoffset}{-1in}
\setlength{\topmargin}{2.5cm}
\setlength{\oddsidemargin}{2.5cm}
\setlength{\evensidemargin}{2.5cm}
\setlength{\textwidth}{\paperwidth - 5cm}
\setlength{\textheight}{\paperheight - 5cm}
\setlength{\footskip}{36pt}
\setlength{\marginparsep}{0.5cm}
\setlength{\marginparwidth}{1.5cm}
\setlength{\headheight}{0pt}
\setlength{\headsep}{0pt}

\RequirePackage{fancyheadings}
\pagestyle{fancyplain}
\setlength{\headrulewidth}{0pt}
\lhead{}
\chead{}
\rhead{}
\lfoot{}
\cfoot{\thepage}
\rfoot{}

\def\@makefnmark{\hbox{$^{\fnsymbol{\@mpfn}}\m@th$}}

\RequirePackage{times}
%\RequirePackage{mathptm}
\RequirePackage{amsmath}
\RequirePackage[square]{natbib}
%\def\degrees{\char23}
%\def\times{{\fontencoding{U}\fontfamily{psy}\selectfont\char180}}
\def\S{{\fontencoding{T1}\selectfont\char159}}
\DeclareSymbolFont{psymbols}{U}{psy}{m}{n}
%\let\times\relax
\DeclareMathSymbol{\times}{\mathbin}{psymbols}{180}

\DeclareMathSymbol{\prime}{\mathord}{operators}{"13}
%\makeatletter
{\catcode`\'=\active \gdef'{\bgroup\prim@s}}
%\makeatother
