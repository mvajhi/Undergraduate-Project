% !TeX program = xelatex
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tehran-poster}[2026/07/11 v1.0 Modular University of Tehran poster]

\LoadClass[12pt]{article}

\RequirePackage[
  paperwidth=59.4cm,
  paperheight=84.1cm,
  margin=0pt,
  noheadfoot
]{geometry}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\usetikzlibrary{calc,positioning}
\RequirePackage{pgfplots}
\pgfplotsset{compat=1.18}
\RequirePackage{amsmath,amssymb}
\RequirePackage{array}
\RequirePackage{booktabs}
\RequirePackage{enumitem}
\RequirePackage{ragged2e}
\RequirePackage{hyperref}
\hypersetup{hidelinks,unicode=true}

% xepersian must be the last package.  This mirrors the accompanying
% tehran-thesis template and gives robust Persian/Latin/math integration.
\RequirePackage{xepersian}

% Bundled fonts make the template reproducible and independent of OS fonts.
\settextfont[
  Path=fonts/,
  BoldFont=IRLotusICEE_Bold.ttf,
  ItalicFont=IRLotusICEE_Iranic.ttf,
  BoldItalicFont=IRLotusICEE_BoldIranic.ttf,
  Scale=1.18
]{IRLotusICEE.ttf}
\setlatintextfont[
  Path=fonts/,
  BoldFont=LiberationSans-Bold.ttf,
  Scale=1
]{LiberationSans-Regular.ttf}
\setdigitfont[
  Path=fonts/,
  BoldFont=IRLotusICEE_Bold.ttf,
  ItalicFont=IRLotusICEE_Iranic.ttf,
  BoldItalicFont=IRLotusICEE_BoldIranic.ttf,
  Scale=1.18
]{IRLotusICEE.ttf}
\defpersianfont\postertitlefont[
  Path=fonts/,
  BoldFont=IRTitr.ttf,
  ItalicFont=IRTitr.ttf,
  Scale=1
]{IRTitr.ttf}

\definecolor{PosterPurple}{HTML}{57215D}
\definecolor{PosterPurpleLight}{HTML}{6E2A76}
\definecolor{PosterBlue}{HTML}{55C6EC}
\definecolor{PosterText}{HTML}{151515}
\definecolor{PosterMuted}{HTML}{555555}
\definecolor{PosterPanel}{HTML}{FFFFFF}

\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{.38cm}
\setlist[itemize]{nosep,leftmargin=1.25cm,label=\textbullet}
\AtBeginDocument{%
  \fontsize{22}{34}\selectfont
  \color{PosterText}%
}

% --------------------------------------------------------------------------
% Poster metadata
% --------------------------------------------------------------------------
\def\PosterTitle{عنوان پروژه}
\def\PosterStudent{}
\def\PosterSupervisor{}
\def\PosterAffiliation{دانشکده مهندسی برق و کامپیوتر، دانشگاه تهران}
\def\PosterDate{}
\def\PosterLeftLogo{assets/engineering-logo.png}
\def\PosterRightLogo{assets/ut-logo.png}

\pgfkeys{
  /tehran poster/.is family,
  /tehran poster,
  title/.store in=\PosterTitle,
  student/.store in=\PosterStudent,
  supervisor/.store in=\PosterSupervisor,
  affiliation/.store in=\PosterAffiliation,
  date/.store in=\PosterDate,
  left logo/.store in=\PosterLeftLogo,
  right logo/.store in=\PosterRightLogo,
}
\newcommand{\PosterSetup}[1]{\pgfkeys{/tehran poster,#1}}

% --------------------------------------------------------------------------
% Reusable building blocks
% --------------------------------------------------------------------------
\newlength\PosterPanelPadding
\setlength\PosterPanelPadding{1.55cm}

\newcommand{\PosterBackground}{%
  \begin{tikzpicture}[remember picture,overlay]
    \fill[PosterPurple] (current page.south west) rectangle (current page.north east);
    \node[anchor=center,inner sep=0,opacity=.42] at (current page.center)
      {\includegraphics[width=\paperwidth,height=\paperheight]{assets/electronics-background.png}};
  \end{tikzpicture}%
}

\newcommand{\PosterHeader}{%
  \begin{tikzpicture}[remember picture,overlay]
    \node[
      anchor=north west,
      rounded corners=.7cm,
      fill=PosterPanel,
      minimum width=56.8cm,
      minimum height=10.65cm,
      inner sep=0
    ] (posterheader) at ([xshift=1.3cm,yshift=-.85cm]current page.north west) {};
    \node[anchor=west,inner sep=0] at ([xshift=1.0cm]posterheader.west)
      {\includegraphics[width=8.2cm,height=6.5cm,keepaspectratio]{\PosterLeftLogo}};
    \node[anchor=east,inner sep=0] at ([xshift=-1.0cm]posterheader.east)
      {\includegraphics[width=6.9cm,height=6.9cm,keepaspectratio]{\PosterRightLogo}};
    \node[anchor=center,align=center,text width=37cm,inner sep=0] at ([yshift=.1cm]posterheader.center) {%
      \setRTL
      {\postertitlefont\fontsize{62}{74}\selectfont \rl{\PosterTitle}\par}
      \vspace{.2cm}
      {\bfseries\fontsize{38}{48}\selectfont
        دانشجو: \PosterStudent\par
        استاد راهنما: \PosterSupervisor\par}
      {\bfseries\fontsize{24}{32}\selectfont \PosterAffiliation
       \ifx\PosterDate\empty\else\quad--\quad\PosterDate\fi\par}
    };
  \end{tikzpicture}%
}

% #1 optional body size, #2 title, #3 x from left, #4 y from top,
% #5 width, #6 height, #7 content.  PowerPoint's original panel title is 54pt.
\NewDocumentCommand{\PosterBlock}{O{26} m m m m m +m}{%
  \begin{tikzpicture}[remember picture,overlay]
    \node[
      anchor=north west,
      rounded corners=.62cm,
      fill=PosterPanel,
      minimum width=#5,
      minimum height=#6,
      inner sep=0
    ] (posterblock) at ([xshift=#3,yshift=-#4]current page.north west) {};
    \node[anchor=north west,inner sep=0] at
      ([xshift=\PosterPanelPadding,yshift=-1.05cm]posterblock.north west) {%
      \begin{minipage}[t][\dimexpr#6-2.05cm\relax][t]{\dimexpr#5-2\PosterPanelPadding\relax}
        % TikZ constructs its node boxes in LTR mode.  Reset the paragraph
        % direction explicitly before typesetting Persian panel content.
        \setRTL
        \centering
        {\postertitlefont\fontsize{54}{66}\selectfont #2\par}
        \vspace{.45cm}
        \RaggedLeft
        \fontsize{#1}{\dimexpr#1pt+8pt\relax}\selectfont
        #7
      \end{minipage}%
    };
  \end{tikzpicture}%
}

\newcommand{\PosterNote}[1]{%
  \par\vspace{.2cm}{\textasteriskcentered\ #1\par}%
}

% Use a real image when available; otherwise show a printable replacement box.
\newcommand{\PosterImage}[3][\linewidth]{%
  \begin{center}
  \IfFileExists{#2}{%
    \includegraphics[width=#1,height=#3,keepaspectratio]{#2}%
  }{%
    \fbox{\begin{minipage}[c][#3][c]{\dimexpr#1-2\fboxsep-2\fboxrule\relax}
      \centering\color{PosterMuted}
      جای تصویر\\[-.1cm]
      {\latinfont\small\texttt{#2}}
    \end{minipage}}%
  }
  \end{center}%
}

\newcommand{\PosterEnglish}[1]{\lr{#1}}
\newcommand{\PosterMath}[1]{\ensuremath{#1}}

% Force correct left-to-right digit-group order for a Persian number that
% mixes a decimal point or thousands separator (۰-۹ combined with . or ٬).
% Without this, XeTeX's bidi algorithm can swap the digit groups around the
% separator inside RTL text (e.g. "۸.۰۵" rendering as "۰۵.۸").
\newcommand{\PosterNum}[1]{\beginL#1\endL}

% Bold propositional subheading inside a block (the "تیتر گزاره‌ای" rule).
\newcommand{\PosterSubhead}[1]{{\fontsize{29}{36}\selectfont\bfseries\color{PosterPurple} #1\par}}

% Thin horizontal rule standing in for a card boundary between insights.
\newcommand{\PosterDivider}{\par\vspace{.10cm}\noindent\textcolor{PosterMuted!45}{\hrulefill}\vspace{.14cm}\par}

\endinput
