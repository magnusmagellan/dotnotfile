#
# ~/.bashrc
#

# If not running interactively, don't do anything
#[[ $- != *i* ]] && return
#export PYTHONPATH=~/illustris_python
alias todo='~/TerminalToDo/terminalTodo'
alias ls='ls --color=auto'
alias qmgmx='/opt/gromacs/gromacs_2019_qmmm/bin/GMXRC'


#alias manimm ='~/.local/bin/manim'

#PS1='[\u@\h \W]\$ '
g09root=/home/magnus/g09-legacy-D01
GAUSS_SCRDIR=/home/magnus/dell
export g09root GAUSS_SCRDIR
. $g09root/g09/bsd/g09.profile
export PYTHONPATH=~/illustris_python
#eval "$(starship init bash)"



alias vlc="devour vlc"
alias sxiv="devour sxiv"
alias zathura="devour zathura"
alias gwenview="devour gwenview"
alias avogadro="devour avogadro"

#PATH=~/.local/bin/manim:$PATH; export PATH

#PATH=/opt/texlive/2020/bin/x86_64-linux/:$PATH; export PATH

#MANPATH=/usr/local/texlive/2018/texmf-dist/doc/man:$MANPATH; export MANPATH

#MANPATH=/opt/texlive/2020/texmf-dist/doc/man:$MANPATH; export MANPATH

#INFOPATH=/opt/texlive/2020/texmf-dist/doc/info:$INFOPATH; export INFOPATH



#PATH=/opt/texlive/2020/bin/x86_64-linux//latex:$PATH






