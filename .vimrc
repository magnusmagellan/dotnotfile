set nocompatible
let &rtp  = '~/.vim/bundle/vimtex,' . &rtp
let &rtp .= ',~/.vim/bundle/vimtex/after'
filetype plugin indent on
syntax enable



let &t_8f = "\<Esc>[38;2;%lu;%lu;%lum"
let &t_8b = "\<Esc>[48;2;%lu;%lu;%lum"
let g:tex_flavor = 'latex'

set termguicolors


set background=dark




" rebinding vim split navigation

"nnoremap <C-J> <C-W><C-J>
"nnoremap <C-K> <C-W><C-K>
"nnoremap <C-L> <C-W><C-L>
"nnoremap <C-H> <C-W><C-H>





set ruler
set number relativenumber
set laststatus=2
set noshowmode 









" bottom bar settings

set showcmd             " show command in bottom bar
"filetype indent on      " load filetype-specific indent files

" folding settings

"set foldmethod=indent
"set foldnestmax=3
"set nofoldenable



"set rtp+=~/.vim/bundle/Vundle.vim



" indendation settings
           
set tabstop=4       " number of visual spaces per TAB
set softtabstop=4   " number of spaces in tab when editing
set expandtab       " tabs are spaces





" bottom bar settings

set showcmd             " show command in bottom bar
filetype indent on      " load filetype-specific indent files

" folding settings

set foldmethod=indent
set foldnestmax=3
set nofoldenable

" misc settings

set wildmenu            " visual autocomplete for command menu
set lazyredraw          " redraw only when we need to.
set scrolloff=3
set autoread
"let g:netrw_winsize = 25




"set t_Co=256
"call vundle#begin()


"Plugin 'SirVer/ultisnips'

" Snippets are separated from the engine. Add this if you want them:
"Plugin 'honza/vim-snippets'

"call vundle#end()            " required


" --- Plugin section --- "
call plug#begin('~/.vim/plugged')

" --- Status line --- "
Plug 'vim-airline/vim-airline'
Plug 'franbach/miramare'
" --- Theming --- "
Plug 'morhetz/gruvbox'
Plug 'vim-airline/vim-airline-themes'
Plug 'UnikMask/iroh-vim'
"Plug 'lervag/vimtex'

"Plug 'lervag/vimtex'
"Plug 'SirVer/ultisnips'

" --- Completion and syntax --- "

" Editing and usability
"Plug 'jiangmiao/auto-pairs'
"Plug 'ryanoasis/vim-devicons'

" --- Programming Languages ---
"Plug 'sheerun/vim-polyglot'
"Plug 'davidhalter/jedi-vim'

" Initialize plugin system
call plug#end()

"if has('termguicolors')
 "   set termguicolors
"endif


"filetype plugin indent on    " required
"syntax on




set splitbelow
set splitright


" gruvbox italic fix (must appear before colorscheme)
"let g:gruvbox_italic = 1
"set background=dark

"colorscheme gruvbox
" Airline powerline fonts fix
"let g:airline_powerline_fonts = 1
" Airline theme
let g:miramare_enable_italic = 1
let g:miramare_disable_italic_comment = 1

colorscheme iroh

"let g:airline_theme = 'gruvbox'
let g:airline_theme = 'iroh'
"let g:gruvbox_contrast_dark = 'dark'
" User-specific Settings.
"hi Directory guifg=#f9cc6c ctermfg=red
let NERDTreeMinimalUI = 1
let NERDTreeDirArrows = 1

" ---Sets---
"
set encoding=utf-8              " UTF-8 Support
set tabstop=4                   " 4 spaces will do
set shiftwidth=4                " control indentation for >> bind
set expandtab                   " spaces instead of tabs
set autoindent                  " always set autoindenting on

set signcolumn=no


"set foldcolumn=2
"let &colorcolumn='+' . join(range(1,256), ',+')


"set relativenumber              " relative line numbers
"set number                      " hybrid numbering with both rnu and number
"set hidden                      " hide buffers instead of closing them
"set ignorecase                  " ignore case when searching
"set smartcase                   " ignore case if all lowercase
"set nobackup                    " don't need swp files
"set noswapfile                  " don't need swp files
"set showmatch                   " Show matching braces when over one
"set backspace=indent,eol,start  " allow backspacing everything in insert
"set hlsearch                    " highlight searches
"set incsearch                   " search as typing
""set laststatus=2                " for lightline.vim plugin

"set mouse+=a
set mouse+=a

"set clipboard^=unnamed

"set clipboard=unnamed
set clipboard=unnamedplus


          " required

" " Trigger configuration. Do not use <tab> if you use
" https://github.com/Valloric/YouCompleteMe.

let g:UltiSnipsExpandTrigger = '<tab>'
let g:UltiSnipsJumpForwardTrigger = '<tab>'
let g:UltiSnipsJumpBackwardTrigger = '<s-tab>'









" Load Vimtex


"autocmd vimenter * NERDTree
"autocmd StdinReadPre * let s:std_in=1
"autocmd VimEnter * if argc() == 1 && isdirectory(argv()[0]) && !exists("s:std_in") | exe 'NERDTree' argv()[0] | wincmd p | ene | exe 'cd '.argv()[0] | endif


let g:UltiSnipsSnippetDirectories=["UltiSnips"]

inoremap <C-f> <Esc>: silent exec '.!inkscape-figures create "'.getline('.').'" "'.b:vimtex.root.'/figures/"'<CR><CR>:w<CR>
nnoremap <C-f> : silent exec '!inkscape-figures edit "'.b:vimtex.root.'/figures/" > /dev/null 2>&1 &'<CR><CR>:redraw!<CR>
"map <C-n> :NERDTreeToggle<CR>

"vmap <leader>y "+y
"vmap <leader>d "+d
"nmap <leader>p "+p
"nmap <leader>P "+P
"vmap <leader>p "+p
"vmap <leader>P "+P
