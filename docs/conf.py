# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
file_dir = os.path.dirname(os.path.realpath(__file__))
matlab_src_dir = os.path.abspath(os.path.join(file_dir,'..')) #must be set according to docs


# -- Project information -----------------------------------------------------

project = 'PyCom'
copyright = '2019, Alec Weiss'
author = 'Alec Weiss'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- Docstring Parsing -------------------------------------------------------
import commonmark
import re

def param_parse_fun(str):
    '''
    @brief function to parse our parameters and kwargs
    @note kwargs should be of the following form 
    @param[in/OPT] kwargs - keyword args as follows:
        - kwarg1 - keywor argument 1
        - kwarg2 - keyword argument 2
    '''
    #first change @param[in] name - blah blah blah to :param name: blah blah blah
    str = ':param '+' '.join(str.strip().split(' ')[1:]).replace(' -',':',1)
    #remove excess newlines while parsing
    str = str.replace('\n','')
    #now lets try and parse a list of arguments 
    splits = str.split('-') #split on '-' and assume [0] is the param naming crap
    split_names = splits[1::2]
    split_vals = splits[2::2]
    #now have each a separate split argument
    split_args = ['*'+name.strip()+'*->'+val.strip() for name,val in zip(split_names,split_vals)]
    str = ' \n ---  '.join([splits[0]]+split_args)
    #remove excess whitespace
    str = '\n'+re.sub(' +',' ',str)+'\n' 
    return str

dox_funct_dict = {
    "brief"  : lambda str: re.sub(' +',' ',str.strip().strip("brief").replace('\n','').strip()),
    "param"  : param_parse_fun,
    "example": lambda str: re.sub(' ',' ',('.. code-block:: python\n'+str.strip().strip('example')).replace('\n','\n   ')+'\n'),
    "return" : lambda str: re.sub(' +',' ',":return: "+str.strip().strip("return").replace('\n','')+'\n'),
    "note"   : lambda str: re.sub(' +',' ',".. note:"+str.strip().strip("note").replace('\n','')+'\n'),
    "warning": lambda str: re.sub(' +',' ',".. warning:"+str.strip().strip("warning").replace('\n','')+'\n'),
    "todo"   : lambda str: re.sub(' +',' ',".. todo::"+str.strip().strip("todo").replace('\n','')+'\n'),
    "cite"   : lambda str: re.sub(' +',' ',".. seealso:: "+"*"+str.strip().strip("cite").replace('\n','')+"*\n"),
}
    
def docstring_preprocess(doc_str):
    '''@brief preprocess our doc strings'''
    doc_str = doc_str.replace('*','\*')
    doc_str = doc_str.replace('"','\"')
    return doc_str

def doxygen2rst(dox_str):
    '''@brief take doxygen-like docstrings that are partially rst and make them restructured text'''
    dox_str = dox_str.strip() #remove any leading or trailing whitespaces
    dox_lines = dox_str.split('@')[1:] #split lines on ampersand and remove first empty bit
    rst_str_list = []
    for dl in dox_lines: #loop through each split line
        for k in dox_funct_dict.keys(): #look for the key
            if dl.startswith(k):
                dl = dox_funct_dict[k](dl) #format the line
        rst_str_list.append(dl) #add the line to the lines
    return ' \n'.join(rst_str_list)

def docstring(app, what, name, obj, options, lines): #change this to not use markdown
    dox  = '\n'.join(lines)
    rst = docstring_preprocess(dox)
    #ast = commonmark.Parser().parse(md)
    #rst = commonmark.ReStructuredTextRenderer().render(ast)
    lines.clear()
    rst = doxygen2rst(rst)
    for line in rst.splitlines():
        lines.append(line)

def setup(app):
    app.connect('autodoc-process-docstring', docstring)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.autodoc',
'sphinxcontrib.matlab',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'sphinx_rtd_theme'
html_theme = 'agogo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
