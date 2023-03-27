# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# pylint: disable=unused-import, redefined-builtin

import shutil
import os
import sys
import re
import string
import datetime
from typer.testing import CliRunner

# Include this to avoid circular import errors
import prompt_toolkit
import sphinx_material

os.environ['MLTK_BUILD_DOCS'] = '1'

import mltk
from mltk import cli
from mltk import MLTK_DIR, MLTK_ROOT_DIR
from mltk.utils.path import (
    clean_directory,
    get_user_setting,
    recursive_listdir,
    fullpath,
    walk_with_depth
)




curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(f'{curdir}/../../../mltk'))
repo_project_name = os.environ.get('MLTK_REPO_PROJECT_NAME', 'siliconlabs')
repo_url = f'https://github.com/{repo_project_name}/mltk'
repo_project_name = get_user_setting('repo_project_name', 'siliconlabs')


# -- Project information -----------------------------------------------------

project = 'MLTK'
copyright = f'{datetime.date.today().year}, Silicon Labs'
author = 'Silicon Labs'
release = mltk.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon',
    #'sphinx.ext.autosectionlabel',
    "myst_nb",
     # "myst_parser", # this is not needed when myst_nb is used
    "sphinx_markdown_tables",
    "sphinx_copybutton",
    'sphinx_design',
    'sphinx_material',
    'sphinx_autodoc_typehints', # This must come last
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_static/templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_static',
    '*lib/site-packages',
    '*lib\\site-packages'
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'
html_logo = '_static/logo.png'
html_favicon = "_static/favicon.ico"

html_show_sourcelink = True
html_use_index = True
html_domain_indices = True
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_js_files = [
    'js/custom.js',
    'js/apitoc.js'
]
html_css_files = [
    'css/custom.css',
]

# https://bashtage.github.io/sphinx-material/customization.html#customization
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    'nav_title': 'Machine Learning Toolkit',

    # Set you GA account ID to enable tracking
    'google_analytics_account': 'G-HZ5MW943WF',

    # Set the color and the accent color
    'color_primary': 'red',
    'color_accent': 'light-blue',

    'base_url': False,
    'repo_url': repo_url,
    'repo_name': 'MLTK Github Repository',

    # Visible levels of the global TOC;
    'globaltoc_depth': 2,
    # If False, expand all TOC entries
    'globaltoc_collapse': True,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': True,
    'heroes': {
        'index': 'A Python package to aid the development of machine learning models for Silicon Lab\'s embedded platforms',
        'docs/installation': 'Install the MLTK Python Package via PIP or in <a href="https://colab.research.google.com/notebooks/welcome.ipynb" target="_blank">Google Colab<a/>',
        'docs/command_line': 'Execute the MLTK modeling features from the command-line',
        'docs/guides/index': 'The MLTK provides numerous features to aid the development of ML models',
        'docs/guides/model_profiler': 'Profile a model to determine how efficiently it may execute on hardware',
        'docs/guides/model_visualizer' : 'Visualize a model using an interactive webpage',
        'docs/guides/model_specification': 'Define a model specification using a standard Python script',
        'docs/guides/model_training': 'Train an ML model using Google Tensorflow',
        'docs/guides/model_evaluation': 'Evaluate an ML model to determine how accurate it is',
        'docs/guides/model_quantization': 'Quantize a model to reduce its memory footprint',
        'docs/guides/model_parameters': 'Embed custom parameters into the generated model file',
        'docs/guides/model_summary': 'Generate a textual summary of a model',
        'docs/guides/model_archive': 'All model development files are store in a distributable archive',
        'docs/guides/model_search_path': 'Specify the model search path',
    },
    'nav_links': [
        {
            'href': 'https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/overview',
            'internal': False,
            'title': 'Gecko SDK Documentation',
        },
        {
            'href': 'https://github.com/tensorflow/tflite-micro',
            'internal': False,
            'title': 'Tensorflow-Lite Micro Repository',
        },
        {
            'href': 'https://www.tensorflow.org/learn',
            'internal': False,
            'title': 'Tensorflow Documentation',
        },
    ],

    'html_minify': False,
    'html_prettify': False,
    'css_minify': False,

    'version_dropdown': False,
    'version_json': None,
    'version_info': None
}


html_sidebars = {
    '**': ['logo-text.html', 'globaltoc.html', 'localtoc.html', 'searchbox.html']
}



# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

pygments_style = 'trac'
add_module_names = False
autosummary_generate = True
#autosummary_imported_members = True
numpydoc_class_members_toctree = False

numpydoc_show_class_members = False
typehints_fully_qualified = False
set_type_checking_flag  = True

autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autoclass_content = "class"
autosectionlabel_prefix_document = True
autodoc_class_signature = 'separated'
#autosectionlabel_maxdepth = 1
myst_heading_anchors = 3
language = "en"
html_last_updated_fmt = ""

todo_include_todos = True

jupyter_execute_notebooks = 'off'

build_dir = sys.argv[-1]


# Copy all the .md files in <mltk root>/docs
# to <mltk root>/docs/website_builder/source/docs
# This allows for sphinx to generate HTML for each .md
docs_src_dir = fullpath(f'{curdir}/../..')
docs_dst_dir = f'{curdir}/docs'
clean_directory(docs_dst_dir)
include_re = re.compile(r'```{include}\s+(.*)\s*```.*')
cpp_path_re = re.compile(r'.*\]\((\.\.\/[\.\.\/]*cpp)[\/\)]+.*')
mltk_path_re = re.compile(r'.*\]\((\.\.\/[\.\.\/]*mltk\/core)[\/\)]+.*')
for fn in recursive_listdir(docs_src_dir, return_relative_paths=True):
    if not fn.endswith(('.md', '.rst')):
        continue
    if 'website_builder' in fn:
        continue

    src_path = f'{docs_src_dir}/{fn}'
    dst_path = f'{docs_dst_dir}/{fn}'
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # First check if the file contains:
    # ```{include} <path>
    #
    # If it does, then update the source path to that path.
    # This way we can process the actual markdown file
    with open(src_path, 'r') as f:
        data = f.read().replace('\r', '').replace('\n', '')
        match = include_re.match(data)
        if match:
            relpath = match.group(1)
            src_path = f'{os.path.dirname(src_path)}/{relpath}'

    # Now process the markdown file
    with open(src_path, 'r') as f:
        data = ''
        for line in f:
            if 'Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file' in line:
                continue

            # Update the relative links to C++ code to point to the github repo
            # https://github.com/siliconlabs/mltk/cpp
            match = cpp_path_re.match(line)
            if match:
                line = line.replace(match.group(1), 'https://github.com/siliconlabs/mltk/tree/master/cpp/')
            else:
                # Update the relative links to python code to point to the github repo
                # https://github.com/siliconlabs/mltk/mltk/core
                match = mltk_path_re.match(line)
                if match:
                    line = line.replace(match.group(1), 'https://github.com/siliconlabs/mltk/tree/master/mltk/core/')
            data += line

    with open(dst_path, 'w') as f:
        f.write(data)


# Generate a markdown file containing each command's CLI --help
runner = CliRunner()
cli.create_cli()
for cmd in (
    'profile', 'train', 'tensorboard', 'ssh',
    'evaluate', 'quantize', 'summarize', 'view',
    'update_params', 'view_audio', 'classify_audio',
    'classify_image', 'fingerprint_reader', 'commander'
):
    result = runner.invoke(cli.root_cli, [cmd, '--help'], color=False, prog_name='mltk')
    help_str = ''.join(c for c in result.stdout if c in string.printable)
    with open(f'{docs_dst_dir}/command_line/{cmd}_cli_help.md', 'w') as f:
        f.write('```text\n')
        f.write(help_str)
        f.write('```\n')

# Copy all the files in <mltk root>/docs/img
# to <mltk root>/docs/website_builder/source/docs/img
# This allows for relative referencing
img_src_dir = f'{curdir}/../../img'
img_dst_dir = f'{curdir}/docs/img'
os.makedirs(img_dst_dir, exist_ok=True)
clean_directory(img_dst_dir)
shutil.copytree(img_src_dir, img_dst_dir, dirs_exist_ok=True)

# Copy <mltk root>/docs/img/models -> <mltk root>/docs/_images/models
# We need to do this because the images are referenced as raw HTML in
# each model python script header comments which won't be processed by sphinx
img_src_dir = f'{curdir}/../../img/models'
img_dst_dir = f'{curdir}/../../_images/models'
os.makedirs(img_dst_dir, exist_ok=True)
clean_directory(img_dst_dir)
shutil.copytree(img_src_dir, img_dst_dir, dirs_exist_ok=True)


# Copy each .ipynb or .md file in <mltk root>/mltk/examples
# to <mltk root>/docs/website_builder/source/mltk/examples directory
# Also update the Github project name as necessary
EXAMPLE_SRC_DIS = [
    'mltk/examples',
    'mltk/tutorials',
    'cpp/shared/uart_stream'
]

url_re = re.compile(r'.*\((https:\/\/siliconlabs\.github\.io\/mltk\/).*', re.I)
url_re2 = re.compile(r'.*\((https:\/\/siliconlabs\.github\.io\/mltk\/).*(\.html).*', re.I)
docs_img_re = re.compile(r'.*\((https:\/\/github.com\/SiliconLabs\/mltk\/raw\/master\/docs\/img\/).*', re.I)
# If the line contains something like:
# (https://siliconlabs.github.io/mltk/docs/python_api/mltk_model.html#mltk.core.TrainMixin.tflite_converter)
# Then we cannot convert it to it's corresponding relative markdown URL, so it's not supported
not_supported_url_re = re.compile(r'.*\(https:\/\/siliconlabs\.github\.io\/mltk\/.*\.html#.*[\.\-].*\)', re.I)

for src_dir in EXAMPLE_SRC_DIS:
    for root, _, files in walk_with_depth(f'{MLTK_ROOT_DIR}/{src_dir}', depth=10):
        for fn in files:
            if not fn.endswith(('.ipynb', '.md')):
                continue
            rel_dir = os.path.relpath(root, MLTK_ROOT_DIR)
            dst_dir = f'{curdir}/{rel_dir}'
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(f'{root}/{fn}', f'{dst_dir}/{fn}')
            data = ''
            with open(f'{dst_dir}/{fn}', 'r', encoding='utf-8') as fp:
                for line in fp:
                    # The notebooks use full URLs.
                    # Convert the URLs to a relative path when we're generating
                    # so we can generate HTML with relative links
                    match = url_re.match(line)
                    not_supported_match = not_supported_url_re.match(line)
                    rel_path_to_root = os.path.relpath(curdir, dst_dir).replace('\\', '/') + '/'
                    if not_supported_match is None and match:
                        match2 = url_re2.match(line)
                        line = line.replace(match.group(1), rel_path_to_root)
                        if match2:
                            line = line.replace(match2.group(2), '.md')

                    match = docs_img_re.match(line)
                    if match:
                        line = line.replace(match.group(1), f'{rel_path_to_root}docs/img/')

                    data += line + '\n'

            with open(f'{dst_dir}/{fn}', 'w', encoding='utf-8') as fp:
                fp.write(data)



# Copy the <mltk root>/mltk/core/tflite_model_parameters/schema/dictionary.fbs
# to the docs build directory
dictionary_fbs_src = f'{MLTK_DIR}/core/tflite_model_parameters/schema/dictionary.fbs'
dictionary_fbs_dst = f'{docs_dst_dir}/python_api/tflite_model/dictionary.fbs'
dictionary_fbs_build_dir = f'{build_dir}/docs/python_api/tflite_model/dictionary.fbs'
shutil.copy(dictionary_fbs_src, dictionary_fbs_dst)
os.makedirs(os.path.dirname(dictionary_fbs_build_dir), exist_ok=True)
shutil.copy(dictionary_fbs_src, dictionary_fbs_build_dir)

def autodoc_skip_member(app, what, name, obj, skip, opts):
    # we can document otherwise excluded entities here by returning False
    # or skip otherwise included entities by returning True
    qual_name = getattr(obj, '__qualname__', '')
    module = getattr(obj, '__module__', '')
    fullname = f'{module}.{qual_name}'

    if 'keras.preprocessing.image.ImageDataGenerator.flow_from_dataframe' in fullname:
        return True

    return None



def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)

