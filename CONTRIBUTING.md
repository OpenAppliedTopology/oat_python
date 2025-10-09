# Open Applied Topology


Welcome to the OAT community!  We're glad you're here.

Users can find a great sense of satisfaction and accomplishment in helping fellow users and/or modifying open source software; that includes

- adding, subtracting, or changing the code
- catching typos and clarifying explanations
- joining discussions
- reporting problems
- and more!

Here's some information to get started.

- [Code of conduct](#code-of-conduct)
- [Get help](#get-help)  
- [Install](#install)
- [Access documnetation](#access-documentation)
- [Project layout](#project-layout)
- [How-to](#how-to)
  - [Report an issue or request a feature](#report-an-issue-or-request-a-feature)
  - [Contribute new code](#contribute-new-code)  
  - [Run unit tests](#run-unit-tests)
- [Style guide](#style-guide)
- [Introduction to Rust](#introduction-to-rust)
- [Introduction to PyO3](#introduction-to-pyo3)



# Code of conduct

Your safety and well being are the top priorities of the OAT community.  The same holds for every one of our users.  See the [code of conduct](./CODE_OF_CONDUCT.md) for what that means.

# Get help

If you're stuck or don't know where to begin, then you're in good company -- we've all been there!  We're here to help, and we'd love to hear from you:

- open a issue report on Github
- email us at <gregory.roek@pnnl.gov>

# Install

**oat_rust**

If you would like to use `oat_rust` in another Rust project, you can just add it to the list of dependencies in that project's `Cargo.toml` file. See the Rust documentation on dependencies for details.

**oat_python**

If you would like to use `oat_python` in another Rust project, you can just add it to the list of dependencies in that project's `Cargo.toml` file. See the Rust documentation on dependencies for details.

If you'd like to use `oat_python` in Python, it can be installed via `pip install oat_python`.

If you'd like to install `oat_python` from source, follow these steps:

  - Ensure that Rust is installed and up to date (for updates you can run `rustup update`)
  - (Optional) if you want to use a local copy of `oat_rust`, update the line `oat_rust = { path = "path/to/oat_rust"}` in `oat_python/Cargo.toml` so that it points to your local copy.
  - Activate a Python environment
  - Install `maturin` via `pip install maturin`
  - CD into the `oat_python` folder
  - Run `maturin develop --release`
  - `oat_python` should now be installed

# Access documentation

**Online**

Documentation for `oat_rust` can be found on Crates.io. Rust documentation for `oat_python` can also be found on Crates.io.

Python documentation for `oat_python` can be found on Read the Docs.

**From source**

To build native Rust documentation for either `oat_rust` or `oat_python` from source, first ensure Rust is installed and up to date. Then cd into the project folder and run `cargo doc --no-deps --open`. This should build the documentation and open it in a web browser.

To build **Sphinx** documentation for `oat_python`

  - Activate a Python environment  
  - Install `oat_python` via the instructions above
  - Install Sphinx dependencies: run `pip install sphinx`, then pip install all Sphinx extensions listed under `extensions = [..]` in `oat_python/docs/conf.py`
  - CD into `oat_python/docs`
  - Run `make html`
  - You should now be able to open the docs by navigating to `oat_python/_build/html/index.html` in a web browser.


# Project layout

The OAT project includes

- Tutorials
  - Jupyter notebook tutorials, available on colab
- `oat_rust`
  - A high-performance, low-level software package written in pure Rust
  - Registered on `crates.io` as `oat_rust`
- `oat_python`
  - Powerful tools for interactive visualization and analysis.  Provides Python bindings for `oat_rust` using [pyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs).
  - Registered on `PyPi` as `oat_python`
  - Registered on `crates.io` as `oat_python`
  - This package has 
    - a Rust component, stored in `oat_python/src`, and 
    - a Python component, stored in `oat_python/python`. 


These components have the following folder structure

```
tutorials       <-- Jupyter notebook tutorials

oat_rust        
├── src         <-- oat_rust source code, in Rust
└── developer   <-- documents and resources for devlopers

oat_python
├── docs        <-- all files relevant to sphinx documentation
├── examples    <-- .py files for the sphinx-gallery examples ("tutorials" section of documentation)
├── python      <-- oat_python source code, written in Python
└── src         <-- oat_python source code, written in Rust
```



### The `core` python submodule

The Rust component of `oat_python` is exposed as a submodule, `oat_python.core`.
The portion of the Rust code that exposes this component is located in `oat_python/src/lib.rs`:

```
#[pymodule(name="core")]
fn oat_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
  ...
}
```

The line `#[pymodule(name="core")]` controls the name of the exported
submodule.  Maturin also requires this name assignment to be
recorded in `oat_python/pyproject.toml` as follows

```
[tool.maturin]
module-name = "oat_python.core"       
```

### pyproject.toml

The file 
`pyproject.toml` also records the location of the Python code for the project, as follows

```
[tool.maturin]
python-source = "python"  
```

For further details see the Maturin documentation on [project layout](https://www.maturin.rs/project_layout.html).

### Rust Documentation

Documentation for `oat_rust` is managed through the Cargo package manager.
Rust documentation for `oat_python` is also managed by Cargo.

### Python documentation

Python documentation for `oat_python` is built using Sphinx. The files mainly responsible for controlling the documentation build are

```
oat_python/docs/
   conf.py
   index.rst
   sidebar.rst
   glossary.rst   

oat_python/examples
```

These play the following roles:

- `conf.py`: controls the configuration parameters for Sphinx
- `index.rst`: generates the homepage
- `sidebar.rst`: generates the sidebar
- `glossary.rst`: generates the glossary
- `oat_python/examples`: generates the content of the Tutorials section of the sidebar

[See the Sphinx documentation](https://www.sphinx-doc.org/en/master/) to better understand  on `conf.py`, `index.rst`, `sidebar.rst`, and `glossary.rst`. 

[See the Sphinx-Gallery documentation](https://sphinx-gallery.github.io/stable/index.html) to better understand `oat_python/examples`.


# How to

The world of open source is wide; it can be a lot to take in!  If this is your first time working with an open source project, then welcome!  If you're an experienced contributor, then welcome back!  Either way, this online resource might help you [get oriented, or bursh up](https://opensource.guide/how-to-contribute/) on the process.

## Report an issue or request a feature

Here are the [steps to creating an issue on github](https://docs.github.com/en/issues/tracking-your-work-with-issues/quickstart)

**Request a feature**
- Search for related issues on Github. You may also want to search for all issues with the `enhance` tag to see a list of all feature requests.
- If the feature you'd like doesn't already appear, create an issue and describe what you're looking for. It can be helpful to indicate whether you plan to work on developing this feature yourself, or would be open to working together with others to implement it.
- Make sure to
  - Use the `enhance` tag!
  - Star issues you find important!


**Report an issue**
- Search for related issues on Github. You might be able to get answer without the hassle of creating an issue.
- Describe the current behavior and explain which behavior you expected to see instead and why. At this point you can also describe any alternatives you've considered, and why they do not work.
  - (If applicable) provide error messages
  - (If applicable) provide a step by step description of the problem; if possible include code that others can use to reproduce it
  - You may want to [include screenshots and animated GIFs](https://www.cockos.com/licecap/) which help you demonstrate the steps or point out the part which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
  - Provide clear, specific title
  - Include details on your setup (operating system, python version, etc.)
- Use the most recent version of this library and the source language (e.g. Rust); that fixes a lot of problems  
- Here are [more details on getting the most out of issue reporting!](https://marker.io/blog/how-to-write-bug-report)

## Contribute new code

Here is a [step-by-step guide to writing new code, and submiting it to the project](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)

The more you know about a software library, the easier it is to get started writing code.  The best way to learn about this project is its the documentation!  See `README.md` to get started.


## Run unit tests

- `oat_rust`: CD into the `oat_rust` directory and run `cargo test`.  
- `oat_python`: the `cargo test` method often fails due to complications with PyO3. Unit tests for this package are currently handled on an adhoc basis, and are under developmnet.

# Style guide

### Comments

-  use !!! to tag items that developers should inspect, in the future
   - review all occurrences of `!!!` in the comments before making a commit

### Naming conventions

- prioritize code legibility to without reference to documentation
  - where possible, test legibility of object and method names by generating functional, working code that uses these methods or functions
    - could a new user accurately guess the meaning of this code without referring to the documentation? 
    - are the new names prone to misinterpretation?
  - where possible, share these test examples with friends or colleagues for input
  - try stepping away for several weeks and 
    - return with fresh eyes to check that the names make sense
    - try writing some new code without referring back to your original source; can you accurately guess the names you assigned several weeks ago?

- abbreviate only when absolutely necessary
  - prefer `ring_operator` to `ring_op`
  - prefer `RingOperator` to `RingOp`

- prefer concreteness to brevity
  - prefer `RingOperator` to `Ring`

- prefer `output_for_input` function names
  - for example, `vietoris_rips_complex.filtration_value_for_simplex( simplex )`
  - this convention is not required for function names, but it is a preferred format for name which include but an input and an output

- prefer snake case and avoid spaces
  - for all methods in Rust and Python
  - for dictionary keys and DataFrame row and column labels, e.g. prefer `"birth_filtration"` to `"birth filtration"`

### Code format 

- use indentation and line breaks liberally






# Introduction to Rust

Rust is a low-level programming language with powerful features for scientific computing, e.g. memory safety and helpful error messages.  It has been voted [most-loved language](https://insights.stackoverflow.com/survey/2021) by the worldwide developer community since 2015.

* **Installation**  The  [Rust website](https://www.rust-lang.org/learn/get-started) has directions!

* **Search for what you need in the documentation** All Rust documenation has a consistent format, which you can search in a web browser.  
  - Lists of objects, functions, etc., appear in two places: either the bottom of a page, or in the menu bar on the left.
  - You can also use the search bar at the top to pull up a list of related terms.  
  - The question mark button to the right of the bar gives other helpful tips (for example, you can search for functions based on their type signature).

* **Save up to 90% coding time, with VS Code** If you are new to Rust, we strongly recommend the [VS Code editor](https://code.visualstudio.com/docs/languages/rust), with the [`rust-analyzer`](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) extension.  *Everyone we know uses VS code for Rust development*.  It has great features for

  - highlighting
  - smart debugging messages
  - command completion
  - bracket coloring; since type parameters are defined with `<>` brackets, it may be useful to add these to your `configuration.json` file, as per this [discussion](https://github.com/microsoft/vscode/issues/132476) and [example](https://github.com/microsoft/vscode/blob/997228d528d687fd17cbf3ba117a0d4f668c1393/extensions/javascript/tags-language-configuration.json#L11)

  

* **Debugging** Rust is very good about providing helpful error messages, as a rule.
It is often possible to find help just by copying these messages into a web search.
The OAT developers have also collected a short list of [debugging tips and tricks](crate::developer::rust_debugging), which you may find useful.

* **Tips**

  - long type definitions: because OAT is highly modular, its type can become quite verbose.  The Rust compiler offers a lot of helpful information about types when running your code, but it abbreviates some types when printing to terminal; note however, that whenever it does this, it also writes a full type description in a file, and prints the file path in the command shell.
  - the [`debugit`](https://docs.rs/debugit/latest/debugit/) crate is useful for debugging


# Introduction to PyO3

PyO3 is a software package to link Rust code with Python.  There are three good ways to learn about PyO3 for this project:

- [PyO3 homepage](https://pyo3.rs/)
- [PyO3 API](https://docs.rs/pyo3) - this contains slightly different information than the homepage
- the [OATpy git respository](https://pyo3.rs/) - many examples that aren't covered in the previous two sources can be found and copy/pasted from the source code!
