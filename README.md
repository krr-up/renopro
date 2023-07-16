# renopro

`renopro` is package implementing reification and reflection of non-ground programs in
Answer Set Programming (ASP) using the `clingo` ASP system's API.

Reification of non-ground programs is achieved by encoding the
abstract syntax tree (AST) of an input non-ground program string into
a set of ASP facts. These facts can then be used in various downstream
applications where reasoning about ASP programs is required.

Reflection is the reverse operation of reification, i.e. transforming
a set of facts encoding an asp program into it's string
representation.  This operation can be used to derive new program
strings which may then e.g. be passed along to an ASP solver.

An application `renopro` implements which makes use of both
reification and reflection is syntactic transformation of input
programs using ASP itself.  First the input program is reified into a
set of facts.  Then, these facts are combined with a user-provided
meta-program which encodes the desired transformations, and passed
along to `clingo`.  Finally, the derived facts representing the
transformed program are reflected back into a program string and
returned to the user.

## Installation

```shell
pip install renopro
```

## Usage

```shell
renopro -h
```

## Development

To improve code quality, we run linters, type checkers, and unit tests. The
tools can be run using [nox]. We recommend installing nox using [pipx] to have
it available globally:

```bash
python -m pip install pipx
python -m pipx install nox
nox
```

You can invoke `nox -s` to run individual sessions. For example, to install
your package into a virtual environment and run your test suite, invoke:

```bash
nox -s test
```

We also provide a nox session that creates an environment for development. The
project is installed in [editable] mode into this environment along with
linting, type checking and formatting tools. Activating it allows your editor
of choice to access these tools for, e.g., linting and autocompletion. To
create and then activate virtual environment run:

```bash
nox -s dev
source .nox/dev/bin/activate
```

Furthermore, we provide individual sessions to easily run linting, type
checking and formatting via nox. These also create editable installs. So you
can safely skip the recreation of the virtual environment and reinstallation of
your package in subsequent runs by passing the `-R` command line argument. For
example, to auto-format your code using [black], run:

```bash
nox -Rs format -- check
nox -Rs format
```

The former command allows you to inspect changes before applying them.

Note that editable installs have some caveats. In case there are issues, try
recreating environments by dropping the `-R` option. If your project is
incompatible with editable installs, adjust the `noxfile.py` to disable them.

We also provide a [pre-commit][pre] config to automate this process. It can be
set up using the following commands:

```bash
python -m pipx install pre-commit
pre-commit install
```

This blackens the source code whenever `git commit` is used.

[doc]: https://potassco.org/clingo/python-api/current/
[nox]: https://nox.thea.codes/en/stable/index.html
[pipx]: https://pypa.github.io/pipx/
[pre]: https://pre-commit.com/
[black]: https://black.readthedocs.io/en/stable/
[editable]: https://setuptools.pypa.io/en/latest/userguide/development_mode.html
