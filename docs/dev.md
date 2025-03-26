# Developer documentation

## Github CI

On Github a few actions are run upon a pull request.

- [Linting](https://docs.astral.sh/ruff/linter/) the code (checking code for possible bugs)
- [Formatting](https://docs.astral.sh/ruff/formatter/) the code (to have a consistent style)
- Checking [typing](https://docs.python.org/3/library/typing.html) and building the package.

For the linting and formatting we use Ruff. Ruff is configured in `pyproject.toml`.

If you follow the [system setup instructions](system_setup.md), upon installing the workflow run the following command instead:

```py
pip install -e .[dev, docs]
```

This will install additional packages that allow you to use the code formatting and linting, as well as building the docs locally.

Sometimes updates of ruff break the workflow, and you have to fix any issues. You can install the latest version with `pip install ruff --upgrade` so that you use the same version locally as the GitHub action.

### Running CI tasks locally

You can run the ruff linter with the following command;

```
ruff check src/
```

It will tell you where any issues are in your code. Sometimes [you can ignore the issues](https://docs.astral.sh/ruff/linter/#error-suppression) if you have a good reason to disagree.

The formatter is run with;

```
ruff format src/
```

It will reformat your code to comply with the code style. You can then commit these changes to git.


Type checking is done with mypy. You can run it with;

```
mypy src/ --install-types
```

The `install-types` flag will install additional type stubs. In some cases the type checker goes wrong. You [can ignore errors on specific lines](https://mypy.readthedocs.io/en/stable/common_issues.html#spurious-errors-and-locally-silencing-the-checker).

### Documentation

Documentation is built with the `mkdocs` python package and hosted on ReadTheDocs. The RTD built is configured in the `readthedocs.yaml` file. The mkdocs configuration is in `mkdocs.yml`.

You can locally view the documentation by running the following command:

```
mkdocs serve
```

This allows you to make changes to the documentation and view them immediately (after the docs are rebuilt).

## Working on the code

When working on the code keep in mind the following things:

- Work in a branch, and merge to the main branch in a Pull Request on GitHub, where other developers can review your code.
- Make sure that your code passes all CI tests before merging.
- Update the documentation so that you and others can use it later.
- Keep track of changes in the `CHANGELOG.md` file.
- Use releases to create versions of the workflow that are easier to refer to later (or referable in a publication).
  - Before a release update the version number in `src/excited_workflow/__init__.py`.
  - Make sure that the Zenodo integration is turned on before making a new release on GitHub.
  - Releases can be made through the GitHub web interface.
