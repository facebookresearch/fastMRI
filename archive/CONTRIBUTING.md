# Contributing to fastMRI

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

We actively welcome your pull requests. We check commits to the repository using
[CircleCI](https://circleci.com/), which runs linters and Python tests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style  

Linters and formatters we use include `black`, `flake8`, and `mypy`. Please include
type annotations in your code where reasonable. We only run `mypy` on the `fastmri`
directory. The `fastmri_examples`, `fastmri`, and `tests` directories are checked with
both `black` and `flake8`.

Configurations for the linters are in the root directory, `.flake8` for `flake8` and
`mypy.ini` for `mypy`. We use default configurations for `black`.

If all of the following commands pass without modifications or failures, your code
should be ready for CircleCI:

```bash
black fastmri_examples fastmri tests
mypy fastmri
flake8 fastmri_examples fastmri tests
```

## License

By contributing to fastMRI, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
