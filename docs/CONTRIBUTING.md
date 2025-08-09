#  Contributing to OutBoxML

Thanks for taking the time to contribute! 
This guide outlines how to get involved and contribute to the codebase.

---

## Project Setup

1. **Fork the repository**  
2. **Clone your fork**

```bash
git clone git@github.com:SVSemyonov/outboxml.git
cd outboxml/outboxml


## Branching Strategy

- `main` → stable, production-ready
- `develop` → active development
- `feature/your-feature` → new features
- `bugfix/your-fix` → bug fixes

Create a branch before starting work:

```bash
git checkout -b feature/your-feature
```

---

## Contribution Types

You're welcome to contribute in many ways:

- Fix bugs
- Add new features
- Improve documentation
- Add or fix tests
- Refactor code

---

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(scope): short description
```

**Examples:**

- `feat(auth): add login endpoint`
- `fix(ui): button alignment issue`
- `docs: update README formatting`

---

## Pull Request Checklist

Before submitting your PR:

- [ ] PR title uses Conventional Commits
- [ ] All existing and new tests pass (`python -m unittest discover`)
- [ ] The branch is up to date with `main`
- [ ] Changes are documented (if needed)

To open a PR:

```bash
git push origin feature/your-feature
```

Then create a pull request at:
https://github.com/SVSemyonov/outboxml/compare

---


##  Questions?

Feel free to:

- Open an issue: https://github.com/SVSemyonov/outboxml/issues
- Ask in Discussions: https://github.com/SVSemyonov/outboxml/discussions
