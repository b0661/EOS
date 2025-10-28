% SPDX-License-Identifier: Apache-2.0
(release-page)=

# Release Process

This document describes how to prepare and publish a new release **via a Pull Request from a fork**,
and how to set a **development version** after the release.

## ✅ Overview of the Process

| Step | Actor       | Action |
|------|-------------|--------|
| 1    | Contributor | Prepare a release branch **in your fork** using Commitizen |
| 2    | Contributor | Open a **Pull Request to upstream** (`Akkudoktor-EOS/EOS`) |
| 3    | Maintainer  | Review and **merge the release PR** |
| 4    | Maintainer  | Create the **GitHub Release and tag** |
| 5    | Maintainer  | Set the **development version marker** via a follow-up PR |

## 🔄 Detailed Workflow

### 1️⃣ Contributor: Prepare the Release in Your Fork

#### Clone and sync your fork

```bash
git clone https://github.com/<your-username>/EOS
cd EOS
git remote add eos https://github.com/Akkudoktor-EOS/EOS

git fetch eos
git checkout main
git pull eos main
````

#### Create the release branch

```bash
git checkout -b release/vX.Y.Z
```

#### Bump the version information

At least update

- pyproject.toml
- haaddon/config.yaml
- src/akkudoktoreos/core/version.py
- src/akkudoktoreos/data/default.config.json
- Makefile

and the generated documentation:

```bash
make bump VERSION=0.1.0+dev NEW_VERSION=X.Y.Z
make gen-docs
```

You may check the changes by:

```bash
git diff
```

#### Create a new CHANGELOG.md entry

Edit CHANGELOG.md

#### Create the new release commit

```bash
git add pyproject.toml src/akkudoktoreos/core/version.py \
    src/akkudoktoreos/data/default.config.json Makefile CHANGELOG.md
git commit -s -m "chore(release): Release vX.Y.Z"
```

#### Push the branch to your fork

```bash
git push --set-upstream origin release/vX.Y.Z
```

### 2️⃣ Contributor: Open the Release Pull Request

| From                                 | To                        |
| ------------------------------------ | ------------------------- |
| `<your-username>/EOS:release/vX.Y.Z` | `Akkudoktor-EOS/EOS:main` |

**PR Title:**

```text
chore(release): release vX.Y.Z
```

**PR Description Template:**

```markdown
## Release vX.Y.Z

This pull request prepares release **vX.Y.Z**.

### Changes
- Version bump
- Changelog update

### Changelog Summary
<!-- Copy key highlights from CHANGELOG.md here -->

See `CHANGELOG.md` for full details.
```

### 3️⃣ Maintainer: Review and Merge the Release PR

**Review Checklist:**

- ✅ Only version files and `CHANGELOG.md` are modified
- ✅ Version numbers are consistent
- ✅ Changelog is complete and properly formatted
- ✅ No unrelated changes are included

**Merge Strategy:**

- Prefer **Merge Commit** (or **Squash Merge**, per project preference)
- Use commit message: `chore(release): Release vX.Y.Z`

### 4️⃣ Maintainer: Publish the GitHub Release

1. Go to **GitHub → Releases → Draft a new release**
2. **Choose tag** → enter `vX.Y.Z` (GitHub creates the tag on publish)
3. **Release title:** `vX.Y.Z`
4. **Paste changelog entry** from `CHANGELOG.md`
5. Optionally enable **Set as latest release**
6. Click **Publish release** 🎉

### 5️⃣ Maintainer: Prepare the Development Version Marker

**Sync local copy:**

```bash
git fetch eos
git checkout main
git pull eos main
```

**Create a development version branch:**

```bash
git checkout -b release/vX.Y.Z_dev
```

**Set development version marker manually:**

```bash
make bump VERSION=X.Y.Z NEW_VERSION=X.Y.Z+dev
make gen-docs
```

```bash
git add pyproject.toml src/akkudoktoreos/core/version.py \
    src/akkudoktoreos/data/default.config.json Makefile
git commit -s -m "chore: set development version marker X.Y.Z+dev"
```

```bash
git push --set-upstream origin release/vX.Y.Z_dev
```

### 6️⃣ Maintainer (or Contributor): Open the Development Version PR

| From                                     | To                        |
| ---------------------------------------- | ------------------------- |
| `<your-username>/EOS:release/vX.Y.Z_dev` | `Akkudoktor-EOS/EOS:main` |

**PR Title:**

```text
chore: development version vX.Y.Z+dev
```

**PR Description Template:**

```markdown
## Development version vX.Y.Z+dev

This pull request marks the repository as back in active development.

### Changes
- Set version to `vX.Y.Z+dev`

No changelog entry is needed.
```

### 7️⃣ Maintainer: Review and Merge the Development Version PR

**Checklist:**

- ✅ Only version files updated to `+dev`
- ✅ No unintended changes

**Merge Strategy:**

- Merge with commit message: `chore: development version vX.Y.Z+dev`

## ✅ Quick Reference

| Step | Actor | Action |
| ---- | ----- | ------ |
| **1. Prepare release branch** | Contributor | Bump version & changelog via Commitizen |
| **2. Open release PR** | Contributor | Submit release for review |
| **3. Review & merge release PR** | Maintainer | Finalize changes into `main` |
| **4. Publish GitHub Release** | Maintainer | Create tag & notify users |
| **5. Prepare development version branch** | Maintainer | Set development marker |
| **6. Open development PR** | Maintainer (or Contributor) | Propose returning to development state |
| **7. Review & merge development PR** | Maintainer | Mark repository as back in development |
