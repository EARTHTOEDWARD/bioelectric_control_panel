# Migration from Monorepo

The legacy folder `Bioelectric_Control_Panel/` is being split into a standalone repo `bioelectric-control-panel` with package name `bcp`.

## Preserve History (Option A: git subtree)

In the monorepo:

```
# Create a split branch with history of the folder
git subtree split --prefix=Bioelectric_Control_Panel -b bcp-split
```

In the new repo:

```
git init
# Add monorepo as a remote and pull the split history into main
git pull ../Control\ Panels bcp-split:main
```

## Preserve History (Option B: git filter-repo)

Requires `git-filter-repo`.

```
git filter-repo --path Bioelectric_Control_Panel/ --to-subdirectory-filter .
```

Then add standard scaffolding (pyproject, bcp/ package, API) or copy from the scaffolded folder in the monorepo.

## Post-Migration Cleanup

- Move `src/data_interface.py` -> `bcp/io/maes_interface.py`
- Move tests -> `tests/integration/` and mark `@pytest.mark.integration`
- Replace absolute paths with environment-configured Settings
- Add `.env.example` and `.gitignore`
- Add CI, pre-commit, and security policy

