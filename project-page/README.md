# SP³O project page

Static, zero-build project page for **Rethinking Critic Learning in PPO: Understanding and Mitigating Value Flattening**.

## Preview locally

```bash
python -m http.server 8000 --directory project-page
```

Open <http://localhost:8000>.

## Deploy

The workflow in `.github/workflows/project-page.yml` deploys this directory to GitHub Pages. In repository **Settings → Pages**, select **GitHub Actions** as the source, then push to `main`.

Expected URL: <https://dodojordi.github.io/SP3O/>.

## Updating the paper

- Replace `assets/paper.pdf` with the latest manuscript.
- Update the BibTeX block in `index.html` once an arXiv identifier is available.
- Web figures are rendered from the paper source PDFs into `assets/img/`.
