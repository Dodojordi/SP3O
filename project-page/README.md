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

- The Paper buttons link to [arXiv:2609.18708](https://arxiv.org/abs/2609.18708), so the repository does not carry a duplicate PDF.
- Keep the BibTeX block in `index.html` synchronized with the arXiv record.
- Web figures are rendered from the paper source PDFs into `assets/img/`.
