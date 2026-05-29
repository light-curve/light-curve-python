"""MkDocs hooks for light-curve-python docs."""

_REPO = "light-curve/light-curve-python"
_BRANCH = "main"


def on_page_content(html, page, config, files, **kwargs):
    """Inject download and Google Colab buttons at the top of every notebook page."""
    if not page.file.src_path.endswith(".ipynb"):
        return html

    nb_path = f"docs/{page.file.src_path}"
    download_url = (
        f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}/{nb_path}"
    )
    colab_url = (
        f"https://colab.research.google.com/github/{_REPO}/blob/{_BRANCH}/{nb_path}"
    )

    buttons = (
        f'<p class="lc-nb-buttons">'
        f'<a href="{download_url}" class="md-button" download>'
        f"⬇️ Download notebook</a> "
        f'<a href="{colab_url}" class="md-button md-button--primary" target="_blank" rel="noopener">'
        f"▶️ Run in Google Colab</a>"
        f"</p>\n"
    )
    return buttons + html
