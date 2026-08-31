"""DAPLIS: Data Analysis Package for LInoSpad.

On import, daplis applies its matplotlib house style through komorebi_mpl, so
plots look consistent out of the box.

Overriding it (your choice always wins):

    import daplis  # noqa: F401  (applies the default style on import)
    import komorebi_mpl

    komorebi_mpl.use("night_wave")   # or any registered style, or "default"

Because the plotting functions never touch rcParams themselves, whatever style
is active when they draw is what you get — set it once, after the imports.

To change the daplis default look, either edit ``_DEFAULT_STYLE`` below or edit
``komorebi_mpl/styles/daplis.mplstyle``.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

#: Package version, taken from the installed distribution metadata. There is no
#: version string in the source tree - setuptools_scm derives it from the git
#: tag at build time, so this is the single place to read it back from.
try:
    __version__ = _version("daplis")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0.dev0"

# Name of the registered komorebi_mpl style applied on import. Rewrite freely.
_DEFAULT_STYLE = "daplis"

#: Path to the bundled style sheet, kept for backwards compatibility with
#: ``plt.style.use(daplis.style_path)``. ``None`` if komorebi_mpl is missing.
style_path = None

try:
    import komorebi_mpl as _komorebi_mpl

    style_path = _komorebi_mpl.style_path(_DEFAULT_STYLE)
    _komorebi_mpl.apply_default(_DEFAULT_STYLE)
except Exception:
    # Styling is optional and must never block importing the analysis code
    # (e.g. komorebi_mpl not installed, or the style name not yet registered).
    pass
