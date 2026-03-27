from pathlib import Path

from matplotlib import pyplot as plt

# Path to the bundled matplotlib style sheet.
# To override: plt.style.use("your_style") after importing daplis.
style_path = str(Path(__file__).parent / "daplis.mplstyle")
plt.style.use(style_path)
