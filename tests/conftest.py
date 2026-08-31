"""Test configuration.

Forces matplotlib onto the headless 'Agg' backend for the whole suite.

Several tests unpickle saved figures, which builds a canvas for whatever
backend is current. On Windows matplotlib sees a display and picks 'TkAgg',
and the Tcl/Tk shipped with the hosted CI runners is incomplete - Python
3.13 there fails with "Can't find a usable tk.tcl" because
'tcl/tk8.6/ttk/scale.tcl' is missing. Linux has no display, falls back to
'Agg' on its own, and never hit this. Nothing in the suite inspects a real
window, so 'Agg' is what these tests should have been using all along.

MPLBACKEND is set as well as calling 'use': the environment variable is what
matplotlib reads when it is first imported, and 'use' covers the case where
something imported pyplot before this module was loaded.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402  (must follow the MPLBACKEND assignment)

matplotlib.use("Agg", force=True)
