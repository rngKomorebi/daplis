import os

import numpy as np
from lmfit.models import GaussianModel, LinearModel
from matplotlib import pyplot as plt
from pyarrow import feather as ft

# Get the .feather file with delta t values
path = r"D:\LinoSPAD2\Data\B7d\Ne640\Ne640_test\139-167\delta_ts_data"

os.chdir(path)
ft_file = r"0000017670-0000017869.feather"


# Drop unnecessary values
data = ft.read_feather(ft_file)
data = data[data > -5e3]
data = data[data < 5e3].dropna()

# Calculate histogram
# Scott's rule for the number of bins
# TODO provide multiplier; if not provided, use the Scott's rule
number_of_bins = 3.5 * np.std(data.values) / len(data.values) ** (1 / 3)
multiplier = number_of_bins / (2500 / 140)
bins = np.arange(np.min(data), np.max(data), 2500 / 140 * 4)
counts, bins = np.histogram(data, bins)
# counts, bins = np.histogram(data, bins='scott')

# Normalize
counts = counts / np.median(counts)

bin_edges = (bins[:-1] + bins[1:]) / 2

# Composite fit: Gaussian for the peak + linear for bckg of random
# coincidences
model_peak = GaussianModel()
model_bckg = LinearModel()

# Guess the initial values of parameters
params_peak = model_peak.guess(counts, x=bin_edges)
params_bckg = model_bckg.guess(counts, x=bin_edges)
params = params_peak + params_bckg

params["amplitude"].min = 0
params["height"].min = 0
params["height"].max = 1
# Combine the models
model = model_peak + model_bckg

# Do the fitting
result = model.fit(counts, params, x=bin_edges, max_nfev=1000)

# Plot results
plt.rcParams.update({"font.size": 27})
fig, ((ax1, _), (ax2, ax3)) = plt.subplots(
    2,
    2,
    figsize=(16, 10),
    gridspec_kw={"width_ratios": [3, 1], "height_ratios": [3, 1]},
)

# Data + fit
ax1.plot(bin_edges, counts, ".", label="Data", color="rebeccapurple")
ax1.plot(bin_edges, result.best_fit, label="Gaussian fit", color="darkorange")
ax1.set_ylabel("Counts (-)")
ax1.yaxis.set_label_coords(-0.105, 0.5)
ax1.set_xticks([], [])
ax1.legend()

# Parameters
fit_params_text = "\n".join(
    [
        "Fit parameters",
        "                           ",
        f"$\sigma$: ({result.params['sigma'].value:.0f}"
        f"± {result.params['sigma'].stderr:.0f}) ps",
        f"$\mu$: ({result.params['center'].value:.0f}"
        f"± {result.params['center'].stderr:.0f}) ps",
        f"C: ({result.params['height'].value*100:.0f}"
        f"± {result.params['height'].stderr/result.params['height'].value*100:.0f}) %",
    ]
)

ax1.text(
    1.05,
    0.5,
    fit_params_text,
    transform=ax1.transAxes,
    fontsize=24,
    bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white"),
)

# Plot the residuals
result.plot_residuals(
    ax=ax2,
    title=" ",
    datafmt=".",
)
ax2.set_ylabel("Residuals (-)")
ax2.set_xlabel("$\Delta$t (ps)")
ax2_lines = ax2.get_lines()
ax2_lines[0].set_color("black")
ax2_lines[1].set_color("rebeccapurple")

# Plot the distribution of residuals with a Gaussian fit
residuals = counts - result.best_fit
counts_residuals, bins_residuals = np.histogram(residuals, bins=20)
bins_residuals_edges = (bins_residuals[:-1] + bins_residuals[1:]) / 2
ax3.plot(
    counts_residuals,
    bins_residuals_edges,
    ".",
    color="rebeccapurple",
)

model_residuals = GaussianModel()
params_residuals = model_residuals.guess(
    counts_residuals, x=bins_residuals_edges
)
result_residuals = model_residuals.fit(
    counts_residuals, params_residuals, x=bins_residuals_edges
)

ax3.plot(
    result_residuals.best_fit,
    bins_residuals_edges,
    color="darkorange",
    label="\n".join(
        [
            f"$\sigma$: {result_residuals.params['sigma'].value:.2f}",
            f"$\mu$: {round(result_residuals.params['center'].value, 3):.2f}",
        ]
    ),
)
y_limits = ax2.get_ylim()
ax3.set_ylim(y_limits)
ax3.set_yticks([], [])
ax3.legend(loc="best", fontsize=15)

fig.delaxes(_)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.subplots_adjust(wspace=0.05)

# Plot the report of the data fit
print(result.fit_report())


### Function


def fit_with_gaussian_fancy(
    path: str,
    ft_file: str = None,
    range_left: float = -5e3,
    range_right: float = 5e3,
):

    os.chdir(path)

    try:
        os.chdir("delta_ts_data")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Folder 'delta_ts_data' with the"
            "timestamps differences was not found."
        )

    # Drop unnecessary values
    data = ft.read_feather(ft_file)
    data = data[data > range_right]
    data = data[data < range_left].dropna()

    # Calculate histogram
    # Scott's rule for the number of bins
    # TODO provide multiplier; if not provided, use the Scott's rule
    number_of_bins = 3.5 * np.std(data.values) / len(data.values) ** (1 / 3)
    multiplier = number_of_bins / (2500 / 140)
    bins = np.arange(np.min(data), np.max(data), 2500 / 140 * 4)
    counts, bins = np.histogram(data, bins)
    # counts, bins = np.histogram(data, bins='scott')

    # Normalize
    counts = counts / np.median(counts)

    bin_edges = (bins[:-1] + bins[1:]) / 2

    # Composite fit: Gaussian for the peak + linear for bckg of random
    # coincidences
    model_peak = GaussianModel()
    model_bckg = LinearModel()

    # Guess the initial values of parameters
    params_peak = model_peak.guess(counts, x=bin_edges)
    params_bckg = model_bckg.guess(counts, x=bin_edges)
    params = params_peak + params_bckg

    params["amplitude"].min = 0
    params["height"].min = 0
    params["height"].max = 1
    # Combine the models
    model = model_peak + model_bckg

    # Do the fitting
    result = model.fit(counts, params, x=bin_edges, max_nfev=1000)

    # Plot results
    plt.rcParams.update({"font.size": 27})
    fig, ((ax1, _), (ax2, ax3)) = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        gridspec_kw={"width_ratios": [3, 1], "height_ratios": [3, 1]},
    )

    # Data + fit
    ax1.plot(bin_edges, counts, ".", label="Data", color="rebeccapurple")
    ax1.plot(
        bin_edges, result.best_fit, label="Gaussian fit", color="darkorange"
    )
    ax1.set_ylabel("Counts (-)")
    ax1.yaxis.set_label_coords(-0.105, 0.5)
    ax1.set_xticks([], [])
    ax1.legend()

    # Parameters
    fit_params_text = "\n".join(
        [
            "Fit parameters",
            "                           ",
            f"$\sigma$: ({result.params['sigma'].value:.0f}"
            f"± {result.params['sigma'].stderr:.0f}) ps",
            f"$\mu$: ({result.params['center'].value:.0f}"
            f"± {result.params['center'].stderr:.0f}) ps",
            f"C: ({result.params['height'].value*100:.0f}"
            f"± {result.params['height'].stderr/result.params['height'].value*100:.0f}) %",
        ]
    )

    ax1.text(
        1.05,
        0.5,
        fit_params_text,
        transform=ax1.transAxes,
        fontsize=24,
        bbox=dict(
            boxstyle="round,pad=0.5", edgecolor="black", facecolor="white"
        ),
    )

    # Plot the residuals
    result.plot_residuals(
        ax=ax2,
        title=" ",
        datafmt=".",
    )
    ax2.set_ylabel("Residuals (-)")
    ax2.set_xlabel("$\Delta$t (ps)")
    ax2_lines = ax2.get_lines()
    ax2_lines[0].set_color("black")
    ax2_lines[1].set_color("rebeccapurple")

    # Plot the distribution of residuals with a Gaussian fit
    residuals = counts - result.best_fit
    counts_residuals, bins_residuals = np.histogram(residuals, bins=20)
    bins_residuals_edges = (bins_residuals[:-1] + bins_residuals[1:]) / 2
    ax3.plot(
        counts_residuals,
        bins_residuals_edges,
        ".",
        color="rebeccapurple",
    )

    model_residuals = GaussianModel()
    params_residuals = model_residuals.guess(
        counts_residuals, x=bins_residuals_edges
    )
    result_residuals = model_residuals.fit(
        counts_residuals, params_residuals, x=bins_residuals_edges
    )

    ax3.plot(
        result_residuals.best_fit,
        bins_residuals_edges,
        color="darkorange",
        label="\n".join(
            [
                f"$\sigma$: {result_residuals.params['sigma'].value:.2f}",
                f"$\mu$: {round(result_residuals.params['center'].value, 3):.2f}",
            ]
        ),
    )
    y_limits = ax2.get_ylim()
    ax3.set_ylim(y_limits)
    ax3.set_yticks([], [])
    ax3.legend(loc="best", fontsize=15)

    fig.delaxes(_)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.subplots_adjust(wspace=0.05)

    # Plot the report of the data fit
    print(result.fit_report())


path = r"D:\LinoSPAD2\Data\B7d\Ne640\Ne640_test\139-167"
ft_file = r"0000017670-0000017869.feather"

fit_with_gaussian_fancy(path, ft_file=ft_file)
