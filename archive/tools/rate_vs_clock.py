    import os
    from glob import glob

    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    

    from daplis.functions import sensor_plot

    init_path = r"D:\LinoSPAD2\Data\board_NL11\Prague\rate_on_clock\Final_try"

    os.chdir(init_path)

    # freqs = [250, 200, 150, 110, 100, 90, 80, 70, 60, 50, 40]
    freqs_fold = glob("*/")
    freqs = []
    for freq in freqs_fold:
        freqs.append(int(freq.split()[0]))
    freqs.sort()
    
    folders = []
    
    for freq in freqs:
        try:
            folders.append(glob(f"{freq}*")[0])
        except:
            continue

    paths = [os.path.join(init_path, folder) for folder in folders]

    rates_dic = {}

    # for path in paths:
    #     plot_tmsp.plot_sensor_population(
    #         path,
    #         daughterboard_number="NL11",
    #         motherboard_number="#21",
    #         firmware_version="2212s",
    #         timestamps=300,
    #         include_offset=False,
    #         fit_peaks=True,
    #         show_fig=True,
    #         correct_pixel_addressing=True,
    #     )
        # os.chdir(path)
        # files = glob("*.dat")
        # for file in files:
        #     plot_tmsp.collect_data_and_apply_mask(
        #         file,
        #         daughterboard_number="NL11",
        #         motherboard_number="#33",
        #         firmware_version="2212s",
        #         timestamps=300,
        #         include_offset=False,
        #         save_to_file=True,
        #     )

        # plot_tmsp.plot_single_pix_hist(
        #     path,
        #     pixels=[12],
        #     daughterboard_number="NL11",
        #     motherboard_number="#33",
        #     firmware_version="2212s",
        #     timestamps=300,
        #     include_offset=False,
        #     fit_average=True,
        # )

    os.chdir(init_path)
    # Collect processed data from txt and plot
    
    total_coll_time = {}
    
    for i, freq in enumerate(freqs):
        folder_freq = glob(f"{freq}*")[0]
        # print(folder_freq)
        
        os.chdir(os.path.join(init_path, folder_freq))
        files = glob("*.dat")
        files.sort(key=os.path.getmtime)
        
        total_coll_time[f"{freq}"] = os.path.getctime(files[-1]) - os.path.getctime(files[0]) 
        
        rates_dic[f"{freq}"] = []
        os.chdir("senpop_data")
        
        for file in files:
            senpop_file = glob(f"*{file[:-4]}*.txt")[0]
            senpop_data = np.loadtxt(senpop_file)
            rate = np.max(senpop_data) / 500 * 250
            rates_dic[f"{freq}"].append(rate)
        os.chdir("../..")

    rates_df = pd.DataFrame.from_dict(rates_dic)

    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow-Green
        "#17becf",  # Turquoise
        "#1a1a1a",  # Black
        "#ff9896",  # Light Red
        "#aec7e8",  # Light Blue
        "#ffbb78",  # Light Orange
        "#98df8a",  # Light Green
        "#c5b0d5",  # Light Purple
        "#c49c94",  # Light Brown
        "#f7b6d2",  # Light Pink
        "#c7c7c7",  # Light Gray
        "#dbdb8d",  # Light Yellow-Green
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams.update({"font.size": 22})
    for index, row in rates_df.iterrows():
        ax.plot(freqs, row, 'o')
    ax.set_xlabel("External clock frequency [Hz]")
    ax.set_ylabel("Rate [kHz]")
    title = ax.set_title("160 file per frequency, all files")
    path_to_save = os.path.join(init_path, title.get_text())
    plt.savefig(f"{path_to_save}.png")

    # Plot the average with error
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams.update({"font.size": 22})
    means = rates_df.mean(axis=0)
    errors = rates_df.std(axis=0)
    ax.errorbar(freqs, means, fmt="o", yerr=errors)
    ax.set_xlabel("External clock frequency [Hz]")
    ax.set_ylabel("Average rate [kHz]")
    title = ax.set_title("160 file per frequency, average")
    path_to_save = os.path.join(init_path, title.get_text())
    plt.savefig(f"{path_to_save}.png")
    plt.show()

    # # Separating the two data sets by frequencies
    # freqs1 = [str(freq) for freq in [250, 200, 150, 100, 80, 50]]
    # freqs2 = [str(freq) for freq in [110, 90, 70, 60, 40]]
    
    # rates_df1 = rates_df[freqs1]
    # rates_df2 = rates_df[freqs2]
    
    # rates_df1.columns = rates_df1.columns.astype(int)
    # rates_df1.sort_index(axis=1, inplace=True)
    # rates_df2.columns = rates_df2.columns.astype(int)
    # rates_df2.sort_index(axis=1, inplace=True)

    # # Two data sets, all files
    # fig, ax = plt.subplots(figsize=(16, 10))
    # plt.rcParams.update({"font.size": 22})
    # for col in rates_df1.columns:
    #     x_values = [col] * len(rates_df1[col])  # Repeat column name to match y-values
    #     ax.plot(x_values, rates_df1[col], 'o', color="salmon", label="Second data set")
    # ax2 = ax.twinx()
    # for col in rates_df2.columns:
    #     x_values = [col] * len(rates_df2[col])  # Repeat column name to match y-values
    #     ax2.plot(x_values, rates_df2[col], 'o', color="teal", label="Second data set")
    # ax.set_xlabel("External clock frequency [Hz]")
    # ax.set_ylabel("Rate [kHz]", color="salmon")
    # ax2.set_ylabel("Rate [kHz]", color="teal")
    # title = ax.set_title("Two data sets, all files, separated")
    # path_to_save = os.path.join(init_path, title.get_text())
    # plt.savefig(f"{path_to_save}.png")
    # plt.show()
    
    # # Two data sets, averages
    
    # means1 = rates_df1.mean(axis=0)
    # errors1 = rates_df1.std(axis=0)
    # means2 = rates_df2.mean(axis=0)
    # errors2 = rates_df2.std(axis=0)
    
    # fig, ax = plt.subplots(figsize=(16, 10))
    # plt.rcParams.update({"font.size": 22})
    # for col in rates_df1.columns:
    #     ax.errorbar(col, means1[col], fmt="o", yerr=errors1[col], color="salmon", label="First data set")
    # ax2 = ax.twinx()
    # for col in rates_df2.columns:
    #     ax2.errorbar(col, means2[col], fmt="o", yerr=errors2[col], color="teal", label="Second data set")
    # ax.set_xlabel("External clock frequency [Hz]")
    # ax.set_ylabel("Rate [kHz]", color="salmon")
    # ax2.set_ylabel("Rate [kHz]", color="teal")
    # title = ax.set_title("Two data sets, averages, separated")
    # path_to_save = os.path.join(init_path, title.get_text())
    # plt.savefig(f"{path_to_save}.png")
    # plt.show()

    final_df = pd.DataFrame({'Ext clock frequency [Hz]':freqs, 'Average rate [Hz]':means.values/1000, 'Errors [Hz]':errors.values/1000}).round(2)
    final_df.set_index('Ext clock frequency [Hz]', inplace=True)
    final_df.sort_index(inplace=True)

    for row in final_df.iterrows():
        print(f"{row[0]}: {row[1].iloc[0]}\xb1{row[1].iloc[1]}")
    
final_df["Data taking time [s]"] = pd.Series(total_coll_time).values.round(2)

# Assign weights
weights = {
    'Average rate [Hz]': 0.35,
    'Errors [Hz]': 0.2,
    'Data taking time [s]': 0.45
}

# Calculate the score
final_df['Score'] = (
    weights['Average rate [Hz]'] * final_df['Average rate [Hz]'] +
    weights['Errors [Hz]'] * (1 - final_df['Errors [Hz]'] / final_df["Errors [Hz]"].max()) +
    weights['Data taking time [s]'] * (1 - final_df['Data taking time [s]'] / final_df['Data taking time [s]'].max())
).round(2)

# Find the frequency with the highest score
best_frequency = final_df.loc[final_df['Score'].idxmax()]

print("Best Frequency Details:")
print(best_frequency)

