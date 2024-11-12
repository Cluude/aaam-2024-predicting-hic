from tkinter.filedialog import askdirectory
from tqdm import trange
import time
import os
import warnings
from subfunctions import Subfunctions


# ignore FutureWarnings from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

def check_ratio(file_path):
    # get the filepath of the current working directory
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    datetime_stamp = time.strftime("%b %d, %Y at %I %M %p")
    output_folder = f"{base_path}/pt3_output/HIC output calculated on {datetime_stamp}/"
    os.mkdir(output_folder)
    temp_hic_file = f"{output_folder}temp_hic_curve.txt"
    
    # read the training and testing data sheets
    pulse_width_data = pd.read_excel(f"{file_path}/Training width output.xlsx")
    testing_data = pd.read_excel(f"{file_path}/Testing width output.xlsx")

    print(pulse_width_data)

    # calculate the ratios (mean and standard deviation)
    all_ratios = [pulse_width_data[x] for x in ["Haversine Width (ms)", "Sine Width (ms)", "Quadratic Width (ms)", "Triangular Width (ms)", "Rectangular Width (ms)"]]
    all_ratios = [x / pulse_width_data["Full Pulse Width (ms)"] for x in all_ratios]
    final_ratios = [x.mean() for x in all_ratios]
    ratio_sd = [x.std() for x in all_ratios]

    # save the ratios to an excel file
    final_ratios_xl = pd.DataFrame(final_ratios).T
    final_ratios_xl = pd.concat([final_ratios_xl, pd.DataFrame(ratio_sd).T], axis=0).set_axis(["Haversine", "Sine", "Quadratic", "Triangular", "Rectangular"], axis=1)
    final_ratios_xl['Metric'] = ['Mean', 'SD']
    print(final_ratios_xl)
    final_ratios_xl.to_excel(f"{output_folder}/Final ratios.xlsx", index=False)
    

    # calculate the HIC values for the testing data
    sf = Subfunctions(temp_hic_file_path=temp_hic_file)
    time.sleep(1)
    sf.setup_nhtsa_signal(base_path)

    headers = ['Test Number', 'Full Pulse Width (ms)', 'Peak Acceleration (g)', 'Full Pulse HIC',
               'Haversine HIC15', 'Sine HIC15', 'Quadratic HIC15', 'Triangular HIC15', 'Rectangular HIC15',
               'Haversine HIC36', 'Sine HIC36', 'Quadratic HIC36', 'Triangular HIC36', 'Rectangular HIC36']
    report = pd.DataFrame(columns=headers)
    for idx in trange(len(testing_data)):
        # get the test number, peak acceleration, true HIC15, and full pulse width
        test_number = testing_data["Test Number"][idx]
        peak_acceleration = testing_data["Peak Acceleration (g)"][idx]
        true_hic15 = testing_data["Full Pulse HIC"][idx]
        full_pulse_width = testing_data["Full Pulse Width (ms)"][idx] * 0.001

        # get the modeled widths
        modeled_widths = [full_pulse_width * x for x in final_ratios]

        # get the HIC values for each modeled width using the HIC calculator
        haversine_hic15, haversine_hic36 = sf.fit_curve_and_get_hic("Haversine", peak_acceleration, modeled_widths[0], 10_000)
        sine_hic15, sine_hic36 = sf.fit_curve_and_get_hic("Sine", peak_acceleration, modeled_widths[1], 10_000)
        quadratic_hic15, quadratic_hic36 = sf.fit_curve_and_get_hic("Quadratic", peak_acceleration, modeled_widths[2], 10_000)
        triangular_hic15, triangular_hic36 = sf.fit_curve_and_get_hic("Triangular", peak_acceleration, modeled_widths[3], 10_000)
        rectangular_hic15, rectangular_hic36 = sf.fit_curve_and_get_hic("Rectangular", peak_acceleration, modeled_widths[4], 10_000)
        
        # add the data to the report
        report = pd.concat([report, pd.DataFrame([[test_number, full_pulse_width * 1000, peak_acceleration, true_hic15,
                                                   haversine_hic15, sine_hic15, quadratic_hic15, triangular_hic15, rectangular_hic15,
                                                   haversine_hic36, sine_hic36, quadratic_hic36, triangular_hic36, rectangular_hic36]], columns=headers)], ignore_index=True)

    # save the report to an excel file
    report.to_excel(f"{output_folder}/HIC comparison output.xlsx", index=False)

    print("Done!")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    file_path = askdirectory(initialdir = f"{base_path}/pt2_output/", title = "Select a folder containing the data files (Data extracted on...).")
    check_ratio(file_path)