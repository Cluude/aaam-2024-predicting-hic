from tkinter.filedialog import askdirectory
from numpy import NaN
from tqdm import trange, tqdm
import time
import os
import warnings
from subfunctions import Subfunctions
from scipy import integrate
from random import sample, seed

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

# set up the output folders
base_path = os.path.dirname(os.path.abspath(__file__)) + "/"
datetime_stamp = time.strftime("%b %d, %Y at %I %M %p")
output_folder = f"{base_path}/pt2_output/Data extracted on {datetime_stamp}/"
ascii_data_folder = f"{output_folder}/ASCII files/"
temp_hic_file = f"{output_folder}temp_hic_curve.txt"

os.makedirs(ascii_data_folder)

"""
This will ask the user to select the folder containing the datafiles generated using get_nhtsa_data.py.
Then it will read the list of occupants and instrumentation headers.
"""
file_path = askdirectory(initialdir = f"{base_path}/pt1_output/", title = "Select a folder containing the data files (...NHTSA cases found on...).")
print(f"Reading occupants and instrumentation data...")
occupants = pd.read_excel(file_path + "/occupants.xlsx")
instrumentation = pd.read_csv(file_path + "/instrumentation details.csv")

# change numeric seat position to words
occupants['SEATPOS'] = occupants['OCCLOC'].map({1: 'LEFT FRONT SEAT', 2: 'RIGHT FRONT SEAT', 3: 'RIGHT REAR SEAT', 4: 'LEFT REAR SEAT'})

"""
Filtering occupants based on the following criteria:
- The most recent year for each unique make and model
- Only drivers
- Only four-door sedans
- First head contact with airbag
- Vehicles with stated pretensioners and limiters, or vehicles newer 2008 when they were standard
"""
print(f"Before filtering: {len(occupants)} occupants.")
occupants.to_excel(f"{output_folder}/Unfiltered occupants.xlsx", index=False)

# For each unique MAKED and MODELD, take the most recent year.
unique_make_models = occupants[['MAKED', 'MODELD']].drop_duplicates()
new_occupants = pd.DataFrame(columns=occupants.columns)
for i in range(len(unique_make_models)):
    make, model = unique_make_models.iloc[i]
    year = occupants.query(f"MAKED == '{make}' and MODELD == '{model}'").sort_values('YEAR').iloc[-1]['YEAR']
    new_occupants = pd.concat([new_occupants, occupants.query(f"MAKED == '{make}' and MODELD == '{model}' and YEAR == {year}")])
occupants = new_occupants
print(f"After make, model, and year filtering: {len(occupants)} occupants.")
occupants.to_excel(f"{output_folder}/Filtered occupants.xlsx", index=False)

occupants = (occupants.query("OCCLOC == 1")                # Only drivers (OCCLOC == 1)
                     .query("BODYD == 'FOUR DOOR SEDAN'")  # Only sedans (BODYD == 'FOUR DOOR SEDAN')
                     .query("CNTRH1D == 'AIR BAG'"))       # Only first head contact with airbag

# Vehicles with stated pretensioners and limiters, or vehicles newer 2008 when they were standard.
occupants = occupants.query("(RSTCOM.str.contains('LIMITER') and RSTCOM.str.contains('PRETENSIONER')) or YEAR >= 2008").reset_index(drop=True)
print(f"After seat position, body type, impacted object, pretensioner, and limiter filtering: {len(occupants)} occupants.")


"""
Extracting the data for each occupant.
"""
sf = Subfunctions(temp_hic_file_path=temp_hic_file)

all_true_curves = []
all_fs = []
test_numbers = []
for i in trange(len(occupants)):
    # get test-specific parameters
    test_number = occupants.at[i, 'TSTNO']
    test_instrumentation = instrumentation.query(f"(TSTNO == {test_number})")
    test_seatloc = int(occupants.at[i, 'OCCLOC'])

    # 1. Get the head acceleration curve for the occupant.
    head_acceleration_curve, _, test_initial_velocity = sf.get_head_acceleration(test_number, test_instrumentation, test_seatloc, file_path)
    if type(head_acceleration_curve) == type(-1):
        continue

    # 2. Get the belt curve for the occupant.
    belt_curve = sf.get_belt_data(test_number, test_instrumentation, test_seatloc, file_path)
    if type(belt_curve) == type(-1):
        continue
    head_acceleration_curve = pd.merge(head_acceleration_curve, belt_curve, how='left', on='time')

    # velocity and displacement
    head_acceleration_curve = head_acceleration_curve.assign(x_cfc1000_velocity=lambda x: integrate.cumulative_trapezoid(x.x_cfc1000 * 9.81, x.time, initial=0) + (0.28 * test_initial_velocity))
    head_acceleration_curve = head_acceleration_curve.assign(x_cfc1000_displacement=lambda x: integrate.cumulative_trapezoid(x.x_cfc1000_velocity, x.time, initial=0))

    all_true_curves.append(head_acceleration_curve)
    all_fs.append(1 / (head_acceleration_curve.time[1] - head_acceleration_curve.time[0]))
    test_numbers.append(test_number)

    pure_resultant = head_acceleration_curve[['time', 'cfc1000']]
    sf.save_as_ascii(pure_resultant, ascii_data_folder, f"{test_number} 0 True")

print(f"After getting the curves: {len(all_true_curves)} curves.")

# 3. Split between training and testing data
seed(1)
indices = list(range(len(all_true_curves)))
training_indices = sample(indices, int(len(indices) * 0.5))

training_test_numbers = [test_numbers[idx] for idx in training_indices]
testing_test_numbers = [test_numbers[idx] for idx in indices if idx not in training_indices]
training_curves = [all_true_curves[idx] for idx in training_indices]
testing_curves = [all_true_curves[idx] for idx in indices if idx not in training_indices]

print(f"Split into {len(training_curves)} training and {len(testing_curves)} testing curves.")

sf.setup_nhtsa_signal(base_path)

"""
Iteratively get the best pulse width for each training curve.
"""
def training_curve_calculation(true_curve, fs, test_number):
    # 4. Calculate the full pulse width
    peak_acceleration, full_pulse_width, full_pulse_positions = sf.full_pulse_width(true_curve)
    full_pulse_values = [true_curve.query(f"time == {x}").cfc1000.values[0] for x in full_pulse_positions]
    
    # 5. Get the HIC of the true curve.
    true_curve[['time', 'cfc1000']].to_csv(temp_hic_file, sep='\t', index=False, header=False)
    true_hic, _ = sf.get_hic_using_nhtsa_signal(temp_hic_file)

    # 6. Get the best modeled pulse width for each model.
    haversine_duration = sf.recursive_get_model_width(peak_acceleration, fs, true_hic, 0, full_pulse_width * 2, "Haversine", 1000)
    sine_duration = sf.recursive_get_model_width(peak_acceleration, fs, true_hic, 0, full_pulse_width * 2, "Sine", 1000)
    quadratic_duration = sf.recursive_get_model_width(peak_acceleration, fs, true_hic, 0, full_pulse_width * 2, "Quadratic", 1000)
    triangular_duration = sf.recursive_get_model_width(peak_acceleration, fs, true_hic, 0, full_pulse_width * 4, "Triangular", 1000)
    rectangular_duration = sf.recursive_get_model_width(peak_acceleration, fs, true_hic, 0, full_pulse_width * 2, "Rectangular", 1000)

    full_pulse_width = round(full_pulse_width * 1000, 4)
    return [test_number, peak_acceleration, true_hic, full_pulse_width,
            haversine_duration, sine_duration, quadratic_duration, triangular_duration, rectangular_duration,
            *[x * 1000 for x in full_pulse_positions], *full_pulse_values, 'Train']
training_data = list(tqdm(map(training_curve_calculation, training_curves, all_fs, training_test_numbers), total=len(training_curves)))

headers = ['Test Number', 'Peak Acceleration (g)', 'Full Pulse HIC', 'Full Pulse Width (ms)',
           'Haversine Width (ms)', 'Sine Width (ms)', 'Quadratic Width (ms)', 'Triangular Width (ms)', 'Rectangular Width (ms)',
           'ti', 'tf', 'tpeak', 'txmax', 'tbelt', 'ati', 'atf', 'atpeak', 'atxmax', 'atbelt', 'Type']
training_data = pd.DataFrame(training_data, columns=headers)
training_data.to_excel(f"{output_folder}/Training width output.xlsx", index=False)

# 7. For the testing curves, save the Test Number, Peak Acceleration, Full Pulse HIC, and Full Pulse Width
def testing_curve_calculations(true_curve, test_number):
    peak_acceleration, full_pulse_width, full_pulse_positions = sf.full_pulse_width(true_curve)
    full_pulse_values = [true_curve.query(f"time == {x}").cfc1000.values[0] for x in full_pulse_positions]
    true_curve[['time', 'cfc1000']].to_csv(temp_hic_file, sep='\t', index=False, header=False)
    true_hic15, _ = sf.get_hic_using_nhtsa_signal(temp_hic_file)

    return [test_number, peak_acceleration, true_hic15, full_pulse_width * 1000,
                         NaN, NaN, NaN, NaN, NaN,
                         *[x * 1000 for x in full_pulse_positions], *full_pulse_values, 'Test']
testing_data = list(tqdm(map(testing_curve_calculations, testing_curves, testing_test_numbers), total=len(testing_curves)))

testing_data = pd.DataFrame(testing_data, columns=headers)  
testing_data.to_excel(f"{output_folder}/Testing width output.xlsx", index=False)

#combine into one report
combined_report = pd.concat([training_data, testing_data], ignore_index=True)
combined_report.to_excel(f"{output_folder}/Combined width output.xlsx", index=False)

# Plot curves
sf.plot_and_save_curves(datetime_stamp, output_folder, training_curves, testing_curves, combined_report)

print('Done!')

# Open the output folder for the user
os.startfile(output_folder)

# automatically run part 3 since part 2 takes hours.
from part2_check_ratio import check_ratio
check_ratio(output_folder)