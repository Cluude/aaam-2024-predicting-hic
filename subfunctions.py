import os
import numpy as np
import pandas as pd
import plotly.express as px
import pyautogui
import pyperclip
from scipy import signal
from tqdm import tqdm

class Subfunctions:
    """A collection of subfunctions to extract, analyze, and save NHTSA data.
    """

    def __init__(self, *, temp_hic_file_path: str = "", hic15_button_position: list = [0, 0], hic36_button_position: list = [0, 0]):
        self.temp_hic_file_path = temp_hic_file_path
        self.hic15_button_position = hic15_button_position
        self.hic36_button_position = hic36_button_position

    def get_datafiles(self, instrumentation: pd.DataFrame, n_curves: int):
        """Extract the correct datafiles, prefilter values, and initial velocities for the occupant.

        ## Args:
            instrumentation (pd.DataFrame): A dataframe containing the instrumentation headers.
            n_curves (int): Number of curves present.

        ## Returns:
            tuple: Numbers corresponding to the datafiles, prefilter value, initial velocity before impact.
        """
        primary_instrumentation = instrumentation.query("CHSTATD == 'PRIMARY'")
        redundant_instrumentation = instrumentation.query("CHSTATD == 'REDUNDANT'")
        if len(primary_instrumentation) >= n_curves:
            primary_instrumentation = primary_instrumentation.head(n_curves)
            datafile_numbers = primary_instrumentation['CURNO'].values
            prefilter = primary_instrumentation['PREFIL'].values
            initial_velocity = primary_instrumentation['INIVEL'].values
        elif len(redundant_instrumentation) >= n_curves:
            redundant_instrumentation = redundant_instrumentation.head(n_curves)
            datafile_numbers = redundant_instrumentation['CURNO'].values
            prefilter = redundant_instrumentation['PREFIL'].values
            initial_velocity = redundant_instrumentation['INIVEL'].values
        else:
            return tuple([[-1]] * n_curves)
        
        return (datafile_numbers, prefilter, initial_velocity)
    
    def get_head_acceleration(self, test_number: int, instrumentation: pd.DataFrame, seat_location: int, file_path: str) -> tuple[pd.DataFrame, str, float]:
        """Get occupant's head acceleration channels.
        
        ## Args:
            int: The test number.
            pd.DataFrame: The instrumentation data for the test.
            int: The seat location of the occupant.
            
        ## Returns:
            tuple:
                pd.DataFrame or int: The head acceleration data or -1 if no valid data.
                str: The prefilter values for the data channels
                float: The initial velocity of the occupant.
            """
        instrumentation = instrumentation.query(f"(SENATTD == 'HEAD CG') and (OCCLOC == {seat_location}) and (SENTYPD == 'ACCELEROMETER') and ((DASTATD == 'AS MEASURED') or (DASTATD == 'NO COMMENT'))")
        
        datafile_numbers, prefilter, initial_velocity = self.get_datafiles(instrumentation, 3)
        initial_velocity = initial_velocity[0] #only x initial velocity

        prefilter = ', '.join([str(x) for x in prefilter])
        datafile_strings = [str(x).zfill(3) for x in datafile_numbers]
        test_string = str(test_number).zfill(5)

        # Get the acceleration data
        ax = pd.read_table(f"{file_path}/instrumentation {test_number}/v{test_string}.{datafile_strings[0]}", header=None, names=['time', 'ax'])
        ay = pd.read_table(f"{file_path}/instrumentation {test_number}/v{test_string}.{datafile_strings[1]}", header=None, names=['time', 'ay'])
        az = pd.read_table(f"{file_path}/instrumentation {test_number}/v{test_string}.{datafile_strings[2]}", header=None, names=['time', 'az'])

        # CFC1000
        x_cfc1000, _ = self.cfc(ax.ax, ax.time, 1000)
        y_cfc1000, _  = self.cfc(ay.ay, ay.time, 1000)
        z_cfc1000, _  = self.cfc(az.az, az.time, 1000)
 
        # CFC600
        x_cfc600, _ = self.cfc(ax.ax, ax.time, 600)
        y_cfc600, _  = self.cfc(ay.ay, ay.time, 600)
        z_cfc600, _  = self.cfc(az.az, az.time, 600)

        # CFC180
        x_cfc180, _ = self.cfc(ax.ax, ax.time, 180)
        y_cfc180, _  = self.cfc(ay.ay, ay.time, 180)
        z_cfc180, _  = self.cfc(az.az, az.time, 180)

        # CFC60
        x_cfc60, _ = self.cfc(ax.ax, ax.time, 60)
        y_cfc60, _  = self.cfc(ay.ay, ay.time, 60)
        z_cfc60, _  = self.cfc(az.az, az.time, 60)

        head_acceleration = pd.DataFrame({'time': ax.time,
                                          'cfc1000': (x_cfc1000.values**2 + y_cfc1000.values**2 + z_cfc1000.values**2)**0.5,
                                          'x_cfc1000': x_cfc1000, 'y_cfc1000': y_cfc1000, 'z_cfc1000': z_cfc1000,
                                          'cfc600': (x_cfc600.values**2 + y_cfc600.values**2 + z_cfc600.values**2)**0.5,
                                          'x_cfc600': x_cfc600, 'y_cfc600': y_cfc600, 'z_cfc600': z_cfc600,
                                          'cfc180': (x_cfc180.values**2 + y_cfc180.values**2 + z_cfc180.values**2)**0.5,
                                          'x_cfc180': x_cfc180, 'y_cfc180': y_cfc180, 'z_cfc180': z_cfc180,
                                          'cfc60': (x_cfc60.values**2 + y_cfc60.values**2 + z_cfc60.values**2)**0.5,
                                          'x_cfc60': x_cfc60, 'y_cfc60': y_cfc60, 'z_cfc60': z_cfc60})
        return (head_acceleration, prefilter, initial_velocity)
    
    def get_belt_data(self, test_number: int, instrumentation: pd.DataFrame, seat_location: int, file_path: str) -> pd.DataFrame:
        """Get occupant's belt load channels.
        
        ## Args:
            int: The test number.
            pd.DataFrame: The instrumentation data for the test.
            int: The seat location of the occupant.
            
        ## Returns:
            pd.DataFrame: The belt load data.
        """
        instrumentation = instrumentation.query(f"(SENATTD.str.contains('SHOULDER')) and (OCCLOC == {seat_location}) and (CHLMAX > 0) and (YUNITSD == 'NEWTONS') and (((DASTATD == 'AS MEASURED') or (DASTATD == 'NO COMMENT')))")
        if len(instrumentation) == 0:
            return -1
        datafile_numbers, _, _ = self.get_datafiles(instrumentation, 1)
        
        test_string = str(test_number).zfill(5)
        datafile_strings = [str(x).zfill(3) for x in datafile_numbers]
        belt = pd.read_table(f"{file_path}/instrumentation {test_number}/v{test_string}.{datafile_strings[0]}", header=None, names=['time', 'belt_load'])
        belt['belt_load'], _ = self.cfc(belt.belt_load, belt.time, 60)
        return belt
    
    def get_angular_velocity(self, test_number: int, instrumentation: pd.DataFrame, seat_location: int, file_path: str) -> pd.DataFrame:
        """Get occupant's angular velocity channels.
        ## Args:
            int: The test number.
            pd.DataFrame: The instrumentation data for the test.
            int: The seat location of the occupant.
        ## Returns:
            pd.DataFrame: The belt load data.
        """
        instrumentation = instrumentation.query(f"(SENTYPD == 'ANGULAR VELOCITY TRANSDUCER') and (OCCLOC == {seat_location}) and (CHLMAX > 0) and (((DASTATD == 'AS MEASURED') or (DASTATD == 'NO COMMENT'))) and (AXISD == 'Y - LOCAL')")
        
        datafile_numbers, _, _ = self.get_datafiles(instrumentation, 1)
        
        test_string = str(test_number).zfill(5)
        datafile_strings = [str(x).zfill(3) for x in datafile_numbers]
        angular_velocity = pd.read_table(f"{file_path}/instrumentation {test_number}/v{test_string}.{datafile_strings[0]}", header=None, names=['time', 'angular_velocity'])
        angular_velocity['angular_velocity'], _ = self.cfc(angular_velocity.angular_velocity, angular_velocity.time, 1000)
        return angular_velocity
         
    def get_vehicle_data(self, file_path: str, test_number: int, instrumentation: pd.DataFrame) -> tuple:
        """Get the datafile numbers for the vehicle's acceleration channels.
        
        ## Args:
            int: The test number.
            pd.DataFrame: The instrumentation data for the test.
            
        ## Returns:
            tuple: (list: datafile numbers, list: prefilter values)
        """

        instrumentation_test = instrumentation_test.query(f"(SENTYPD == 'ACCELEROMETER') and (AXISD == 'X - GLOBAL')")
        instrumentation_test = instrumentation_test[instrumentation_test['DASTATD'].str.contains('AS MEASURED|NO COMMENT')]
        instrumentation_test = instrumentation_test[instrumentation_test['SENATTD'].str.contains('SEAT|SILL|FLOORPAN')]

        datafile_numbers, _, _ = self.get_datafiles(instrumentation, 1)

        test_string = str(test_number).zfill(5)
        vehicle_datafile_strings = [str(x).zfill(3) for x in datafile_numbers]
        left =  pd.read_table(f"{file_path}/instrumentation {test_number}/v{test_string}.{vehicle_datafile_strings[0]}", header=None, names=['time', 'left'])
        right = pd.read_table(f"{file_path}/instrumentation {test_number}/v{test_string}.{vehicle_datafile_strings[0]}", header=None, names=['time', 'right'])
        vehicle_average = (left.left + right.right) / 2

        vehicle_resultant = pd.DataFrame({'time': left.time, 'vehicle': vehicle_average})
        return vehicle_resultant
    
    def get_neck_data(self, test_number: int, instrumentation: pd.DataFrame, seat_location: int, file_path: str) -> tuple:
        """Get the data for the occupants neck x load and y moment channels.
        
        ## Args:
            int: The test number.
            pd.DataFrame: The instrumentation data for the test.
            
        ## Returns:
            pd.DataFrame with columns: time (for merging), neck x load, and neck y moment.
        """
        instrumentation_x_load = instrumentation.query(f"(OCCLOC == {seat_location}) and ((DASTATD == 'AS MEASURED') or (DASTATD == 'COMPUTED')) and ((SENATTD == 'NECK - UPPER') or (SENATTD == 'NECK  UPPER')) and (AXISD == 'X - LOCAL') and (YUNITSD == 'NEWTONS')")
        instrumentation_y_moment = instrumentation.query(f"(OCCLOC == {seat_location}) and ((DASTATD == 'AS MEASURED') or (DASTATD == 'COMPUTED')) and ((SENATTD == 'NECK - UPPER') or (SENATTD == 'NECK  UPPER')) and (AXISD == 'Y - LOCAL') and (YUNITSD == 'NEWTON-METERS')")
        
        if (len(instrumentation_x_load) == 0) or (len(instrumentation_y_moment) == 0):
            return -1
        
        datafile_numbers_x_load, _, _ = self.get_datafiles(instrumentation_x_load, 1)
        datafile_numbers_y_moment, _, _ = self.get_datafiles(instrumentation_y_moment, 1)

        datafile_strings_x_load = [str(x).zfill(3) for x in datafile_numbers_x_load]
        datafile_strings_y_moment = [str(x).zfill(3) for x in datafile_numbers_y_moment]
        test_string = str(test_number).zfill(5)

        x_load = pd.read_table(f"{file_path}/instrumentation {test_number}/v{test_string}.{datafile_strings_x_load[0]}", header=None, names=['time', 'x_load'])
        y_moment = pd.read_table(f"{file_path}/instrumentation {test_number}/v{test_string}.{datafile_strings_y_moment[0]}", header=None, names=['time', 'y_moment'])

        neck_data = pd.DataFrame({'time': x_load.time, 'neck_x_load': x_load.x_load, 'neck_y_moment': y_moment.y_moment})
        return neck_data

    def cfc(self, y: pd.Series, t: pd.Series, cfc_value: float) -> tuple[pd.Series, float]:
        """Run a CFC filter on the data.
        
        ## Args:
            pd.Series: Acceleration data to be filtered.
            pd.Series: Time data of the same length as the acceleration data.
            float: The channel frequency class for the filter (cutoff frequency = cfc * (5/3))
            
        ## Returns:
            tuple:
                pd.Series: The filtered acceleration data.
                float: The sampling frequency of the data.
        """
        fs = 1 / (t[1] - t[0])
        cutoff = cfc_value * (5/3)
        sos = signal.butter(2, cutoff, btype='low', fs=fs, output='sos')
        filtered_signal = pd.Series(signal.sosfiltfilt(sos, y.values))
        return (filtered_signal, fs)
    
    def full_pulse_width(self, curve: pd.DataFrame) -> tuple[float, float, list[float]]:
        """ Calculate the pulse width of the full curve by fft.
        
        ## Args:
            pd.DataFrame: The resultant data to be analyzed. Must have columns 'time' and 'cfc1000'.
            
        ## Returns:
            tuple: (float: peak acceleration, float: pulse width, list: [pulse start time, pulse end time, peak time])
        """
        # Get pulse start (2g threshold)
        pulse_start = curve.time.values[curve.cfc1000.values > 2][0]

        # Get pulse end
        max_displacement_idx = curve.x_cfc1000_displacement.idxmax()
        time_max_displacement = curve.time[max_displacement_idx]

        peak_index = curve.query(f'time < {time_max_displacement}').cfc1000.idxmax()
        peak_time = curve.time[peak_index]
        peak_value = curve.cfc1000[peak_index]

        trunc_dataset = curve.query(f"(time > {peak_time}) and (belt_load <= 0)").reset_index(drop=True)
        end_belt = trunc_dataset.time.values[0] if len(trunc_dataset) > 0 else -0.001
        
        # ----- Local min between max displacement and belt -----
        trunc_curve = curve.query(f"(time >= {time_max_displacement}) and (time <= {end_belt})")
        min_index = trunc_curve.cfc1000.idxmin()
        pulse_end = trunc_curve.time[min_index]

        
        return (peak_value, pulse_end - pulse_start, [pulse_start, pulse_end, peak_time, time_max_displacement, end_belt])
    
    def save_as_ascii(self, data, folder, filename):
        data.to_csv(f"{folder}/{filename}.txt", sep='\t', index=False, header=False)

    def slider_helper(self, fig, labels):
        fig.data = fig.data[1:]
        steps = []
        n_lines = len(fig.data)//len(labels)
        for i in range(len(labels)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}],
                label=f"Test {labels[i]}"
            )
            step["args"][0]["visible"][(i * n_lines):(i * n_lines) + n_lines] = [True] * n_lines
            steps.append(step)
        sliders = [dict(active=0, pad={"t": 50}, steps=steps)]
        fig.update_layout(sliders=sliders)

        return fig
    
    def plot_and_save_curves(self, title: str, output_folder: str, training_curves: pd.DataFrame, testing_curves: pd.DataFrame, combined_report: pd.DataFrame):
        """Create a single plotly scatter plot of all the data curves and save it to an html file.

        ## Args:
            title (str): Scatterplot title.
            output_folder (str): Folder to save the html file.
            training_curves (pd.DataFrame): Training curves.
            testing_curves (pd.DataFrame): Testing curves.
            combined_report (pd.DataFrame): Combined report with all curve data.
        
        ## Returns:
            None
        """
        fig = px.scatter(title=title)

        print("Plotting curves...")
        all_curves = training_curves + testing_curves
        labels = combined_report['Test Number'].values

        for idx, curve in enumerate(tqdm(all_curves)):
            test_data = combined_report.iloc[idx]
            # Add true curve
            fig = fig.add_scatter(x = 1000 * curve.time, y = curve.cfc1000, mode = 'lines', name = f"{test_data['Test Number']}", visible=True if idx == 0 else False)
            
            #ti
            fig = fig.add_scatter(x=[test_data.ti, test_data.ti], y=[0, curve.cfc1000.max()], mode='lines+text', name=f"True {labels[idx]} Start", text=["", f"t<sub>i</sub>"], textposition="top center",
                                    line=dict(dash='dash'), visible=True if idx == 0 else False)
            #tf
            fig = fig.add_scatter(x=[test_data.tf, test_data.tf], y=[0, curve.cfc1000.max()], mode='lines+text', name=f"True {labels[idx]} End", text=["", f"t<sub>f</sub>"], textposition="top center",
                                    line=dict(dash='dash'), visible=True if idx == 0 else False)
            #tpeak
            fig = fig.add_scatter(x=[test_data.tpeak, test_data.tpeak], y=[0, curve.cfc1000.max()], mode='lines+text', name=f"True {labels[idx]} Peak", text=["", f"t<sub>peak</sub>"], textposition="top center",
                                    line=dict(dash='dash'), visible=True if idx == 0 else False)
            #txmax
            fig = fig.add_scatter(x=[test_data.txmax, test_data.txmax], y=[0, curve.cfc1000.max() * 0.8], mode='lines+text', name=f"True {labels[idx]} Max Displacement", text=["", f"t<sub>xmax</sub>"], textposition="top center",
                                    line=dict(dash='dash'), visible=True if idx == 0 else False)
            #tbelt
            fig = fig.add_scatter(x=[test_data.tbelt, test_data.tbelt], y=[0, curve.cfc1000.max() * 1.2], mode='lines+text', name=f"True {labels[idx]} Belt", text=["", f"t<sub>belt</sub>"], textposition="top center",
                                    line=dict(dash='dash'), visible=True if idx == 0 else False)
        
        fig = self.slider_helper(fig, labels)

        y_max = max([max(curve.cfc1000) for curve in all_curves])
        y_min = min([min(curve.cfc1000) for curve in all_curves])
        x_max = max([max(curve.time) for curve in all_curves])
        x_min = min([min(curve.time) for curve in all_curves])
        fig.update_layout(yaxis_range=[y_min, y_max * 1.1], xaxis_range=[x_min * 1000, x_max * 1000])

        fig.write_html(f"{output_folder}/{title}.html")

    def setup_nhtsa_signal(self, base_path: str) -> list:
        """Open the NHTSA Signal HIC software and get the positions of the HIC15 and HIC36 buttons.

        ## Args:
            base_path (str): The current working directory with the NCAP datafiles.

        ## Returns:
            None
        """
        for i in range(3):
            print(f"Starting NHTSA Signal in {3 - i} seconds...")
            pyautogui.sleep(1)
        print("Starting now.")

        # Windows key
        pyautogui.press('win')
        pyautogui.sleep(0.5)

        # type HIC.exe
        pyautogui.write('HIC.exe')
        pyautogui.sleep(0.5)

        # Press enter
        pyautogui.press('enter')
        pyautogui.sleep(0.5)

        # Press esc
        pyautogui.press('esc')
        pyautogui.sleep(0.5)

        # Fullscreen the window
        pyautogui.hotkey('alt', 'space')
        pyautogui.sleep(0.1)
        pyautogui.press('r')
        pyautogui.sleep(0.2)
        pyautogui.hotkey('alt', 'space')
        pyautogui.sleep(0.1)
        pyautogui.press('x')
        pyautogui.sleep(2)

        # get position of hic15 and hic36 buttons
        hic15_button_pos = pyautogui.locateCenterOnScreen(f"{base_path}/nhtsa_hic_15.png")
        hic36_button_pos = pyautogui.locateCenterOnScreen(f"{base_path}/nhtsa_hic_36.png")

        self.hic15_button_position = hic15_button_pos
        self.hic36_button_position = hic36_button_pos
        
    def get_hic_using_nhtsa_signal(self, file_path: str):
        """Get the HIC15 and HIC36 values using the NHTSA Signal software.

        ## Args:
            file_path (str): path to the file to be analyzed.
            
        ## Returns:
            tuple: (float: HIC15 value, float: HIC36 value)
        """
        def transfer_hic_value():
            # Access menu
            pyautogui.doubleClick(self.hic15_button_position[0], self.hic15_button_position[1] + 50)
            pyautogui.sleep(0.1)

            # Press tab
            pyautogui.press('tab')
            pyautogui.sleep(0.05)

            # Copy (ctrl + c)
            pyautogui.hotkey('ctrl', 'c')

            # Get clipboard data
            subtitle = pyperclip.paste()

            # Press esc
            pyautogui.press('esc')
            pyautogui.sleep(0.1)
            return subtitle
        
        # Click into window (213, 16)
        pyautogui.click(self.hic15_button_position[0], self.hic15_button_position[1] + 50)
        pyautogui.sleep(0.1)

        # Control + O to open file
        pyautogui.hotkey('ctrl', 'o')
        pyautogui.sleep(0.1)

        # Press tab
        pyautogui.press('tab')
        pyautogui.sleep(0.05)

        # Type full file path with extension
        pyautogui.write(file_path)
        pyautogui.sleep(0.1)

        # Press tab 3 times
        for _ in range(3):
            pyautogui.press('tab')
            pyautogui.sleep(0.02)

        # Press down arrow
        pyautogui.press('down')
        pyautogui.sleep(0.02)

        # Press tab
        pyautogui.press('tab')
        pyautogui.sleep(0.02)

        # press enter
        pyautogui.press('enter')
        pyautogui.sleep(0.1)

        # HIC15 button
        pyautogui.click(*self.hic15_button_position)
        pyautogui.sleep(0.1)
        hic15 = transfer_hic_value()
        hic15_value = float(hic15.split(' ')[4])

        hic36_value = 0

        return (hic15_value, hic36_value)
    
    def fit_curve_and_get_hic(self, curve_type: str, peak_acceleration: float, duration: float, fs: float) -> float:
        """Fit a curve to the given parameters and get the HIC15 and HIC36 values.

        Args:
            curve_type (str): Name of the pulse function to fit.
            peak_acceleration (float): Peak acceleration of the pulse.
            duration (float): Duration of the pulse.
            fs (float): Sampling frequency to use.

        Returns:
            list: (hic 15, hic 36)
        """
        pulse_time = np.linspace(0, duration, round(duration * fs))
        if curve_type == "Haversine":
            modeled_trace = peak_acceleration * (np.sin(np.pi * pulse_time / duration))**2
        elif curve_type == "Sine":
            modeled_trace = peak_acceleration * np.sin(np.pi * pulse_time / duration)
        elif curve_type == "Quadratic":
            modeled_trace = -4 * peak_acceleration / (duration**2) * (pulse_time - duration / 2)**2 + peak_acceleration
        elif curve_type == "Triangular":
            modeled_trace = peak_acceleration * (1 - abs(((2 * pulse_time) / duration)-1))
        elif curve_type == "Rectangular":
            modeled_trace = np.repeat(peak_acceleration, len(pulse_time))
        else:
            print(f"Invalid curve type: {curve_type} with type: {type(curve_type)}. Comparison {curve_type == "Haversine"}.")
        pd.DataFrame([pulse_time, modeled_trace]).T.to_csv(self.temp_hic_file_path, sep='\t', index=False, header=False)
        hic15, hic36 = self.get_hic_using_nhtsa_signal(self.temp_hic_file_path)
        hic36 = 0
        return (hic15, hic36)

    def recursive_get_model_width(self, peak_acceleration: float, fs: float, true_hic: float, lower_duration: float, upper_duration: float, curve_type: str, iterations: int):
        """Recursively find the best pulse width for the given HIC value.

        Args:
            peak_acceleration (float): Peak acceleration value for the pulse.
            fs (float): Sampling frequency of the data.
            true_hic (float): The true HIC value to match.
            lower_duration (float): lower bound of where the ideal pulse width could be.
            upper_duration (float): upper bound of where the ideal pulse width could be.
            curve_type (str): The type of curve to fit.
            iterations (int): Maximum number of iterations to run.

        Returns:
            float: Ideal pulse width for the given HIC value. -1 if error.
        """
        next_duration = (lower_duration + upper_duration) / 2

        # Generate the with the middle duration and get the HIC
        middle_hic, _ = self.fit_curve_and_get_hic(curve_type, peak_acceleration, next_duration, fs)
        
        tqdm.write(f"Lower/Upper: {round(lower_duration * 1000, 4)}ms/{round(upper_duration * 1000, 4)}ms, Difference: {round((upper_duration - lower_duration) * 1000, 4)}ms, True/Check HIC: {true_hic}/{middle_hic}")

        hic_accuracy = 0.001
        duration_accuracy = 0.000_001

        if (abs(true_hic - middle_hic) < hic_accuracy):
            # tqdm.write(f"Found Duration: {next_duration} giving HIC: {middle_hic} with error {abs(true_hic - middle_hic)}.")
            # tqdm.write(f"Stopped because HIC error is {abs(true_hic - middle_hic)}.")
            return round(next_duration * 1000, 4)
        if abs(lower_duration - upper_duration) < duration_accuracy:
            # tqdm.write(f"Found Duration: {next_duration} giving HIC: {middle_hic} with error {abs(true_hic - middle_hic)}.")
            # tqdm.write(f"Stopped because duration difference is {abs(lower_duration - upper_duration)}.")
            return round(next_duration * 1000, 4)
        if iterations <= 0:
            tqdm.write(f"Found Duration: {next_duration} giving HIC: {middle_hic} with error {abs(true_hic - middle_hic)}.")
            tqdm.write(f"Stopped because iterations are {100 - iterations}.")
            return round(next_duration * 1000, 4)
        
        if true_hic > middle_hic:
            return self.recursive_get_model_width(peak_acceleration, fs, true_hic, next_duration, upper_duration, curve_type, iterations - 1)
        elif true_hic < middle_hic:
            return self.recursive_get_model_width(peak_acceleration, fs, true_hic, lower_duration, next_duration, curve_type, iterations - 1)
        else:
            tqdm.write("Error in recursive_get_model_width")
            return -1