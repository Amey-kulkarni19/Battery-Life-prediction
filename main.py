import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter 
from data_extraction import extract_discharge_data, extract_capacities
from Linear_regression import linear_regression, predict, manual_standard_scaler, mean_absolute_error, root_mean_squared_error
from plots import plot_predictions, plot_voltage_vs_time, plot_capacity_vs_time, plot_combined_degradation

def main():
    files = {
        'Battery B0005': 'B0005.mat',
        'Battery B0006': 'B0006.mat',
        'Battery B0007': 'B0007.mat',
        'Battery B0018': 'B0018.mat'
    }
    results = {}

    # Lists to store combined degradation data for all batteries
    all_capacity_degradations = []
    all_voltage_degradations = []
    all_battery_labels = []

    all_capacity_predicted = []
    all_voltage_predicted = []
    
    all_actual_voltages = []
    all_predicted_voltage_deg = []
    all_actual_capacities = []
    all_predicted_capacity_deg = []

    for label, file_path in files.items():
        try:
            # Extract discharge data and capacities from .mat files
            discharge_cycles = extract_discharge_data(file_path)
            capacities = extract_capacities(discharge_cycles)

            # Create cycle number array
            cycle_numbers = np.arange(len(capacities))

            # Assume the initial values are at cycle 0
            initial_capacity = capacities[0]
            initial_voltage = discharge_cycles[0]['data']['Voltage_measured'][0][0].flatten()[0]

            # Extract features (cycle number, temperature, etc.)
            temperatures = np.array([cycle['data']['Temperature_measured'][0][0].flatten()[0] for cycle in discharge_cycles])
            voltages = np.array([cycle['data']['Voltage_measured'][0][0].flatten()[0] for cycle in discharge_cycles])

            smooth_voltages = savgol_filter(voltages, window_length=90, polyorder=2)
            voltages = smooth_voltages

            # Calculate the target variables
            remaining_life = len(capacities) - cycle_numbers  # Remaining life: total cycles - current cycle
            capacity_degradation = initial_capacity - capacities  # Capacity degradation: initial - current capacity
            voltage_degradation = initial_voltage - voltages  # Voltage degradation: initial voltage - current voltage

            # Stack the features into a matrix
            features = np.column_stack((cycle_numbers, temperatures, voltages))

            # Normalize the features
            features_scaled = manual_standard_scaler(features)

            # Predict remaining useful life (RUL)
            beta_rul = linear_regression(features_scaled, remaining_life)
            predicted_rul = predict(features_scaled, beta_rul)

            # Predict capacity degradation
            beta_capacity = linear_regression(features_scaled, capacity_degradation)
            predicted_capacity_deg = predict(features_scaled, beta_capacity)
            pred_cap = initial_capacity - predicted_capacity_deg


            # Predict voltage degradation
            beta_voltage = linear_regression(features_scaled, voltage_degradation)
            predicted_voltage_deg = predict(features_scaled, beta_voltage)
            pred_volt = initial_voltage - predicted_voltage_deg

            # Store results
            results[label] = {
                'predicted_rul': predicted_rul,
                'predicted_capacity_deg': predicted_capacity_deg,
                'predicted_voltage_deg': predicted_voltage_deg,
                'cycle_numbers': cycle_numbers,
                'remaining_life': remaining_life,
                'capacity_degradation': capacity_degradation,
                'voltage_degradation': voltage_degradation,
                'pred_volt' : pred_volt,
                'pred_cap' : pred_cap
            }

             # Calculate MAE and RMSE for capacity and voltage predictions
            mae_capacity = mean_absolute_error(capacity_degradation, predicted_capacity_deg)
            rmse_capacity = root_mean_squared_error(capacity_degradation, predicted_capacity_deg)

            mae_voltage = mean_absolute_error(voltage_degradation, predicted_voltage_deg)
            rmse_voltage = root_mean_squared_error(voltage_degradation, predicted_voltage_deg)

            # Print MAE and RMSE for each battery
            print(f"Results for {label}:")
            print(f"  Capacity Degradation - MAE: {mae_capacity:.4f}, RMSE: {rmse_capacity:.4f}")


            # Store capacity and voltage degradation for combined plot
            all_capacity_degradations.append(capacity_degradation)
            all_voltage_degradations.append(voltage_degradation)
            all_battery_labels.append(label)
            all_capacity_predicted.append(pred_cap)
            all_voltage_predicted.append(pred_volt)
            all_predicted_voltage_deg.append(predicted_voltage_deg)
            all_predicted_capacity_deg.append(predicted_capacity_deg)

            # Store actual and predicted voltage and capacity for time vs voltage/capacity plots
            all_actual_voltages.append(voltages)
            all_actual_capacities.append(capacities)

            # Plot the predictions
            plot_predictions(cycle_numbers, capacity_degradation, predicted_capacity_deg, label, 'Capacity Degradation')
            plot_predictions(cycle_numbers, voltage_degradation, predicted_voltage_deg, label, 'Voltage Degradation')

        except Exception as e:
            print(f"Error processing {label}: {e}")

    # Plot combined degradation (Actual vs Predicted) across all batteries
    plot_combined_degradation(all_capacity_degradations, all_predicted_capacity_deg, all_battery_labels, 'Capacity Degradation')
    plot_combined_degradation(all_voltage_degradations, all_predicted_voltage_deg, all_battery_labels, 'Voltage Degradation')

    # Plot combined Voltage vs Time (Actual vs Predicted) across all batteries
    plot_voltage_vs_time(all_actual_voltages, all_voltage_predicted, all_battery_labels)

    # Plot combined Capacity vs Time (Actual vs Predicted) across all batteries
    plot_capacity_vs_time(all_actual_capacities, all_capacity_predicted, all_battery_labels)

    return results

if __name__ == "__main__":
    main()
