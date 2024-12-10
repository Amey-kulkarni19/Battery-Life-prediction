import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(cycle_numbers, true_values, predicted_values, label, y_label):
    plt.figure(figsize=(10, 6))
    plt.plot(cycle_numbers, true_values, label='True Values', color='b', marker='o')
    plt.plot(cycle_numbers, predicted_values, label='Predicted Values', color='r', linestyle='--')
    plt.xlabel('Cycle Number')
    plt.ylabel(y_label)
    plt.title(f'{label} - {y_label}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_combined_degradation(all_actual_degradations, all_predicted_degradations, all_battery_labels, y_label):
    plt.figure(figsize=(10, 6))
    
    for actual, predicted, label in zip(all_actual_degradations, all_predicted_degradations, all_battery_labels):
        # Plot actual degradation
        plt.plot(np.arange(len(actual)), actual, label=f'Actual - {label}', linestyle='-', marker='o')
        # Plot predicted degradation
        plt.plot(np.arange(len(predicted)), predicted, label=f'Predicted - {label}', linestyle='--')
    
    plt.xlabel('Cycle Number')
    plt.ylabel(y_label)
    plt.title(f'Combined {y_label} (Actual vs Predicted) across all batteries')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_voltage_vs_time(all_actual_voltages, all_predicted_voltages, all_battery_labels):
    plt.figure(figsize=(10, 6))
    
    for actual, predicted, label in zip(all_actual_voltages, all_predicted_voltages, all_battery_labels):
        # Plot actual voltage vs time
        plt.plot(np.arange(len(actual)), actual, label=f'Actual Voltage - {label}', linestyle='-', marker='o')
        # Plot predicted voltage vs time
        plt.plot(np.arange(len(predicted)), predicted, label=f'Predicted Voltage - {label}', linestyle='--')
    
    plt.xlabel('Time (Cycles)')
    plt.ylabel('Voltage (V)')
    plt.title('Combined Voltage vs Time (Actual vs Predicted) across all batteries')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_capacity_vs_time(all_actual_capacities, all_predicted_capacities, all_battery_labels):
    plt.figure(figsize=(10, 6))
    
    for actual, predicted, label in zip(all_actual_capacities, all_predicted_capacities, all_battery_labels):
        # Plot actual capacity vs time
        plt.plot(np.arange(len(actual)), actual, label=f'Actual Capacity - {label}', linestyle='-', marker='o')
        # Plot predicted capacity vs time
        plt.plot(np.arange(len(predicted)), predicted, label=f'Predicted Capacity - {label}', linestyle='--')
    
    plt.xlabel('Time (Cycles)')
    plt.ylabel('Capacity (Ah)')
    plt.title('Combined Capacity vs Time (Actual vs Predicted) across all batteries')
    plt.legend()
    plt.grid(True)
    plt.show()