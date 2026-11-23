import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

def log_test_accuracies_per_db(file, all_acc, database):
    lines = []

    # Block 1: individual test accuracies
    lines.append("=== Test Accuracies per Database ===")
    for db_id, acc_set in zip(database, all_acc):
        test_accs = acc_set[2]  # index 2 = test
        formatted = ", ".join(f"{acc:.4f}" for acc in test_accs)
        lines.append(f"DB {db_id}: {formatted}")

    # Block 2: average test accuracy per DB
    lines.append("\n=== Average Test Accuracy per Database ===")
    for db_id, acc_set in zip(database, all_acc):
        test_accs = acc_set[2]
        avg = sum(test_accs) / len(test_accs)
        lines.append(f"DB {db_id}: {avg:.4f}")

    # Save to file if needed
    if file:
        with open(file, 'w') as f:
            f.write("\n".join(lines))

    return lines 

def log_gestures_accuracy(file, all_acc, all_sets):
    gesture_totals = defaultdict(float)
    gesture_counts = defaultdict(int)
    gesture_individuals = defaultdict(list)

    for db_acc, db_sets in zip(all_acc, all_sets):
        for run_idx, gesture_split in enumerate(db_sets):
            for split_idx, gestures in enumerate(gesture_split):  # 0: train, 1: val, 2: test
                acc = db_acc[split_idx][run_idx]
                if math.isnan(acc):  # Skip NaNs
                    continue
                for gesture in gestures:
                    gesture_totals[gesture] += acc
                    gesture_counts[gesture] += 1
                    gesture_individuals[gesture].append(acc)

    # Compute average
    gesture_avg = {
        gesture: gesture_totals[gesture] / gesture_counts[gesture]
        for gesture in gesture_totals
    }

    # Format output
    lines = []

    # First block: average accuracy
    lines.append("=== Average Accuracy per Gesture ===")
    for gesture in sorted(gesture_avg):
        lines.append(f"Gesture {gesture:02d}: {gesture_avg[gesture]:.4f}")

    # Second block: list of individual accuracies
    lines.append("\n=== Individual Accuracies per Gesture ===")
    for gesture in sorted(gesture_individuals):
        accs = ", ".join(f"{acc:.4f}" for acc in gesture_individuals[gesture])
        lines.append(f"Gesture {gesture:02d}: {accs}")

    # Write to file
    if file:
        with open(file, 'w') as f:
            f.write("\n".join(lines))

    return gesture_avg, gesture_individuals

def log(database, file, gesture_sets, accuracies, train_times, predict_times, prates, total_time):

    train_acc, val_acc, test_acc = accuracies

    acc_avg, acc_std = calculate_stats(results=test_acc)
    prates_avg, prates_std = calculate_stats(results=prates)

    with open(file, 'w') as f:
        f.write(f"Gesture Sets - Tr Va Te: {gesture_sets}\n")
        f.write(f'Accuracies: {accuracies}\n')
        f.write(f'Acc avg:    {acc_avg}\n')
        f.write(f'Acc std:    {acc_std}\n')
        f.write(f'Train time: {train_times}\n')
        f.write(f'Pred time:  {predict_times}\n')
        f.write(f'Prate time: {prates}\n')
        f.write(f'Prate avg:  {prates_avg}\n')
        f.write(f'Prate std:  {prates_std}\n')
        f.write(f"Total time: {total_time}s\n\n")

    # Open this time as 'append' to avoid overwriting
    # Detailed table
    with open(file, 'a') as f:
        f.write('Detailed table:\n')
        f.write('\t\t\\midrule\n\t\t')

        for i in range(len(train_acc)):
            # if 1st part of loop
            if i == 0:
                f.write(f'{database} & ')
            else:
                f.write('{} & ')

            f.write(f'{i+1} & ')
            f.write(f'{gesture_sets[i][0]}{gesture_sets[i][1]}{gesture_sets[i][2]} & ')
            f.write(f'{train_acc[i]:.4f} & ')
            f.write(f'{val_acc[i]:.4f} & ')
            f.write(f'{test_acc[i]:.4f} & ')
            
            f.write(f'{predict_times[i]:.2f} & ')
            f.write(f'{prates[i]:.6f} & ')
            f.write(f'{train_times[i]:.1f} \\\\\n\t\t')
        
        f.write('\\bottomrule\n\n')

    # Summary table row
    with open(file, 'a') as f:
        f.write('Summary table:\n\t\t')
        f.write(f'{database} & ')
        f.write(f'{acc_avg:.4f} & ')
        f.write(f'{acc_std:.6f} & ')
        f.write(f'{prates_avg:.6f} & ')
        f.write(f'{prates_std:.6f} \\\\\n\n')
        

    # Per-Gesture Accuracy for Current DB 
    gesture_totals = defaultdict(float)
    gesture_counts = defaultdict(int)
    gesture_individuals = defaultdict(list)

    for i in range(len(gesture_sets)):
        for split_idx in range(3):  # 0: train, 1: val, 2: test
            acc = accuracies[split_idx][i]
            if math.isnan(acc):  # Skip NaNs
                continue
            for gesture in gesture_sets[i][split_idx]:
                gesture_totals[gesture] += acc
                gesture_counts[gesture] += 1
                gesture_individuals[gesture].append(acc)

    # Safe averaging
    gesture_avg = {
        gesture: gesture_totals[gesture] / gesture_counts[gesture]
        for gesture in gesture_totals
    }

    with open(file, 'a') as f:
        f.write('\n=== Gesture Accuracy Summary (Single DB) ===\n')
        f.write('Average Accuracy per Gesture:\n')
        for gesture in sorted(gesture_avg):
            f.write(f"Gesture {gesture:02d}: {gesture_avg[gesture]:.4f}\n")

        f.write('\nIndividual Accuracies per Gesture:\n')
        for gesture in sorted(gesture_individuals):
            accs = ", ".join(f"{acc:.4f}" for acc in gesture_individuals[gesture])
            f.write(f"Gesture {gesture:02d}: {accs}\n")


# def plot_bar_chart():
#     # Sample data
#     databases  = ['DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB7']
#     techniques = ['Atzori 2014', 'Random Forest', 'MLP', 'Atzori 2016', 'CNN']

#     # Average accuracy scores (rows: databases, columns: techniques)
#     means = np.array([
#         # Atzori 2014, RF,     MLP,    Atz 16, CNN
#         [0.7532,       0.5946, 0.6246, 0.6659, 0.5576],  # DB1
#         [0.7527,       0.6077, 0.5873, 0.6027, 0.5222],  # DB2
#         [0.4627,       0.4359, 0.3181, 0.3809, 0.3075],  # DB3
#         [0.0000,       0.5660, 0.6132, 0.0000, 0.5902],  # DB4
#         [0.0000,       0.6053, 0.6505, 0.0000, 0.6552],  # DB5
#         [0.0000,       0.6493, 0.7009, 0.0000, 0.6478]   # DB7
#     ])

#     # Standard deviations
#     stds = np.array([
#         [0.0000, 0.026518, 0.019179, 0.0000, 0.033522],  # DB1
#         [0.0000, 0.061135, 0.027552, 0.0000, 0.048779],  # DB2
#         [0.0000, 0.033971, 0.069168, 0.0000, 0.087616],  # DB3
#         [0.0000, 0.045234, 0.045025, 0.0000, 0.062800],  # DB4
#         [0.0000, 0.031930, 0.036007, 0.0000, 0.028054],  # DB5
#         [0.0000, 0.055550, 0.031098, 0.0000, 0.035104]   # DB7
#     ])

#     # Bar chart setup
#     x = np.arange(len(databases))  # the label locations
#     width = 0.25  # width of the bars

#     fig, ax = plt.subplots(figsize=(10, 6))

#     for i in range(len(techniques)):
#         ax.bar(x + i*width, means[:, i], width, yerr=stds[:, i], label=techniques[i], capsize=5)

#     # X-axis setup
#     ax.set_ylabel('Accuracy')
#     ax.set_xlabel('Database')
#     ax.set_title('Accuracy Comparison Across Databases and Techniques')
#     ax.set_xticks(x + width)
#     ax.set_xticklabels(databases)
#     ax.set_ylim(0, 1.0)
#     ax.legend()

#     plt.tight_layout()
#     plt.savefig(f'Code/all_results.png', dpi=300, bbox_inches='tight')

#     plt.show()


def plot_bar_chart():
    # Sample data
    databases  = ['DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB7']

    # Reference material acc
    # techniques = ['Atzori 2014', 'RF', 'Zhou 2019', 'MLP', 'Atzori 2016', 'CNN', 'RNN', 'Transformer']
    # # Average accuracy scores (rows: databases, columns: techniques)
    # means = np.array([
    #     [0.7532, 0.5946, 0.7520, 0.6246, 0.6659, 0.5576, 0.5070, 0.5634],
    #     [0.7527, 0.6077, 0.0000, 0.5873, 0.6027, 0.5222, 0.5161, 0.5266],
    #     [0.4627, 0.4359, 0.0000, 0.3181, 0.3809, 0.3075, 0.3389, 0.3762],
    #     [0.0000, 0.5660, 0.0000, 0.6132, 0.0000, 0.5902, 0.4871, 0.5616],
    #     [0.0000, 0.6053, 0.0000, 0.6505, 0.0000, 0.6552, 0.5102, 0.5930],
    #     [0.0000, 0.6493, 0.0000, 0.7009, 0.0000, 0.6478, 0.6085, 0.6395]
    # ])
    # # Standard deviations
    # stds = np.array([
    #     [0.0569, 0.026518, 0.0000, 0.019179, 0.0640, 0.033522, 0.028431, 0.026636],
    #     [0.0789, 0.061135, 0.0000, 0.027552, 0.0770, 0.048779, 0.015617, 0.050809],
    #     [0.0789, 0.033971, 0.0000, 0.069168, 0.1429, 0.087616, 0.017881, 0.020213],
    #     [0.0000, 0.045234, 0.0000, 0.045025, 0.0000, 0.062800, 0.034709, 0.045634],
    #     [0.0000, 0.031930, 0.0000, 0.036007, 0.0000, 0.028054, 0.023652, 0.031857],
    #     [0.0000, 0.055550, 0.0000, 0.031098, 0.0000, 0.035104, 0.018538, 0.028841]
    # ])

    # Just techniques acc
    techniques = ['RF', 'MLP', 'CNN', 'RNN', 'Transformer']
    means = np.array([
    [0.5946, 0.6246, 0.5576, 0.5070, 0.5634],
    [0.6077, 0.5873, 0.5222, 0.5161, 0.5266],
    [0.4359, 0.3181, 0.3075, 0.3389, 0.3762],
    [0.5660, 0.6132, 0.5902, 0.4871, 0.5616],
    [0.6053, 0.6505, 0.6552, 0.5102, 0.5930],
    [0.6493, 0.7009, 0.6478, 0.6085, 0.6395]
    ])
    stds = np.array([
    [0.026518, 0.019179, 0.033522, 0.028431, 0.026636],
    [0.061135, 0.027552, 0.048779, 0.015617, 0.050809],
    [0.033971, 0.069168, 0.087616, 0.017881, 0.020213],
    [0.045234, 0.045025, 0.062800, 0.034709, 0.045634],
    [0.031930, 0.036007, 0.028054, 0.023652, 0.031857],
    [0.055550, 0.031098, 0.035104, 0.018538, 0.028841]
    ])

    # # Just techniques times
    # techniques = ['RF', 'MLP', 'CNN', 'RNN', 'Transformer']
    # means = np.array([
    # [0.001641, 0.000039, 0.000143, 0.000160, 0.000169],
    # [0.001893, 0.000040, 0.000149, 0.000166, 0.000171],
    # [0.000174, 0.000098, 0.000311, 0.000316, 0.000334],
    # [0.000138, 0.000131, 0.000400, 0.000405, 0.000436],
    # [0.000148, 0.000140, 0.000458, 0.000461, 0.000453],
    # [0.000131, 0.000091, 0.000298, 0.000311, 0.000380]
    # ])
    # stds = np.array([
    # [0.000072, 0.000001, 0.000004, 0.000003, 0.000004],
    # [0.000262, 0.000003, 0.000012, 0.000008, 0.000012],
    # [0.000055, 0.000004, 0.000013, 0.000006, 0.000014],
    # [0.000001, 0.000004, 0.000012, 0.000014, 0.000016],
    # [0.000001, 0.000005, 0.000011, 0.000010, 0.000012],
    # [0.000012, 0.000002, 0.000004, 0.000008, 0.000142]
    # ])


    # Bar chart setup
    n_databases = len(databases)
    n_techniques = len(techniques)
    bar_width = 0.12  # Smaller bar width to allow spacing
    group_width = bar_width * n_techniques  # total width occupied by a group (without gap)
    gap = 0.20  # small gap between groups

    x = np.arange(n_databases) * (group_width + gap)  # spread out the groups with small gaps

    fig, ax = plt.subplots(figsize=(12, 7))

    for i in range(n_techniques):
        ax.bar(x + i * bar_width, means[:, i], bar_width, yerr=stds[:, i],
               label=techniques[i], capsize=5)

    # X-axis setup
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Database', fontsize=12)
    # ax.set_title('Accuracy Comparison Across Databases and Techniques', fontsize=14)
    ax.set_xticks(x + group_width / 2 - bar_width/2)  # center labels under the group
    ax.set_xticklabels(databases)
    ax.set_ylim(0, 1.0)
    ax.legend()

    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Code/all_results_implemented_predict_times.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_stats(results):
    if not results:
        return None, None  # Avoid division by zero or empty input
    average = statistics.mean(results)
    std_dev = statistics.stdev(results) if len(results) > 1 else 0.0
    return average, std_dev

def plot_bar_chart_gestures():
    # Sample data
    databases  = ['DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB7']

    # Reference material acc
    techniques = ['Atzori 2014', 'RF', 'Zhou 2019', 'MLP', 'Atzori 2016', 'CNN', 'RNN', 'Transformer']
    # Average accuracy scores (rows: databases, columns: techniques)
    means = np.array([
        [0.7532, 0.5946, 0.7520, 0.6246, 0.6659, 0.5576, 0.5070, 0.5634],
        [0.7527, 0.6077, 0.0000, 0.5873, 0.6027, 0.5222, 0.5161, 0.5266],
        [0.4627, 0.4359, 0.0000, 0.3181, 0.3809, 0.3075, 0.3389, 0.3762],
        [0.0000, 0.5660, 0.0000, 0.6132, 0.0000, 0.5902, 0.4871, 0.5616],
        [0.0000, 0.6053, 0.0000, 0.6505, 0.0000, 0.6552, 0.5102, 0.5930],
        [0.0000, 0.6493, 0.0000, 0.7009, 0.0000, 0.6478, 0.6085, 0.6395]
    ])
    # Standard deviations
    stds = np.array([
        [0.0569, 0.026518, 0.0000, 0.019179, 0.0640, 0.033522, 0.028431, 0.026636],
        [0.0789, 0.061135, 0.0000, 0.027552, 0.0770, 0.048779, 0.015617, 0.050809],
        [0.0789, 0.033971, 0.0000, 0.069168, 0.1429, 0.087616, 0.017881, 0.020213],
        [0.0000, 0.045234, 0.0000, 0.045025, 0.0000, 0.062800, 0.034709, 0.045634],
        [0.0000, 0.031930, 0.0000, 0.036007, 0.0000, 0.028054, 0.023652, 0.031857],
        [0.0000, 0.055550, 0.0000, 0.031098, 0.0000, 0.035104, 0.018538, 0.028841]
    ])

    # Bar chart setup
    n_databases = len(databases)
    n_techniques = len(techniques)
    bar_width = 0.12  # Smaller bar width to allow spacing
    group_width = bar_width * n_techniques  # total width occupied by a group (without gap)
    gap = 0.20  # small gap between groups

    x = np.arange(n_databases) * (group_width + gap)  # spread out the groups with small gaps

    fig, ax = plt.subplots(figsize=(12, 7))

    for i in range(n_techniques):
        ax.bar(x + i * bar_width, means[:, i], bar_width, yerr=stds[:, i],
               label=techniques[i], capsize=5)

    # X-axis setup
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Database', fontsize=12)
    # ax.set_title('Accuracy Comparison Across Databases and Techniques', fontsize=14)
    ax.set_xticks(x + group_width / 2 - bar_width/2)  # center labels under the group
    ax.set_xticklabels(databases)
    ax.set_ylim(0, 1.0)
    ax.legend()

    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Code/all_results_implemented_predict_times.png', dpi=300, bbox_inches='tight')
    plt.show()


# # Main
# results = [0.5575186412593206, 0.5999792186201164, 0.5384161237785017, 0.5907604251839739, 0.55519884009942]
# avg, std = calculate_stats(results)
# print(f"Average: {avg:.4f}, Standard Deviation: {std:.4f}")

# plot_bar_chart()
