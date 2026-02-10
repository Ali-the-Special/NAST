from utils.causal_tracing_utils import generate_tracing_chart_filepath, generate_tracing_result_filepath

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Experiment configuration
CONFIG = {
    'encoder': 'text',    # vision/text
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'foil',  # foil/caption
    'segment': 'correct'  # correct/ambiguous/incorrect
}

h5_filepath = generate_tracing_result_filepath(CONFIG)

with h5py.File(h5_filepath, 'r') as hdf:
    # Load results
    results = np.array(hdf['results'])

negator_position = 3
effect_std = (results[negator_position:]).std(axis=0)

OUTPUT_FORMAT = 'png' # [eps, png]

# Plotting
generic_token_labels = ['[SOT]',
                        'there',
                        'is/are',
                        'a/some',
                        'first subject token',
                        'further subject tokens',
                        '.',
                        '[EOT]']

non_agg_pos = [l for l in range(len(generic_token_labels)) if generic_token_labels[l] != 'further subject tokens']

if OUTPUT_FORMAT == 'png':

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1],
                                                                'width_ratios': [15, 1]})

    sns.set_theme(style="whitegrid")
    hm = sns.heatmap(results, annot=True, fmt=".2f", ax=axes[0, 0], cbar_ax=axes[0, 1])

    hm.set_title('Causal tracing effect per layer and input position', fontsize=16, fontweight='bold')

    hm.set(ylabel="Position")
    hm.set_xticks([])
    hm.set_yticklabels(labels=generic_token_labels, rotation=0)

    bp = sns.barplot(effect_std, ax=axes[1, 0], color='#02456b', linewidth=0)
    bp.set(xlabel='Position', ylabel='CTE std dev')

    # Add text annotations
    for idx, (key, value) in enumerate(CONFIG.items()):
        axes[1,1].text(0, 1 - idx * 0.1, f'{str(key).capitalize()}: {str(value).upper()}',
                transform=axes[1,1].transAxes, verticalalignment='top')
    axes[1, 1].axis('off')

    plt.tight_layout()

    plot_filepath = generate_tracing_chart_filepath(CONFIG)

    # plt.savefig(plot_filepath)

    print(results.sum(0))

    # plt.figure()
    # sns.barplot(results.sum(0))
    # plt.show()

# Save eps version for final report
if OUTPUT_FORMAT == 'eps':
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 11
    })

    fig, axes = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1],
                                                                'width_ratios': [15, 1]})

    sns.set_theme(style="whitegrid")
    hm = sns.heatmap(results, annot=True, fmt=".2f", ax=axes[0, 0], cbar_ax=axes[0, 1], annot_kws={'size': 9})

    hm.set_title('Causal tracing effect per layer and input position')

    hm.set(ylabel="Position")
    hm.set_xticks([])
    hm.set_yticklabels(labels=generic_token_labels, rotation=0)

    bp = sns.barplot(effect_std, ax=axes[1, 0], color='#02456b', linewidth=0)
    bp.set(xlabel='Layer', ylabel='CTE std dev')

    axes[1,1].axis('off')

    plt.tight_layout()

    plot_filepath_eps = generate_tracing_chart_filepath(CONFIG).replace('.png', '.eps')
    plt.savefig(plot_filepath_eps, format='eps')


    print(f'The average CTE std is {effect_std.mean():.3f}.')

