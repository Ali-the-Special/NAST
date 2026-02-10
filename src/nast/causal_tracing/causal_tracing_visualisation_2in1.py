from utils.causal_tracing_utils import generate_tracing_chart_filepath, generate_tracing_result_filepath

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

negator_position = 3

CONFIG_FOIL = {
    'encoder': 'text',    # vision/text
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'foil',  # foil/caption
    'segment': 'correct'  # correct/ambiguous/incorrect
}

h5_filepath_foil = generate_tracing_result_filepath(CONFIG_FOIL)

with h5py.File(h5_filepath_foil, 'r') as hdf:
    # Load results
    results_foil = np.array(hdf['results'])

results_std_foil = (results_foil[negator_position:]).std(axis=0)
results_foil_annot = np.array([[f'{x:.2f}'.lstrip('0').replace('-0', '-') for x in row] for row in results_foil])


CONFIG_CAPTION = {
    'encoder': 'text',    # vision/text
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'caption',  # foil/caption
    'segment': 'correct'  # correct/ambiguous/incorrect
}

h5_filepath_caption = generate_tracing_result_filepath(CONFIG_CAPTION)

with h5py.File(h5_filepath_caption, 'r') as hdf:
    # Load results
    results_caption = np.array(hdf['results'])

results_std_caption = (results_caption[negator_position:]).std(axis=0)
results_caption_annot = np.array([[f'{x:.2f}'.lstrip('0').replace('-0', '-') for x in row] for row in results_caption])


generic_token_labels = ['[SOT]',
                        'there',
                        'is/are',
                        'a/some',
                        r'\textit{First subject token}',
                        r'\textit{Further subject tokens}',
                        '.',
                        '[EOT]']


plt.rcParams.update({
    "text.usetex": False,
    "font.size": 9
})

fig, axes = plt.subplots(2, 3, figsize=(6.4,3.5), gridspec_kw={'height_ratios': [3, 1],
                                            'width_ratios': [20,20,1]})

sns.set_theme(style="whitegrid")

hm_foil = sns.heatmap(results_foil, annot=results_foil_annot, fmt='', ax=axes[0, 0], cbar_ax=axes[0, 2], annot_kws={'size': 7})
hm_foil.set_title('Negation in foil', size=10)
# hm_foil.set(ylabel="Position")
hm_foil.set_xticks([])
hm_foil.set_yticklabels(labels=generic_token_labels, rotation=0)

bp_foil = sns.barplot(results_std_foil, ax=axes[1, 0], color='#02456b', linewidth=0)
bp_foil.set(xlabel='Layer', ylabel='CTE std dev')

hm_caption = sns.heatmap(results_caption, annot=results_caption_annot, fmt='', ax=axes[0, 1], cbar=False, annot_kws={'size': 7})
hm_caption.set_title('Negation in caption', size=10)
hm_caption.set_xticks([])
hm_caption.set_yticks([])

bp_caption = sns.barplot(results_std_caption, ax=axes[1, 1], color='#02456b', linewidth=0)
bp_caption.set(xlabel='Layer')
bp_caption.set_yticks([])

axes[1,2].axis('off')

plt.tight_layout(pad=0.5)
# plt.show()

plot_filename_eps = f'causal_tracing_text_{CONFIG_FOIL["dataset"]}_{CONFIG_FOIL["segment"]}_2in1.eps'
plot_filepath = f"../../output/charts/causal_tracing/{plot_filename_eps}"

plt.savefig(plot_filepath, format='eps')
print(f'Output saved at {plot_filepath}')
