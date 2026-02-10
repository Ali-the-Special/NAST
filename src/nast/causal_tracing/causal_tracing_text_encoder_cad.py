from utils.data_utils import load_dataset, generate_clip_input
from utils.model_utils import load_model
from utils.causal_tracing_utils import create_tracing_hook_fn, generate_tracing_chart_filepath, store_causal_tracing_results

import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import seaborn as sns
import torch
from tqdm import tqdm

# Experiment configuration
CONFIG = {
    'dataset_name': 'cad',      # 'valse' or 'cad'
    'encoder': 'text',          # vision/text
    'dataset': 'rephrased',     # only for valse: standard/rephrased (not used for cad)
    'negation': 'foil',      # foil/caption
    'segment': 'correct'        # correct/ambiguous/incorrect
}

present_idx = 0 if CONFIG['negation'] == 'foil' else 1
negation_idx = 1 if CONFIG['negation'] == 'foil' else 0

# Load dataset and model
dataset = load_dataset(
    dataset_name=CONFIG['dataset_name'],
    type=CONFIG['dataset'],  # This will be ignored for CAD
    negation=CONFIG['negation'],
    segment=CONFIG['segment']
)

pprint(CONFIG)
print(f'Input examples: {len(dataset)}')

if len(dataset) == 0:
    print("WARNING: No samples found with the specified filters!")
    print("Consider checking your segment and negation settings.")
    exit()

model, processor = load_model()
tracing_model, _ = load_model()

# Extract model parameters
layers = model.text_model.config.num_hidden_layers
position_agg_start = 5
position_agg_end = -2
positions_analysed = position_agg_start - position_agg_end + 1

scores_list = []
further_token_counts = []

for data in tqdm(dataset.values()):

    inputs = generate_clip_input(data, processor)
    positions = inputs.input_ids.shape[1]
    outputs = model(**inputs, output_hidden_states=True)

    logit_caption, logit_foil = outputs.logits_per_image.squeeze().tolist()
    score = logit_caption - logit_foil

    activations = outputs.text_model_output.hidden_states
    negation_activations = torch.stack([layer[negation_idx] for layer in activations])

    scores_tmp = np.zeros((positions, layers + 1), dtype=float)

    for layer in range(layers + 1):  # layer 0 = embedding output
        for position in range(positions):
            hook_fn = create_tracing_hook_fn(layer, present_idx, position, negation_activations)
            if layer == 0:  # Embedding output
                hook = tracing_model.text_model.embeddings.register_forward_hook(hook_fn)
            else:  # Transformer layer output
                hook = tracing_model.text_model.encoder.layers[layer - 1].register_forward_hook(hook_fn)

            outputs_tracing = tracing_model(**inputs)
            logit_traced = outputs_tracing.logits_per_image.squeeze().tolist()[present_idx]

            if CONFIG['negation'] == 'foil':
                score_traced = logit_caption - logit_traced
            if CONFIG['negation'] == 'caption':
                score_traced = logit_traced - logit_foil
            tracing_effect = score_traced / score

            scores_tmp[position, layer] = tracing_effect
            hook.remove()

    # Aggregate scores across post-subject token positions
    scores = np.zeros((positions_analysed, layers + 1), dtype=float)
    # Start to first subject token
    scores[:position_agg_start] = scores_tmp[:position_agg_start]
    # Further subject tokens (aggregated)
    further_tokens = scores_tmp[position_agg_start:position_agg_end].shape[0]
    if further_tokens > 0:
        scores[position_agg_start] = scores_tmp[position_agg_start:position_agg_end].mean(0)
    # Full stop and end
    scores[position_agg_end:] = scores_tmp[position_agg_end:]

    # Append results from this instance to lists
    further_token_counts.append(further_tokens)
    scores_list.append(scores)

# Stack together result lists
scores_final = np.stack(scores_list)
further_token_counts = np.array(further_token_counts)

# Calculate average over instances, weighting the aggregated tokens by their number
weights = np.ones_like(scores_final)
weights[:, position_agg_start, :] = np.tile(np.expand_dims(further_token_counts, 1), 13)
avg_scores = np.average(scores_final, axis=0, weights=weights)

# Save results
store_causal_tracing_results(results=avg_scores, config=CONFIG)

print("\nCausal tracing analysis complete!")
print(f"Results shape: {avg_scores.shape}")
print(f"Results saved for config: {CONFIG}")