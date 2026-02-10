import random
import pandas as pd
import json
import os
from openai import OpenAI
from tqdm import tqdm

# -----------------------------------------------------
# Initialize OpenAI client
# -----------------------------------------------------
client = OpenAI()

# -----------------------------------------------------
# Affirmative mapping
# -----------------------------------------------------
affirmative_mapping = {
    "Atelectasis=0": "well-aerated expanded lungs",
    "Atelectasis=1": "atelectasis",
    "Cardiomegaly=0": "normal heart size",
    "Cardiomegaly=1": "cardiomegaly",
    "Consolidation=0": "clear lung parenchyma",
    "Consolidation=1": "consolidation",
    "Edema=0": "normal pulmonary vascularity",
    "Edema=1": "pulmonary edema",
    "Enlarged Cardiomediastinum=0": "normal mediastinal contours",
    "Enlarged Cardiomediastinum=1": "enlarged cardiomediastinum",
    "Fracture=0": "intact bony structures",
    "Fracture=1": "fracture",
    "Lung Lesion=0": "homogeneous lung parenchyma",
    "Lung Lesion=1": "lung lesion",
    "Lung Opacity=0": "well-aerated lung fields",
    "Lung Opacity=1": "lung opacity",
    "Pleural Effusion=0": "sharp costophrenic angles",
    "Pleural Effusion=1": "pleural effusion",
    "Pleural Other=0": "smooth pleural surfaces",
    "Pleural Other=1": "pleural abnormality",
    "Pneumonia=0": "aerated alveoli",
    "Pneumonia=1": "pneumonia",
    "Pneumothorax=0": "fully expanded lungs",
    "Pneumothorax=1": "pneumothorax",
    "Support Devices=0": "device-free chest",
    "Support Devices=1": "support devices"
}


# -----------------------------------------------------
# Load and filter dataset
# -----------------------------------------------------
def load_and_filter_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # List of condition columns (excluding 'No Finding')
    condition_cols = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices"
    ]

    # Filter rows with at least 3 zeros AND at least 2 ones
    filtered_rows = []
    for idx, row in df.iterrows():
        values = [row[col] for col in condition_cols if pd.notna(row[col])]
        num_zeros = values.count(0.0)
        num_ones = values.count(1.0)

        if num_zeros >= 3 and num_ones >= 2:
            filtered_rows.append(row)

    return pd.DataFrame(filtered_rows), condition_cols


# -----------------------------------------------------
# Select 3 random conditions from available ones
# -----------------------------------------------------
def select_conditions_from_row(row, condition_cols, num_conditions=3):
    # Get conditions that have valid values (0 or 1, not NaN)
    available_conditions = [col for col in condition_cols
                            if pd.notna(row[col]) and row[col] in [0.0, 1.0]]

    if len(available_conditions) < num_conditions:
        return None

    return random.sample(available_conditions, num_conditions)


# -----------------------------------------------------
# Generate raw_choices for selected conditions
# -----------------------------------------------------
def generate_raw_choices(row, conditions):
    # Get correct values from the row
    correct_values = {c: int(row[c]) for c in conditions}
    correct_label = random.choice(["A", "B", "C", "D"])

    raw_choices = {correct_label: correct_values}
    existing_variants = {tuple(sorted(correct_values.items()))}

    # Generate 3 more unique variants
    for label in [x for x in ["A", "B", "C", "D"] if x != correct_label]:
        while True:
            variant = correct_values.copy()
            to_flip = random.sample(conditions, random.randint(1, len(conditions)))
            for c in to_flip:
                variant[c] = 1 - variant[c]
            variant_tuple = tuple(sorted(variant.items()))
            if variant_tuple not in existing_variants:
                existing_variants.add(variant_tuple)
                raw_choices[label] = variant
                break

    return raw_choices, correct_label


# -----------------------------------------------------
# Generate negation question
# -----------------------------------------------------
def generate_negation_question(conditions, raw_choices):
    relevant_map = {}
    for cond in conditions:
        affirmative_key = f"{cond}=1"
        relevant_map[cond] = affirmative_mapping.get(affirmative_key, cond)

    prompt = f"""
You are a radiologist creating a multiple-choice question based on chest radiograph findings
Task:
- Write a clear and neutral question stem that asks which option correctly describes the image findings. Keep the question generic, without revealing information about what is or isn't present.
- Rephrase each option into natural medical language and radiology report summary.
- For conditions marked =1, use varied ways to express presence, e.g. "with Cardiomegaly", "showing Cardiomegaly", "evidence of Cardiomegaly", "Cardiomegaly noted".
- For conditions marked =0, use varied ways to express absence, e.g. "without Edema", "no Edema", "Edema absent", "free of Edema", "lacking Edema".
- Use natural conjunctions and negations, with patterns like: "A and B without C", "A, B, but not C", "C absent while A and B present", "No C, with A and B", "A noted, no sign of B or C".
- State ONLY the presence or absence of the specified conditions, and avoid explanatory phrases or redundant descriptions.
- Output only the question and its options (A–D), with no explanations or extra text.
Condition names to use (USE THESE EXACT TERMS ONLY):
{relevant_map}
Ground truth values:
{raw_choices}
"""

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=250,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()


# -----------------------------------------------------
# Generate affirmative question
# -----------------------------------------------------
def generate_affirmative_options(negation_options, raw_choices, conditions):
    rewrite_map = {}
    for cond in conditions:
        rewrite_map[f"{cond}=1"] = cond
        rewrite_map[f"{cond}=0"] = affirmative_mapping.get(f"{cond}=0", cond)

    # Format the negation options for the prompt
    options_text = f"""A. {negation_options['A']}
        B. {negation_options['B']}
        C. {negation_options['C']}
        D. {negation_options['D']}"""

    prompt = f"""
You are rewriting radiology multiple-choice options to replace ONLY the negation phrases with affirmative alternatives.

CRITICAL INSTRUCTIONS:
1. Keep the overall sentence structure, word order, and style as close as possible to the original
2. ONLY change phrases that negate a condition (e.g., "no X", "without X", "X absent", "free of X", "lacking X", "no evidence of X", "no radiographic evidence of X")
3. Replace negation phrases with the affirmative alternative provided below
4. Make MINIMAL grammatical adjustments ONLY when necessary for the sentence to remain natural and grammatically correct
5. For conditions that are present (=1), keep the EXACT original wording unchanged
6. Ensure the result is grammatically correct and sounds NATURAL - add or remove or change necessary prepositions (e.g., "and", "with", "showing") or conjunctions when needed
7. If a negation phrase includes words like "while", "and", "with" followed by the negation, adjust minimally to maintain natural grammar
8. If replacing a negated abnormality (=0) with a normal/benign affirmative finding (e.g., "aerated alveoli", "fully expanded lungs", "normal heart size"), REMOVE or REPLACE adversative conjunctions like "but", "however", or contrastive commas with neutral conjunctions (e.g., ", with", "and") when needed for natural radiology phrasing.
9. If replacing a negation results in repeated "with" constructions (e.g., ", with X, with Y"), replace the later "with" with "and" when it improves fluency.

Output format:
A. [rewritten option A]
B. [rewritten option B]
C. [rewritten option C]
D. [rewritten option D]

Do NOT include question stem, explanations, or extra text.

Affirmative replacements for absent conditions (=0):
{rewrite_map}

Options to rewrite:
{options_text}

Ground truth (1=present, 0=absent):
{raw_choices}

Remember: Stay as close to the original as possible, but ensure the result is grammatically correct and sounds natural.
"""

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


# -----------------------------------------------------
# Parse MCQ options from generated text
# -----------------------------------------------------
def parse_mcq_options(text):
    lines = text.split('\n')
    options = {"A": "", "B": "", "C": "", "D": ""}
    question = ""

    for line in lines:
        line = line.strip()
        if line.startswith("A)") or line.startswith("A."):
            options["A"] = line[2:].strip()
        elif line.startswith("B)") or line.startswith("B."):
            options["B"] = line[2:].strip()
        elif line.startswith("C)") or line.startswith("C."):
            options["C"] = line[2:].strip()
        elif line.startswith("D)") or line.startswith("D."):
            options["D"] = line[2:].strip()
        elif not any(line.startswith(x) for x in ["A)", "B)", "C)", "D)", "A.", "B.", "C.", "D."]) and line:
            question += line + " "

    return question.strip(), options


# -----------------------------------------------------
# Parse only options from generated text
# -----------------------------------------------------
def parse_options_only(text):
    lines = text.split('\n')
    options = {"A": "", "B": "", "C": "", "D": ""}

    for line in lines:
        line = line.strip()
        if line.startswith("A)") or line.startswith("A."):
            options["A"] = line[2:].strip()
        elif line.startswith("B)") or line.startswith("B."):
            options["B"] = line[2:].strip()
        elif line.startswith("C)") or line.startswith("C."):
            options["C"] = line[2:].strip()
        elif line.startswith("D)") or line.startswith("D."):
            options["D"] = line[2:].strip()

    return options


# -----------------------------------------------------
# Load existing results for resume
# -----------------------------------------------------
def load_existing_results(output_json_path):
    """Load existing results if the output file exists."""
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f:
                results = json.load(f)
            print(f"Resuming: Found {len(results)} existing samples")
            return results
        except Exception as e:
            print(f"Could not load existing results: {e}")
            return []
    return []


# -----------------------------------------------------
# Get processed study IDs
# -----------------------------------------------------
def get_processed_study_ids(results):
    """Extract set of already processed study_ids."""
    return set((r['subject_id'], r['study_id']) for r in results)


# -----------------------------------------------------
# Save results incrementally
# -----------------------------------------------------
def save_results(results, output_json_path):
    """Save results to JSON file."""
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)


# -----------------------------------------------------
# Main processing
# -----------------------------------------------------
def process_dataset(csv_path, output_json_path, num_samples='all', save_frequency=10):
    """
    Process MIMIC-CXR dataset to generate benchmark with resume capability.

    Args:
        csv_path: Path to the MIMIC-CXR CSV file
        output_json_path: Path to save the output JSON
        num_samples: Number of samples to process
                    - 'all': process all filtered samples
                    - int: process exactly that many samples (stops when target is met)
        save_frequency: Save results every N samples (default: 10)
    """
    # Load existing results if resuming
    results = load_existing_results(output_json_path)
    processed_ids = get_processed_study_ids(results)
    processed_count = len(results)

    # Load and filter data
    df, condition_cols = load_and_filter_dataset(csv_path)
    print(f"Filtered dataset: {len(df)} samples with at least 3 zeros and 2 ones")

    # Determine target number
    if num_samples == 'all':
        target_samples = len(df)
        print(f"Processing all {target_samples} samples")
    else:
        target_samples = min(num_samples, len(df))
        print(f"Processing {target_samples} samples")

    if processed_count >= target_samples:
        print(f"Already processed {processed_count} samples. Target reached!")
        return

    print(f"Starting from sample {processed_count + 1}")

    # Process each sample with progress bar
    progress_bar = tqdm(total=target_samples, initial=processed_count, desc="Processing samples")

    for idx, row in df.iterrows():
        # Stop when we've successfully processed enough samples
        if processed_count >= target_samples:
            break

        # Skip already processed samples
        sample_id = (int(row["subject_id"]), int(row["study_id"]))
        if sample_id in processed_ids:
            continue
        try:
            # Select 3 random conditions
            conditions = select_conditions_from_row(row, condition_cols, num_conditions=3)
            if conditions is None:
                continue

            # Generate raw choices
            raw_choices, correct_label = generate_raw_choices(row, conditions)

            # Generate negation question
            negation_text = generate_negation_question(conditions, raw_choices)
            question, negation_options = parse_mcq_options(negation_text)

            # Generate affirmative options only (using negation options)
            affirmative_text = generate_affirmative_options(negation_options, raw_choices, conditions)
            affirmative_options = parse_options_only(affirmative_text)

            # Build result entry
            result = {
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
                "question": question,
                "binary_raw_choice_1": raw_choices["A"],
                "binary_raw_choice_2": raw_choices["B"],
                "binary_raw_choice_3": raw_choices["C"],
                "binary_raw_choice_4": raw_choices["D"],
                "negation_mcq_choice_1": negation_options["A"],
                "negation_mcq_choice_2": negation_options["B"],
                "negation_mcq_choice_3": negation_options["C"],
                "negation_mcq_choice_4": negation_options["D"],
                "affirmative_mcq_choice_1": affirmative_options["A"],
                "affirmative_mcq_choice_2": affirmative_options["B"],
                "affirmative_mcq_choice_3": affirmative_options["C"],
                "affirmative_mcq_choice_4": affirmative_options["D"],
                "correct_answer": correct_label
            }

            results.append(result)
            processed_count += 1
            progress_bar.update(1)

            # Save periodically
            if processed_count % save_frequency == 0:
                save_results(results, output_json_path)
                print(f"\nCheckpoint saved: {processed_count} samples")

        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            continue

    progress_bar.close()

    # Final save
    save_results(results, output_json_path)

    print(f"\nProcessed {len(results)} samples successfully")
    print(f"Results saved to {output_json_path}")


# -----------------------------------------------------
# Run the benchmark generation
# -----------------------------------------------------
if __name__ == "__main__":
    csv_path = "mimic-cxr-2.0.0-chexpert.csv"
    output_path = "mednega_cxr_benchmark.json"

    # Examples:
    # Process all samples (saves every 10 samples by default)
    process_dataset(csv_path, output_path, num_samples="all")

    # Process exactly 100 samples with custom save frequency
    # process_dataset(csv_path, output_path, num_samples=100, save_frequency=5)

    # Process exactly 500 samples
    # process_dataset(csv_path, output_path, num_samples=500, save_frequency=20)