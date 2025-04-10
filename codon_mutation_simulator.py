#! /usr/bin/env python3

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import HTML, display

# -----------------------------
# Configuration Parameters
# -----------------------------
STRING_LENGTH = 50
LETTERS = 'ACDEFGHIMN'
LETTER_TO_INT = {letter: idx for idx, letter in enumerate(LETTERS)}
INT_TO_LETTER = {idx: letter for idx, letter in enumerate(LETTERS)}
POPULATION_SIZE = 1               # Number of strings in the population
NUM_MUTATIONS_PER_STRING = 1      # Each string generates 1 mutation
NUM_LETTERS_PER_MUTATION = 1      # Each mutation changes 1 letter
TOP_K = 1                          # Number of top mutations to select
SELECTION_THRESHOLD = 1           # Minimum score required for a mutation to be selected
MAX_BENEFIT_VALUE = 5.0            # Maximum score a mutation can have
MUTATIONS_PER_STEP = 1             # Number of mutations to insert per optimization step
TOTAL_STEPS = 1                    # Number of optimization steps for thorough testing
REWARD_VALUE = 1.0
PENALTY_VALUE = -0.5

# -----------------------------
# Validation: Ensure MUTATIONS_PER_STEP does not exceed POPULATION_SIZE
# -----------------------------
if MUTATIONS_PER_STEP > POPULATION_SIZE:
    raise ValueError("MUTATIONS_PER_STEP cannot exceed POPULATION_SIZE.")

# -----------------------------
# Define Reward Positions
# -----------------------------
REWARD_POSITIONS = {
    5: LETTER_TO_INT['A'],
    10: LETTER_TO_INT['F'],
    20: LETTER_TO_INT['C'],
    30: LETTER_TO_INT['D'],
    40: LETTER_TO_INT['E']
}

# -----------------------------
# Define Specific String
# -----------------------------
specific_string = 'FDCDHGEDHFEEEFDAGFEEDIIGDHDFAANINAGAFAEANNENMANDAA'  # 46 letters
# Pad the specific string to 50 letters
specific_string = specific_string.ljust(STRING_LENGTH, 'A')  # Pad with 'A's to make it 50 letters

# Verify specific string length
if len(specific_string) != STRING_LENGTH:
    raise ValueError(f"Specific string length ({len(specific_string)}) does not match STRING_LENGTH ({STRING_LENGTH})")

# -----------------------------
# Helper Functions
# -----------------------------

def create_scoring_matrix_with_neutral_string(specific_string, string_length=STRING_LENGTH):
    """
    Create a scoring matrix where the letters in the specific string are neutral,
    with rewards at specified positions and one additional neutral letter at each position.

    Args:
        specific_string (str): The specific string to include and set as neutral in the scoring matrix.
        string_length (int): The length of the string.

    Returns:
        scoring_matrix (np.ndarray): The scoring matrix where the specific string's letters are neutral,
                                     with rewards and an extra neutral letter.
    """
    # Initialize the scoring matrix with penalties for all letters
    scoring_matrix = np.full((string_length, len(LETTERS)), PENALTY_VALUE, dtype=np.float32)

    # Convert the specific string to integers using LETTER_TO_INT
    specific_string_int = [LETTER_TO_INT[char] for char in specific_string]

    # Set the letters in the specific string as neutral at their respective positions
    for pos, letter_int in enumerate(specific_string_int):
        scoring_matrix[pos, letter_int] = 0  # Neutral score

    # Add one more neutral letter at each position
    for pos in range(string_length):
        # Determine if position is a reward position
        if pos in REWARD_POSITIONS:
            reward_letter = REWARD_POSITIONS[pos]
            # Set reward letter
            scoring_matrix[pos, :] = PENALTY_VALUE  # Reset all to penalties at this position
            scoring_matrix[pos, reward_letter] = REWARD_VALUE  # Set reward letter

            # Determine neutral letters
            specific_letter = specific_string_int[pos]
            if specific_letter != reward_letter:
                # Specific string's letter is not the reward letter, set it as neutral
                scoring_matrix[pos, specific_letter] = 0  # Neutral

                # Choose one more neutral letter, excluding reward letter and specific string's letter
                possible_letters = [l for l in range(len(LETTERS)) if l != reward_letter and l != specific_letter]
                if possible_letters:
                    extra_neutral_letter = np.random.choice(possible_letters)
                    scoring_matrix[pos, extra_neutral_letter] = 0  # Set extra neutral
            else:
                # Specific string's letter is the reward letter, set one more neutral letter
                possible_letters = [l for l in range(len(LETTERS)) if l != reward_letter]
                if possible_letters:
                    extra_neutral_letter = np.random.choice(possible_letters)
                    scoring_matrix[pos, extra_neutral_letter] = 0  # Set extra neutral
        else:
            # Non-reward position
            # Specific string's letter is already set as neutral
            specific_letter = specific_string_int[pos]
            # Choose one more neutral letter, excluding specific string's letter
            possible_letters = [l for l in range(len(LETTERS)) if l != specific_letter]
            if possible_letters:
                extra_neutral_letter = np.random.choice(possible_letters)
                scoring_matrix[pos, extra_neutral_letter] = 0  # Set extra neutral

    return scoring_matrix

def initialize_population_with_specific_string(specific_string, pop_size=POPULATION_SIZE):
    """
    Initialize a population with the specific string and fill the rest with copies.

    Args:
        specific_string (str): The specific string to include as the first entry in the population.
        pop_size (int): The total number of strings in the population.

    Returns:
        population (np.ndarray): The population containing the specific string and its copies.
    """
    # Convert the specific string from letters to integers using LETTER_TO_INT
    specific_string_int = np.array([LETTER_TO_INT[char] for char in specific_string], dtype=np.int8)

    # Initialize the population with the specific string replicated 'pop_size' times
    population = np.tile(specific_string_int, (pop_size, 1))

    return population

def generate_mutations(population, num_mutations=NUM_MUTATIONS_PER_STRING, num_letters_per_mutation=NUM_LETTERS_PER_MUTATION):
    """
    Generate mutations for the population. Each string will generate 'num_mutations' mutations by changing 'num_letters_per_mutation' random letters.

    Args:
        population (np.ndarray): Current population of shape (pop_size, string_length)
        num_mutations (int): Number of mutations per string.
        num_letters_per_mutation (int): Number of letters to change per mutation.

    Returns:
        mutations (np.ndarray): Array of all mutations, shape (pop_size * num_mutations, string_length)
    """
    pop_size, string_length = population.shape
    total_mutations = pop_size * num_mutations

    # Repeat the population strings for mutations
    mutations = np.repeat(population, num_mutations, axis=0).copy()

    # For each mutation, choose 'num_letters_per_mutation' distinct random positions to mutate
    mutation_positions = []
    for _ in range(total_mutations):
        pos = np.random.choice(string_length, size=num_letters_per_mutation, replace=False)
        mutation_positions.append(pos)

    mutation_positions = np.array(mutation_positions)  # Shape: (total_mutations, num_letters_per_mutation)

    # Get current letters at mutation positions and apply mutations
    for i in range(total_mutations):
        positions = mutation_positions[i]

        for pos in positions:
            # Get current letter
            current_letter = mutations[i, pos]

            # Generate a new letter different from the current one
            new_letter = np.random.randint(0, len(LETTERS))
            while new_letter == current_letter:
                new_letter = np.random.randint(0, len(LETTERS))

            # Assign the new letter to the mutation position
            mutations[i, pos] = new_letter

    return mutations

def calculate_scores(mutations, scoring_matrix):
    """
    Calculate scores for all mutations based on the scoring matrix.

    Args:
        mutations (np.ndarray): Array of mutations, shape (num_mutations, STRING_LENGTH)
        scoring_matrix (np.ndarray): Scoring matrix, shape (STRING_LENGTH, len(LETTERS))

    Returns:
        scores (np.ndarray): Array of scores, shape (num_mutations,)
    """
    # Vectorized score calculation
    # For each mutation, sum the scores based on the scoring matrix
    scores = scoring_matrix[np.arange(mutations.shape[1]), mutations].sum(axis=1)

    # Apply the maximum benefit cap
    scores = np.minimum(scores, MAX_BENEFIT_VALUE)

    return scores

def select_top_mutations(mutations, scores, top_k=TOP_K, selection_threshold=SELECTION_THRESHOLD):
    """
    Select mutations based on top_k and/or selection_threshold.

    Args:
        mutations (np.ndarray): Array of mutations, shape (num_mutations, STRING_LENGTH)
        scores (np.ndarray): Array of scores, shape (num_mutations,)
        top_k (int): Number of top mutations to select.
        selection_threshold (float): Minimum score required for a mutation to be selected.

    Returns:
        selected_mutations (np.ndarray): Array of selected mutations.
        selected_scores (np.ndarray): Array of selected scores.
    """
    # First, select mutations that meet or exceed the selection_threshold
    threshold_indices = np.where(scores >= selection_threshold)[0]
    threshold_mutations = mutations[threshold_indices]
    threshold_scores = scores[threshold_indices]

    if len(threshold_scores) >= top_k:
        # If enough mutations meet the threshold, select the top_k among them
        sorted_indices = np.argsort(-threshold_scores)
        selected_mutations = threshold_mutations[sorted_indices[:top_k]]
        selected_scores = threshold_scores[sorted_indices[:top_k]]
    else:
        # If not enough, select all that meet the threshold and fill the rest with top mutations below the threshold
        selected_mutations = threshold_mutations
        selected_scores = threshold_scores

        remaining_k = top_k - len(selected_scores)
        if remaining_k > 0:
            # Select additional top mutations below the threshold
            below_threshold_indices = np.argsort(-scores)[:remaining_k]
            selected_mutations = np.vstack((selected_mutations, mutations[below_threshold_indices]))
            selected_scores = np.concatenate((selected_scores, scores[below_threshold_indices]))

    return selected_mutations, selected_scores

def simulated_annealing_acceptance(current_best_score, top_scores, temperature):
    """
    Decide whether to accept each of the top mutations based on Simulated Annealing.

    Args:
        current_best_score (float): The best score in the current population.
        top_scores (np.ndarray): Array of top scores, shape (top_k,)
        temperature (float): Current temperature in the cooling schedule.

    Returns:
        accept_mask (np.ndarray): Boolean array indicating acceptance, shape (top_k,)
    """
    delta = top_scores - current_best_score
    accept = delta > 0
    negative_delta = delta <= 0

    if temperature > 0 and np.any(negative_delta):
        acceptance_probs = np.exp(delta[negative_delta] / temperature)
        random_probs = np.random.rand(np.sum(negative_delta))
        accept[negative_delta] = random_probs < acceptance_probs

    return accept

def update_population(current_population, top_mutations, top_scores, temperature, scoring_matrix):
    """
    Update the population based on the selected top mutations and Simulated Annealing.
    Inserts a specified number of mutations per step.

    Args:
        current_population (np.ndarray): Current population, shape (pop_size, STRING_LENGTH)
        top_mutations (np.ndarray): Top mutations, shape (top_k, STRING_LENGTH)
        top_scores (np.ndarray): Top scores, shape (top_k,)
        temperature (float): Current temperature in the cooling schedule.
        scoring_matrix (np.ndarray): Scoring matrix, shape (STRING_LENGTH, len(LETTERS))

    Returns:
        new_population (np.ndarray): Updated population, shape (pop_size, STRING_LENGTH)
    """
    # Calculate current population scores
    population_scores = calculate_scores(current_population, scoring_matrix)
    current_best_score = np.max(population_scores)

    # Decide which mutations to accept
    accept_mask = simulated_annealing_acceptance(current_best_score, top_scores, temperature)
    accepted_mutations = top_mutations[accept_mask]
    num_accept = len(accepted_mutations)

    # Limit the number of mutations to insert per step
    if num_accept > MUTATIONS_PER_STEP:
        # Randomly select MUTATIONS_PER_STEP mutations from the accepted ones
        selected_indices = np.random.choice(num_accept, MUTATIONS_PER_STEP, replace=False)
        accepted_mutations = accepted_mutations[selected_indices]
        num_accept = MUTATIONS_PER_STEP

    # Initialize new population as a copy of the current population
    new_population = current_population.copy()

    if num_accept > 0:
        # Identify indices of the worst-performing members in the population
        worst_indices = np.argsort(population_scores)[:num_accept]

        # Replace the worst-performing members with the accepted mutations
        new_population[worst_indices] = accepted_mutations

    return new_population

def shuffle_mutations_and_scores(mutations, scores):
    """
    Shuffle mutations and scores in unison.

    Args:
        mutations (np.ndarray): Array of mutations, shape (num_mutations, STRING_LENGTH)
        scores (np.ndarray): Array of scores, shape (num_mutations,)

    Returns:
        shuffled_mutations (np.ndarray): Shuffled array of mutations.
        shuffled_scores (np.ndarray): Shuffled array of scores.
    """
    permutation = np.random.permutation(len(scores))
    shuffled_mutations = mutations[permutation]
    shuffled_scores = scores[permutation]
    return shuffled_mutations, shuffled_scores

def categorize_letters(string, pos, scoring_matrix, specific_string_int, REWARD_POSITIONS):
    """
    Categorize each letter in the string based on its position.

    Args:
        string (np.ndarray): Array of letters in integer form.
        pos (int): Position of the letter.
        scoring_matrix (np.ndarray): Scoring matrix.
        specific_string_int (list): Integer representation of the specific string.
        REWARD_POSITIONS (dict): Dictionary of reward positions.

    Returns:
        category (str): Category of the letter ('reward', 'starting', 'neutral', 'penalty')
    """
    letter_int = string[pos]

    if pos in REWARD_POSITIONS and letter_int == REWARD_POSITIONS[pos]:
        return 'reward'
    elif letter_int == specific_string_int[pos]:
        return 'starting'
    elif scoring_matrix[pos, letter_int] == 0:
        return 'neutral'
    else:
        return 'penalty'

def generate_colored_html(string, scoring_matrix, specific_string_int, REWARD_POSITIONS):
    """
    Generate an HTML string with colored letters based on their categories.

    Args:
        string (np.ndarray): Array of letters in integer form.
        scoring_matrix (np.ndarray): Scoring matrix.
        specific_string_int (list): Integer representation of the specific string.
        REWARD_POSITIONS (dict): Dictionary of reward positions.

    Returns:
        colored_html (str): HTML string with colored letters.
        counts (dict): Dictionary with counts of each category.
    """
    categories = []
    colored_letters = []
    counts = {
        'starting': 0,
        'neutral': 0,
        'reward': 0,
        'penalty': 0
    }

    for pos in range(len(string)):
        letter_int = string[pos]
        category = categorize_letters(string, pos, scoring_matrix, specific_string_int, REWARD_POSITIONS)
        categories.append(category)
        if category == 'reward':
            color = 'green'
            counts['reward'] += 1
        elif category == 'starting':
            color = 'blue'
            counts['starting'] += 1
        elif category == 'neutral':
            color = 'black'
            counts['neutral'] += 1
        else:
            color = 'red'
            counts['penalty'] += 1
        colored_letters.append(f'<span style="color:{color}">{INT_TO_LETTER[letter_int]}</span>')

    colored_html = ''.join(colored_letters)
    return colored_html, counts

def optimize_string_with_sa(specific_string, scoring_matrix, total_steps=TOTAL_STEPS):
    """
    Optimize the string using a population-based approach with Simulated Annealing.
    Inserts a specified number of mutations per step.

    Args:
        specific_string (str): The specific string to start with.
        scoring_matrix (np.ndarray): Scoring matrix, shape (STRING_LENGTH, len(LETTERS))
        total_steps (int): Number of optimization steps.

    Returns:
        final_population_letters (list of str): Final optimized strings with colored HTML.
        final_scores (np.ndarray): Scores of the final population.
        final_counts (list of dict): List of dictionaries with counts for each string.
    """
    # Initialize population with copies of the specific string
    population = initialize_population_with_specific_string(specific_string, POPULATION_SIZE)

    # Convert specific string to integer list for categorization
    specific_string_int = [LETTER_TO_INT[char] for char in specific_string]

    # Temperature schedule for Simulated Annealing
    initial_temperature = 1000
    final_temperature = 100
    temperatures = np.linspace(initial_temperature, final_temperature, total_steps)

    best_scores = []

    # Optimization loop with progress bar
    for step in tqdm(range(total_steps), desc="Optimization Steps"):
        temperature = temperatures[step]

        # Generate mutations
        mutations = generate_mutations(population, NUM_MUTATIONS_PER_STRING, NUM_LETTERS_PER_MUTATION)

        # Calculate scores
        scores = calculate_scores(mutations, scoring_matrix)

        # Shuffle mutations and scores
        mutations, scores = shuffle_mutations_and_scores(mutations, scores)

        # Select top mutations based on TOP_K and SELECTION_THRESHOLD
        top_mutations, top_scores = select_top_mutations(
            mutations, scores, top_k=TOP_K, selection_threshold=SELECTION_THRESHOLD
        )

        # Update population with Simulated Annealing, inserting MUTATIONS_PER_STEP mutations
        population = update_population(population, top_mutations, top_scores, temperature, scoring_matrix)

        # Log the best score
        best_score = np.max(calculate_scores(population, scoring_matrix))
        best_scores.append(best_score)

    # Plotting the optimization progress
    plt.figure(figsize=(12, 6))
    plt.plot(best_scores, label='Best Score', marker='o')
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.title('Optimization Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prepare final population for display
    final_population_letters = []
    final_counts = []
    for string in population:
        colored_html, counts = generate_colored_html(string, scoring_matrix, specific_string_int, REWARD_POSITIONS)
        final_population_letters.append(colored_html)
        final_counts.append(counts)

    final_scores = calculate_scores(population, scoring_matrix)

    return final_population_letters, final_scores, final_counts

def print_all_starting_strings(starting_population):
    """
    Print all starting strings by converting them from integer representations to letters.

    Args:
        starting_population (np.ndarray): The starting population where each string is an array of integers.
    """
    print("All Starting Strings:")
    for idx, string in enumerate(starting_population, 1):
        string_letters = ''.join([INT_TO_LETTER[int(char)] for char in string])
        print(f"{idx}. {string_letters}")
    print("\n")

# -----------------------------
# Execution
# -----------------------------

# Create a scoring matrix where the letters in the specific string are neutral,
# with rewards and an extra neutral letter
scoring_matrix = create_scoring_matrix_with_neutral_string(specific_string)

# Initialize population with copies of the specific string
starting_population = initialize_population_with_specific_string(specific_string, POPULATION_SIZE)

# Print all starting strings
print_all_starting_strings(starting_population)

# Run the optimization
final_population, final_scores, final_counts = optimize_string_with_sa(specific_string, scoring_matrix)

# Generate HTML table for final strings with colored letters and counts
html_table = """
<table border="1" cellpadding="5" cellspacing="0">
    <tr>
        <th>String</th>
        <th>Starting Letters (Blue)</th>
        <th>Reward Letters (Green)</th>
        <th>Neutral Letters (Black)</th>
        <th>Penalty Letters (Red)</th>
    </tr>
"""

for string_html, score, counts in zip(final_population, final_scores, final_counts):
    html_table += f"""
    <tr>
        <td>{string_html}</td>
        <td>{counts['starting']}</td>
        <td>{counts['reward']}</td>
        <td>{counts['neutral']}</td>
        <td>{counts['penalty']}</td>
    </tr>
    """

html_table += "</table>"

# Display the HTML table
display(HTML(html_table))

# Additionally, print the final scores
print("\nFinal Scores:")
for idx, score in enumerate(final_scores, 1):
    print(f"String {idx}: Score = {score}")
