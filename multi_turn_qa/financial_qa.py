import asyncio
import numpy as np
from tqdm import tqdm
import traceback
from collections import defaultdict
from functools import wraps
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from datasets import load_dataset
import re
import math
import numpy as np
from pyfpe_ff3 import FF3Cipher, format_align_digits
import torch
from privacypromptrewriting.utils import *
import matplotlib.pyplot as plt
import os


def process_financial_data_with_noise(text, epsilon=1.0):
    output_dict = {
        'dollar_amounts': [],
        'table_values': []
    }
    
    # First extract tables and their contents
    table_pattern = r'<table.*?</table>'
    tables = re.findall(table_pattern, text, re.DOTALL)
    text_without_tables = re.sub(table_pattern, '', text, flags=re.DOTALL)
    
    # Process dollar amounts in main text (outside tables)
    dollar_pattern = r'\$\s+(\d+(?:,\d{3})*(?:\.\d+)?)'
    dollar_matches = re.finditer(dollar_pattern, text_without_tables)
    for match in dollar_matches:
        amount = float(match.group(1).replace(',', ''))
        n_lower, n_upper = determine_range(amount)
        noised_amount = M_epsilon(amount, n_lower, n_upper, epsilon, discretization_size=1000, int_out=False)
        output_dict['dollar_amounts'].append(noised_amount)
    
    # Process tables
    for table in tables:
        rows = re.findall(r'<tr>(.*?)</tr>', table, re.DOTALL)
        
        for row_idx, row in enumerate(rows):
            if row_idx == 0:  # Skip header row
                continue
                
            cells = re.findall(r'<td>(.*?)</td>', row)
            
            for cell_idx, cell in enumerate(cells):
                if cell_idx <= 1:  # Skip first two columns
                    continue
                
                # Updated pattern to handle numbers with symbols like %
                paired_pattern = r'(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*%?\s*\(\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*%?\s*\)'
                paired_match = re.search(paired_pattern, cell)
                
                if paired_match:
                    value = float(paired_match.group(1).replace(',', ''))
                    n_lower, n_upper = determine_range(value)
                    noised_value = M_epsilon(value, n_lower, n_upper, epsilon, discretization_size=1000, int_out=False)
                    # Add both the noised value and its absolute value
                    output_dict['table_values'].append(noised_value)
                    output_dict['table_values'].append(abs(noised_value))
                else:
                    # Look for single numbers
                    number_pattern = r'-?(\d+(?:,\d{3})*(?:\.\d+)?)'
                    number_matches = re.finditer(number_pattern, cell)
                    for match in number_matches:
                        value = float(match.group(1).replace(',', ''))
                        n_lower, n_upper = determine_range(value)
                        noised_value = M_epsilon(value, n_lower, n_upper, epsilon, discretization_size=1000, int_out=False)
                        output_dict['table_values'].append(noised_value)

    return output_dict

def extract_financial_data(text):
    output_dict = {
        'dollar_amounts': [],
        'table_values': []
    }
    
    # First extract tables and their contents
    table_pattern = r'<table.*?</table>'
    tables = re.findall(table_pattern, text, re.DOTALL)
    
    # Create text without tables for processing regular dollar amounts
    text_without_tables = re.sub(table_pattern, '', text, flags=re.DOTALL)
    
    # Process dollar amounts in main text
    dollar_pattern = r'\$\s+(\d+(?:,\d{3})*(?:\.\d+)?)'
    dollar_matches = re.finditer(dollar_pattern, text_without_tables)
    for match in dollar_matches:
        amount = float(match.group(1).replace(',', ''))
        output_dict['dollar_amounts'].append(amount)
    
    # Process tables separately
    for table in tables:
        rows = re.findall(r'<tr>(.*?)</tr>', table, re.DOTALL)
        
        for row_idx, row in enumerate(rows):
            if row_idx == 0:  # Skip header row
                continue
                
            cells = re.findall(r'<td>(.*?)</td>', row)
            
            for cell_idx, cell in enumerate(cells):
                if cell_idx <= 1:  # Skip first two columns
                    continue
                
                # Look for paired numbers in cells, e.g., "-27 (27)" or "$ 1,000 ($ 1,000)"
                paired_pattern = r'(-?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?)\s*\(\s*(-?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?)\s*\)'
                paired_matches = re.finditer(paired_pattern, cell)
                for match in paired_matches:
                    value = float(re.sub(r'[,\$()]', '', match.group(1)))
                    output_dict['table_values'].append(value)
                    # Since paired, we skip the second number to ensure it's treated as identical
                    break  # Assuming only one pair per cell
                
                else:
                    # If no paired numbers, look for single numbers
                    if '$' in cell:
                        cell_dollar_matches = re.finditer(r'\$\s+(\d+(?:,\d{3})*(?:\.\d+)?)', cell)
                        for match in cell_dollar_matches:
                            value = float(match.group(1).replace(',', ''))
                            output_dict['table_values'].append(value)
                    else:
                        # Look for plain numbers
                        plain_numbers = re.finditer(r'-?(\d+(?:,\d{3})*(?:\.\d+)?)', cell)
                        for match in plain_numbers:
                            value = float(match.group(1).replace(',', ''))
                            output_dict['table_values'].append(value)
    
    return output_dict

def determine_range(number):
    """Determine appropriate range for a number based on its magnitude"""
    magnitude = abs(number)
    if magnitude < 5: # Single digits
        return (0,10)
    elif magnitude < 10:
        return (0,20)
    elif magnitude < 100:  # two digits
        return (0, 200)
    elif magnitude < 1000:  # Three digits
        return (0, 2000)
    elif magnitude < 10000:  # Four digits
        return (0, 20000)
    elif magnitude < 100000:  # Five digits
        return (0, 200000)
    elif magnitude < 1000000:  # Six digits
        return (0, 2000000)
    elif magnitude < 10000000:  # Seven digits
        return (0, 20000000)
    else:  # Very large numbers
        # Create a range that's appropriate for the number
        power = math.floor(math.log10(magnitude))
        upper_bound = 10 ** (power + 1)
        return (0, upper_bound)

def apply_noise_to_financial_data(output_dict, epsilon=1.0):
    """Apply metric DP noise to extracted financial data"""
    noised_output = {
        'dollar_amounts': [],
        'table_values': []
    }
    
    # Apply noise to dollar amounts
    for amount in output_dict['dollar_amounts']:
        n_lower, n_upper = determine_range(amount)
        noised_amount = M_epsilon(amount, n_lower, n_upper,  epsilon,discretization_size=1000, int_out=False)
        noised_output['dollar_amounts'].append(noised_amount)
    
    # Apply noise to table values
    for value in output_dict['table_values']:
        n_lower, n_upper = determine_range(value)
        noised_value = M_epsilon(value, n_lower, n_upper, epsilon,discretization_size=1000, int_out=False)
        noised_output['table_values'].append(noised_value)
    
    return noised_output

def extract_and_noise_financial_data(text, epsilon=1.0):
    # First extract the data
    extracted_data = extract_financial_data(text)
    
    # Then apply noise
    noised_data = apply_noise_to_financial_data(extracted_data, epsilon)
    
    return noised_data



def update_text_with_noised_data(text, noised_data):
    dollar_idx = 0
    value_idx = 0
    
    # Function to handle dollar amount replacements
    def replace_dollar(match):
        nonlocal dollar_idx
        if dollar_idx < len(noised_data['dollar_amounts']):
            replacement = f"$ {noised_data['dollar_amounts'][dollar_idx]:.2f}"
            dollar_idx += 1
            return replacement
        return match.group(0)
    
    # Function to handle table value replacements
    def replace_table_cell(cell_content, is_dollar=False):
        nonlocal value_idx
        if value_idx >= len(noised_data['table_values']):
            return cell_content
            
        value = noised_data['table_values'][value_idx]
        value_idx += 1
        
        if is_dollar:
            return f"$ {value:.2f}"
        else:
            # Preserve negative sign if original was negative
            if value < 0:
                return f"-{abs(value):.2f}"
            return f"{value:.2f}"
    
    # Function to handle paired numbers in table cells
    def replace_paired_numbers(match):
        nonlocal value_idx
        if value_idx >= len(noised_data['table_values']):
            return match.group(0)
        
        value = noised_data['table_values'][value_idx]
        value_idx += 1
        
        # Determine if the pair is a dollar amount
        is_dollar = '$' in match.group(0)
        if is_dollar:
            noised_str = f"$ {value:.2f} ($ {value:.2f})"
        else:
            if value < 0:
                noised_str = f"-{abs(value):.2f} ({abs(value):.2f})"
            else:
                noised_str = f"{value:.2f} ({value:.2f})"
        return noised_str
    
    # First replace dollar amounts in non-table text
    modified_text = text
    parts = re.split(r'(<table.*?</table>)', modified_text, flags=re.DOTALL)
    
    for i in range(len(parts)):
        if not parts[i].startswith('<table'):  # Non-table text
            parts[i] = re.sub(r'\$\s+(\d+(?:,\d{3})*(?:\.\d+)?)', replace_dollar, parts[i])
        else:  # Table text
            table = parts[i]
            rows = re.findall(r'<tr>(.*?)</tr>', table, re.DOTALL)
            modified_rows = []
            
            for row_idx, row in enumerate(rows):
                cells = re.findall(r'<td>(.*?)</td>', row)
                modified_cells = []
                
                for cell_idx, cell in enumerate(cells):
                    # Skip header row (row_idx == 0) or first two columns
                    if row_idx == 0 or cell_idx <= 1:
                        modified_cells.append(f"<td>{cell}</td>")
                        continue
                        
                    modified_cell = cell
                    
                    # First handle paired numbers like "num1 (num2)"
                    paired_pattern = r'(-?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?)\s*\(\s*(-?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?)\s*\)'
                    if re.search(paired_pattern, modified_cell):
                        modified_cell = re.sub(paired_pattern, replace_paired_numbers, modified_cell)
                    else:
                        # Handle individual dollar amounts
                        if '$' in cell:
                            modified_cell = re.sub(
                                r'\$\s+(\d+(?:,\d{3})*(?:\.\d+)?)',
                                lambda m: replace_table_cell(m.group(0), True),
                                cell
                            )
                        else:
                            # Handle individual plain numbers
                            modified_cell = re.sub(
                                r'-?(\d+(?:,\d{3})*(?:\.\d+)?)',
                                lambda m: replace_table_cell(m.group(0)),
                                cell
                            )
                    
                    modified_cells.append(f"<td>{modified_cell}</td>")
                
                modified_rows.append(f"<tr>{''.join(modified_cells)}</tr>")
            
            parts[i] = f"<table>{''.join(modified_rows)}</table>"
    
    return ''.join(parts)

def process_and_update_text(original_text, epsilon=1.0):
    # First extract and noise the data
    noised_data = process_financial_data_with_noise(original_text, epsilon)
    
    # Then update the text with the noised values
    updated_text = update_text_with_noised_data(original_text, noised_data)
    
    return updated_text, noised_data





system_prompt = '''Answer the provided query with only a single number containing your answer'''

async def query_llm(text, client):
    """
    Asynchronously query a language model using OpenAI's API.

    Args:
        text (str): The user's input text to be processed by the model.
        system_text (str): The system message to set the context for the model.

    Returns:
        dict: The response from the language model.
    """
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user","content": text},
        ],
        temperature=0,
            max_tokens=32,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

def retry(exceptions, tries=3, delay=1):
    """
    A decorator that retries the decorated function in case of specified exceptions.

    Args:
        exceptions (Exception or tuple): The exception(s) to catch and retry on.
        tries (int): The maximum number of attempts. Defaults to 3.
        delay (int): The delay between retries in seconds. Defaults to 1.

    Returns:
        function: The decorated function with retry logic.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_tries = 0
            while current_tries < tries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    print(f"Exception caught: {e}")
                    await asyncio.sleep(delay)
                    current_tries += 1
            raise Exception("Max retries exceeded, failed to execute function.")

        return wrapper

    return decorator

@retry(Exception, tries=3, delay=1)
async def query_with_retry(text,client):
    """
    Query the language model with retry logic to handle potential exceptions.

    This function applies the retry decorator to handle any exceptions,
    including 502 Bad Gateway errors.

    Args:
        text (str): The user's input text to be processed by the model.
        system_text (str): The system message to set the context for the model.

    Returns:
        dict: The response from the language model.

    Raises:
        Exception: If the maximum number of retries is exceeded.
    """
    return await query_llm(text, client)

def is_number(string):
    """Check if a string can be converted to a float."""
    try:
        float(string)
        return True
    except ValueError:
        return False

def clean_number(text):
    """Extract and clean numerical values from text."""
    text = text.replace(',', '')
    text = text.replace('$', '')
    text = text.replace('%', '')
    # Find all numbers in the text
    numbers = [word for word in text.split() if is_number(word)]
    return float(numbers[0]) if numbers else None


def check_magnitude_and_sign_match(pred_num, true_num, tolerance=10):
    """
    Check if two numbers match after adjusting for sign and/or order of magnitude.
    
    Args:
        pred_num: Predicted number
        true_num: True number
        tolerance: Number of orders of magnitude difference allowed
    
    Returns:
        tuple: (bool indicating if numbers match after adjustments,
               adjusted prediction,
               list of adjustments made ['sign', 'magnitude'])
    """
    if pred_num == 0 or true_num == 0:
        return False, pred_num
    
    original_relative_error = abs(pred_num - true_num) / (abs(true_num) + 1e-4)
    if original_relative_error < .1:
        return False, pred_num  
    
    #adjustments_made = []
    adjusted_pred = pred_num
    
    # Check sign
    if (pred_num * true_num) < 0:
        adjusted_pred = -adjusted_pred
        #print("sign")
        #adjustments_made.append('sign')
    
    # Calculate order of magnitude difference
    magnitude_diff = math.floor(math.log10(abs(true_num))) - math.floor(math.log10(abs(adjusted_pred)))
    
    if (abs(magnitude_diff) <= tolerance) and (abs(magnitude_diff) > 0):
        # Adjust prediction by the order of magnitude
        adjusted_pred = adjusted_pred * (10 ** magnitude_diff)
        #adjustments_made.append('magnitude')
        
    # Check if the adjusted prediction matches within a small relative error
    relative_error = abs(adjusted_pred - true_num) / (abs(true_num) + 1e-4)
    
    if relative_error < .1:  # low tolerance
        #print("mag")
        return True, adjusted_pred#, adjustments_made

    return False, pred_num#, []


def extract_shared_context(query):
    """
    Extracts the shared context from the query by splitting at "Conversations:".
    The shared context is the part before "Conversations:".

    Args:
        query (str): The full query string.

    Returns:
        tuple: (shared_context (str), conversation_history (str))
    """
    split_marker = "Conversations:"
    if split_marker not in query:
        return query.strip(), ""
    parts = query.split(split_marker, 1)
    shared_context = parts[0].strip()
    conversation_history = split_marker + parts[1].strip()
    return shared_context, conversation_history

async def evaluate_model_with_multiple_noise(
    dataset,
    client,
    epsilon_values=[0.1, 0.5, 1.0, 2.0],
    max_questions=10000
):
    """
    Evaluates the model with multiple noise levels (epsilons) on a dataset.
    Processes the clean baseline separately and caches clean predictions.

    Args:
        dataset (list): A list of dialogue items.
        client (OpenAIClient): The OpenAI client instance.
        epsilon_values (list): List of epsilon values for DP sanitization.
        max_questions (int): Maximum number of noisy questions to process.

    Returns:
        dict: Results categorized by epsilon values, including the clean baseline.
    """
    results_by_epsilon = defaultdict(lambda: defaultdict(list))
    
    # Initialize cache for sanitized contexts per epsilon and dialogue_id
    context_noise_cache = {eps: {} for eps in epsilon_values}
    
    # Initialize cache for sanitized answers per epsilon and dialogue_id
    answers_noise_cache = {eps: defaultdict(dict) for eps in epsilon_values}
    
    # Initialize separate caches for the clean baseline
    clean_epsilon = 'clean'  # Identifier for the clean baseline
    context_noise_cache[clean_epsilon] = {}
    answers_noise_cache[clean_epsilon] = defaultdict(dict)

    # Cache for clean predictions
    clean_predictions_cache = defaultdict(dict)
    processed_clean_questions = 0
    
    print("[Evaluation] Starting clean baseline processing...")
    # --- Step 1: Process Clean Baseline ---
    for item in tqdm(dataset, desc="Processing clean baseline"):
        if processed_clean_questions >= max_questions:
            break
        dialogue_id = item.get('dialogue_id')
        query = item.get('query')
        true_answer = item.get('answer')
        turn = item.get('turn')
        
        if not dialogue_id or not query or true_answer == 0:
            continue
        
        try:
            # Extract shared context and conversation history
            shared_context, conversation_history = extract_shared_context(query)

            previous_answers = answers_noise_cache[clean_epsilon][dialogue_id]
            
            # Reconstruct the clean query
            clean_query = shared_context + "\n" + conversation_history

            # Replace placeholders with sanitized previous answers
            for idx, answer in previous_answers.items():
                placeholder = f"{{answer{idx}}}"
                clean_query = clean_query.replace(placeholder, str(answer))
            
            # Send the clean query to the model
            clean_prediction = await query_with_retry(clean_query, client)
            
            # Clean the model's prediction
            pred_num_clean = clean_number(clean_prediction)
            true_num = clean_number(true_answer)
            
            if pred_num_clean is not None and true_num is not None:
                # Compute relative error for clean baseline
                relative_error = abs((pred_num_clean - true_num) / (abs(true_num) + 1e-3))
                results_by_epsilon[clean_epsilon]['relative_errors'].append(relative_error)
                
                # Store the clean prediction for consistency checks
                clean_predictions_cache[dialogue_id][turn] = pred_num_clean
                
                # Store the clean answer for future turns
                answers_noise_cache[clean_epsilon][dialogue_id][turn] = pred_num_clean
                
            else:
                print(f"[Metrics] Warning: Unable to compute clean metrics for dialogue_id {dialogue_id}")

            processed_clean_questions += 1
        
        except Exception as e:
            print(f"[Error] Processing clean baseline for dialogue_id {dialogue_id}: {e}")
            traceback.print_exc()
            continue
    
    print("[Evaluation] Clean baseline processing completed.")
    print("[Evaluation] Starting noisy epsilon processing...")
    
    # Initialize a counter for the number of processed noisy questions
    processed_noisy_questions = 0
    
    # --- Step 2: Process Noisy Epsilons ---
    for item in tqdm(dataset, desc="Processing noisy epsilons"):
        if processed_noisy_questions >= max_questions:
            break
        
        dialogue_id = item.get('dialogue_id')
        query = item.get('query')
        true_answer = item.get('answer')
        turn = item.get('turn')
        
        if not dialogue_id or not query or true_answer == 0:
            continue
        
        try:
            # Extract shared context and conversation history
            shared_context, conversation_history = extract_shared_context(query)
            
            if not shared_context:
                print(f"[Processing] Warning: Shared context is empty for dialogue_id {dialogue_id}")
            
            # Iterate over all epsilon values
            for eps in epsilon_values:
                if processed_noisy_questions >= max_questions:
                    break
                
                # Check if sanitized context is already cached
                if dialogue_id not in context_noise_cache[eps]:
                    # Sanitize the shared context with noise based on epsilon
                    sanitized_context, _ = process_and_update_text(shared_context, eps)
                    context_noise_cache[eps][dialogue_id] = sanitized_context
                else:
                    sanitized_context = context_noise_cache[eps][dialogue_id]
                
                # Retrieve sanitized previous answers
                previous_answers = answers_noise_cache[eps][dialogue_id]
                
                # Reconstruct the full sanitized query
                sanitized_query = sanitized_context + "\n" + conversation_history
                
                # Replace placeholders with sanitized previous answers
                for idx, answer in previous_answers.items():
                    placeholder = f"{{answer{idx}}}"
                    sanitized_query = sanitized_query.replace(placeholder, str(answer))
                
                # Send the sanitized query to the model
                model_response = await query_with_retry(sanitized_query, client)
                
                # Clean the model's prediction
                pred_num = clean_number(model_response)
                true_num = clean_number(true_answer)
                
                if pred_num is not None and true_num is not None:
                    # Sanitize prediction based on epsilon
                    is_match, adjusted_pred = check_magnitude_and_sign_match(pred_num, true_num)
                    final_pred = adjusted_pred if is_match else pred_num
                    
                    relative_error = abs((final_pred - true_num) / (abs(true_num) + 1e-3))
                    
                    # Retrieve clean prediction for consistency
                    clean_pred = clean_predictions_cache.get(dialogue_id, {}).get(turn)
                    
                    if clean_pred is not None:
                        pred_consistency = abs((final_pred - clean_pred) / (abs(clean_pred) + 1e-3))
                        results_by_epsilon[eps]['relative_errors'].append(relative_error)
                        results_by_epsilon[eps]['consistencies'].append(pred_consistency)
                    
                    # Store the sanitized answer for future turns
                    answers_noise_cache[eps][dialogue_id][turn] = pred_num
                    
                    # Increment the processed noisy questions counter
                else:
                    print(f"[Metrics] Warning: Unable to compute metrics for dialogue_id {dialogue_id}, epsilon {eps}")

            processed_noisy_questions += 1
        
        except Exception as e:
            print(f"[Error] Processing dialogue_id {dialogue_id}: {e}")
            traceback.print_exc()
            continue
    
    print("[Evaluation] Noisy epsilon processing completed.")
    return results_by_epsilon

def plot_results(results_by_epsilon, output_dir="results"):
    """
    Create and save line plots for median relative error and prediction consistency, 
    including the clean baseline. Also saves numerical results to a text file.

    Args:
        results_by_epsilon (dict): The evaluation results categorized by epsilon values
        output_dir (str): Directory to save results and plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Access 'clean' data without modifying the original dictionary
    clean_data = results_by_epsilon.get('clean', {})
    
    # Separate noisy epsilons
    noisy_epsilons = sorted([eps for eps in results_by_epsilon if eps != 'clean'])
    
    # Prepare statistics for Relative Error
    medians_rel_err = []
    q25_rel_err = []
    q75_rel_err = []
    
    for eps in noisy_epsilons:
        rel_errors = results_by_epsilon[eps]['relative_errors']
        if rel_errors:
            medians_rel_err.append(np.median(rel_errors))
            q25_rel_err.append(np.percentile(rel_errors, 25))
            q75_rel_err.append(np.percentile(rel_errors, 75))
        else:
            medians_rel_err.append(np.nan)
            q25_rel_err.append(np.nan)
            q75_rel_err.append(np.nan)
    
    # Clean baseline Relative Error
    clean_median_rel_err = None
    if clean_data and clean_data.get('relative_errors'):
        clean_median_rel_err = np.median(clean_data['relative_errors'])
        clean_q25_rel_err = np.percentile(clean_data['relative_errors'], 25)
        clean_q75_rel_err = np.percentile(clean_data['relative_errors'], 75)
    
    # Prepare statistics for Prediction Consistency
    medians_consist = []
    q25_consist = []
    q75_consist = []
    
    for eps in noisy_epsilons:
        consistencies = results_by_epsilon[eps]['consistencies']
        if consistencies:
            medians_consist.append(np.median(consistencies))
            q25_consist.append(np.percentile(consistencies, 25))
            q75_consist.append(np.percentile(consistencies, 75))
        else:
            medians_consist.append(np.nan)
            q25_consist.append(np.nan)
            q75_consist.append(np.nan)
    
    # Create Figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Relative Error Plot ---
    ax1.plot(noisy_epsilons, medians_rel_err, marker='o', color='blue', label='Median Relative Error')
    ax1.fill_between(noisy_epsilons, q25_rel_err, q75_rel_err, color='blue', alpha=0.2, label='25th-75th Percentile')
    
    if clean_median_rel_err is not None:
        ax1.axhline(y=clean_median_rel_err, color='red', linestyle='--', label='Clean Baseline')
    
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Relative Error')
    ax1.set_title('Relative Error vs. Epsilon')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # --- Prediction Consistency Plot ---
    ax2.plot(noisy_epsilons, medians_consist, marker='s', color='green', label='Median Prediction Consistency')
    ax2.fill_between(noisy_epsilons, q25_consist, q75_consist, color='green', alpha=0.2, label='25th-75th Percentile')
    
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Prediction Consistency')
    ax2.set_title('Prediction Consistency vs. Epsilon')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'privacy_evaluation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results to a text file
    results_path = os.path.join(output_dir, 'numerical_results.txt')
    with open(results_path, 'w') as f:
        f.write("Privacy Evaluation Results\n")
        f.write("=========================\n\n")
        
        for i, eps in enumerate(noisy_epsilons):
            f.write(f"Epsilon: {eps:.1f}\n")
            
            if not np.isnan(medians_rel_err[i]):
                f.write(f"  Median Relative Error: {medians_rel_err[i]:.4f}\n")
                f.write(f"  25th Percentile Relative Error: {q25_rel_err[i]:.4f}\n")
                f.write(f"  75th Percentile Relative Error: {q75_rel_err[i]:.4f}\n")
            else:
                f.write("  Median Relative Error: N/A\n")
                f.write("  25th Percentile Relative Error: N/A\n")
                f.write("  75th Percentile Relative Error: N/A\n")
            
            if not np.isnan(medians_consist[i]):
                f.write(f"  Median Prediction Consistency: {medians_consist[i]:.4f}\n")
                f.write(f"  25th Percentile Prediction Consistency: {q25_consist[i]:.4f}\n")
                f.write(f"  75th Percentile Prediction Consistency: {q75_consist[i]:.4f}\n")
            else:
                f.write("  Median Prediction Consistency: N/A\n")
                f.write("  25th Percentile Prediction Consistency: N/A\n")
                f.write("  75th Percentile Prediction Consistency: N/A\n")
            f.write("\n")
        
        if clean_median_rel_err is not None:
            f.write("Clean Baseline:\n")
            f.write(f"  Median Relative Error: {clean_median_rel_err:.4f}\n")
            f.write(f"  25th Percentile Relative Error: {clean_q25_rel_err:.4f}\n")
            f.write(f"  75th Percentile Relative Error: {clean_q75_rel_err:.4f}\n")
        else:
            f.write("Clean Baseline Relative Error: N/A\n")

    print(f"\nResults saved to {output_dir}/")
    print(f"- Plot saved as: privacy_evaluation_results.png")
    print(f"- Numerical results saved as: numerical_results.txt")
        
# Run evaluation
async def main():
    """Main execution function"""
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("ChanceFocus/flare-convfinqa")
    
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    print("Starting evaluation...")
    # Run evaluation with specified epsilon values
    epsilon_values = [0.1, 0.5, 1.0, 2.0]  # Can be modified as needed
    results = await evaluate_model_with_multiple_noise(
        ds['test'], 
        client,
        epsilon_values=epsilon_values,
        max_questions=10000  # Adjust as needed
    )
    
    # Plot and display results
    plot_results(results)
    return results

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())