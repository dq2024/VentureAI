import openai
import csv
import os
import time
import random
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_solution_with_openai(prompt, context, model="gpt-4o-mini", max_retries=5):
    """Generate a travel itinerary using OpenAI API based on prompt and context"""
    full_prompt = (
        "You are a travel planning assistant. Create a detailed travel itinerary based on the following request. "
        "Use ONLY the information provided in the [Context] section. Do not invent or add any places, attractions, "
        "restaurants, bars, or hotels that are not mentioned in the context. If the context doesn't have enough "
        "information for certain days or activities, work with what you have and be creative with timing and "
        "organization.\n\n"
        f"{prompt}\n\n"
        f"{context}\n\n"
        "Organize your response as a day-by-day itinerary with specific activities, places to eat, and accommodations "
        "for each day."
    )
    
    messages = [{"role": "user", "content": full_prompt}]
    
    retries = 0
    while retries < max_retries:
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,  # Slightly higher temperature for more creative itineraries
                max_tokens=1500   # Limit response length to avoid excessive outputs
            )
            solution = response.choices[0].message.content.strip()
            return solution
        except openai.RateLimitError as e:
            wait_time = random.uniform(5, 15)  # Increased wait time for rate limits
            print(f"[RateLimit] Waiting {wait_time:.2f}s before retrying...")
            time.sleep(wait_time)
            retries += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(random.uniform(2, 5))  # Wait a bit on general errors
            retries += 1
    
    return None

def extract_data_from_csv(csv_file_path, max_samples=None):
    """Extract data from the CSV file and optionally limit total samples"""
    prompts_responses = []
    
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompt = row['prompt']
            response = row['response']  # This should be the "[Context]..." content
            
            prompts_responses.append((prompt, response))
    
    # Sample data if needed
    if max_samples and max_samples < len(prompts_responses):
        prompts_responses = random.sample(prompts_responses, max_samples)
    
    print(f"Extracted {len(prompts_responses)} prompt-context pairs")
    return prompts_responses

def main():
    input_csv_file = '../split_data/split_4.csv'
    output_csv_file = './train_data/split_4_train.csv'
    
    # Configuration
    max_samples = None     # Set to a number to limit total samples, or None for all
    batch_size = 3         # Safe for gpt-4o-mini without exploding
    max_workers = batch_size
    
    # Extract data
    prompt_context_pairs = extract_data_from_csv(
        input_csv_file, 
        max_samples=max_samples
    )
    
    # Shuffle to distribute workload
    random.shuffle(prompt_context_pairs)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for idx, (prompt, context) in enumerate(tqdm(prompt_context_pairs, desc="Processing prompts")):
            future = executor.submit(generate_solution_with_openai, prompt, context)
            futures.append((idx, prompt, context, future))

            # If we reach the batch size, wait for all to finish
            if len(futures) >= batch_size:
                for idx, prompt, context, future in futures:
                    try:
                        response = future.result()
                        results.append({
                            "prompt": prompt,
                            "context": context,
                            "response": response if response else "ERROR: No response"
                        })
                    except Exception as e:
                        print(f"[Batch Error] {e}")
                        results.append({
                            "prompt": prompt,
                            "context": context,
                            "response": "ERROR: Exception occurred"
                        })

                # Clear the futures after processing
                futures = []

                # Delay between batches to avoid rate limiting
                time.sleep(random.uniform(2, 4))

        # Process any remaining futures
        for idx, prompt, context, future in futures:
            try:
                response = future.result()
                results.append({
                    "prompt": prompt,
                    "context": context,
                    "response": response if response else "ERROR: No response"
                })
            except Exception as e:
                print(f"[Final Batch Error] {e}")
                results.append({
                    "prompt": prompt,
                    "context": context,
                    "response": "ERROR: Exception occurred"
                })

    # Save the results
    with open(output_csv_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["prompt", "context", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Done! Saved {len(results)} rows to {output_csv_file}")

if __name__ == "__main__":
    main()