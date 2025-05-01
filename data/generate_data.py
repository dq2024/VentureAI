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

def generate_solution_with_openai(prompt, model="gpt-4o-mini", max_retries=5):
    messages = [{"role": "user", "content": prompt}]
    
    retries = 0
    while retries < max_retries:
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
            )
            solution = response.choices[0].message.content.strip()
            return solution
        except openai.RateLimitError as e:
            wait_time = random.uniform(5, 10)
            print(f"[RateLimit] Waiting {wait_time:.2f}s before retrying...")
            time.sleep(wait_time)
            retries += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    return None

def extract_prompts_and_data(csv_file_path):
    prompts_responses = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompt = row['prompt']
            response = row['response']
            prompts_responses.append((prompt, response))
    return prompts_responses

def main():
    input_csv_file = './prompt_wikidata.csv'
    output_csv_file = './results.csv'

    prompts_responses = extract_prompts_and_data(input_csv_file)
    results = []

    batch_size = 3   # Safe for gpt-4o-mini without exploding
    max_workers = batch_size

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for idx, (prompt, data) in enumerate(tqdm(prompts_responses, desc="Processing prompts")):
            full_prompt = (
                "I am providing you with a Query that will ask you to plan a trip for me with instructions. "
                "I will also provide you with context data regarding the location I want to travel to. "
                "Only use the context provided to answer the prompt; do not come up with anything new.\n\n"
                f"{prompt}\n\n"
                f"{data}\n"
            )

            future = executor.submit(generate_solution_with_openai, full_prompt)
            futures.append((idx, prompt, future))

            # If we reach the batch size, wait for all to finish
            if len(futures) >= batch_size:
                for idx, prompt, future in futures:
                    try:
                        response = future.result()
                        results.append({
                            "prompt": prompt,
                            "response": response if response else "ERROR: No response"
                        })
                    except Exception as e:
                        print(f"[Batch Error] {e}")
                        results.append({
                            "prompt": prompt,
                            "response": "ERROR: Exception occurred"
                        })

                # Clear the futures after processing
                futures = []

                # Tiny sleep after each batch
                time.sleep(random.uniform(1.5, 2.5))

        # Process any remaining futures
        for idx, prompt, future in futures:
            try:
                response = future.result()
                results.append({
                    "prompt": prompt,
                    "response": response if response else "ERROR: No response"
                })
            except Exception as e:
                print(f"[Final Batch Error] {e}")
                results.append({
                    "prompt": prompt,
                    "response": "ERROR: Exception occurred"
                })

    # Save the results
    with open(output_csv_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["prompt", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Done! Saved {len(results)} rows to {output_csv_file}")

if __name__ == "__main__":
    main()
