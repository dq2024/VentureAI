import json
import pandas as pd
import re
from typing import List, Dict
import os
import csv
import random

# Random cities to simulate traveling FROM
FROM_CITIES = [
    "London", "Paris", "Tokyo", "Sydney", "Toronto", "Dubai", "Rome", "Berlin", "Singapore", "Los Angeles"
]

# List of important sections we care about
IMPORTANT_SECTIONS = [
    "Understand", "Get in", "Get around", "See", "Do", "Buy", "Eat", "Drink", "Sleep", "Go next"
]

# Junk patterns to remove
JUNK_PATTERNS = [
    r'\[\[File:.*?\]\]',
    r'Image:.*?\|',
    r'thumb\|.*?\|',
    r'\{\{.*?\}\}',
    r'\|.*?=',
    r'wikipedia=.*',
    r'wikidata=.*',
    r'lastedit=.*',
    r'\<.*?\>',
    r'\[https?:.*?\]',
    r'\[http.*?\]',
    r'\[\[.*?:.*?\]\]',
    r'==+See also==+',
    r'==+Related pages==+',
    r'==+External links==+',
    r'==+References==+',
    r'\*\s*$',
    r'\|.*?\n'
]

PROMPT_VARIATIONS = [
    "Can you create a {days}-day travel itinerary from {from_city} to {to_city}?",
    "I'm planning a {days}-day trip from {from_city} to {to_city}. Could you build a day-by-day travel itinerary?",
    "Design a detailed {days}-day travel itinerary for a trip from {from_city} to {to_city}.",
    "Help me plan a {days}-day itinerary for my trip from {from_city} to {to_city}.",
    "Please draft a {days}-day travel plan from {from_city} to {to_city}.",
    "I'd love a day-by-day {days}-day itinerary for my trip from {from_city} to {to_city}.",
    "What would a {days}-day itinerary look like for a trip from {from_city} to {to_city}?",
    "Can you suggest a detailed {days}-day travel itinerary from {from_city} to {to_city}?"
]


def load_city_list(file_path: str) -> List[str]:
    titles = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            title = line.strip()
            if title:
                titles.append(title)
    print(f"Loaded {len(titles)} cities from '{file_path}'.")
    return titles


def load_cleaned_wikivoyage(file_path: str) -> Dict[str, str]:
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    pages = data.get("mediawiki", {}).get("page", [])
    for page in pages:
        title = page.get("title")
        if not title or "redirect" in page:
            continue
        revision = page.get("revision", {})
        text = revision.get("text", "")
        if isinstance(text, dict):
            text = text.get("#text", "")
        if text:
            mapping[title] = text
    print(f"Loaded {len(mapping)} pages from cleaned Wikivoyage.")
    return mapping


def clean_text(text: str) -> str:
    for pattern in JUNK_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def extract_relevant_sections(wikivoyage_text: str) -> str:
    sections = {}
    current_section = None
    buffer = []
    lines = wikivoyage_text.split('\n')

    for line in lines:
        main_section = re.match(r'^==([^=]+)==$', line.strip())
        subsection = re.match(r'^===([^=]+)===$', line.strip())

        if main_section:
            if current_section and buffer:
                cleaned_text = '\n'.join(buffer).strip()
                cleaned_text = re.sub(r'^=+\s*(.*?)\s*=+$', '', cleaned_text, flags=re.MULTILINE)  # Remove rogue headers
                cleaned_text = clean_text(cleaned_text)
                sections[current_section] = cleaned_text
                buffer = []
            section_title = main_section.group(1).strip()
            current_section = section_title
        elif subsection:
            # Skip subsection titles entirely
            continue
        elif current_section:
            buffer.append(line)

    if current_section and buffer:
        cleaned_text = '\n'.join(buffer).strip()
        cleaned_text = re.sub(r'^=+\s*(.*?)\s*=+$', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = clean_text(cleaned_text)
        sections[current_section] = cleaned_text

    extracted_content = []
    for section in IMPORTANT_SECTIONS:
        content = sections.get(section)
        if content:
            extracted_content.append(f"[{section}]\n{content}\n")

    return '\n'.join(extracted_content).strip()



def create_prompts_responses(titles: List[str], mapping: Dict[str, str], prompts_per_city: int = 5) -> List[Dict[str, str]]:
    data = []
    for idx, title in enumerate(titles, start=1):
        text = mapping.get(title)
        if not text:
            print(f"Title {idx}: '{title}' not found in JSON. Skipping.")
            continue

        structured_text = extract_relevant_sections(text)
        if not structured_text:
            print(f"Title {idx}: '{title}' has no relevant sections. Skipping.")
            continue

        for _ in range(prompts_per_city):
            from_city = random.choice(FROM_CITIES)
            days = random.choice([5, 6, 7, 8, 9, 10])
            prompt_template = random.choice(PROMPT_VARIATIONS)
            prompt = prompt_template.format(from_city=from_city, to_city=title, days=days) + f"\n\nThe information below is about {title}. Use it in your response.\n"

            data.append({
                "title": title,
                "prompt": prompt,
                "response": structured_text
            })

    print(f"\nTotal structured prompts and responses created: {len(data)}")
    return data




def save_to_csv(data: List[Dict[str, str]], output_file: str):
    if not data:
        print("No data to save.")
        return

    df = pd.DataFrame(data)
    df['response'] = df['response'].str.replace('"', '""')
    df.to_csv(
        output_file,
        index=False,
        encoding='utf-8-sig',
        quoting=csv.QUOTE_ALL,
        escapechar='\\'
    )
    print(f"Data successfully saved to '{output_file}'.")


def main():
    city_list_path = "city_list.txt"
    cleaned_wikivoyage_path = "cleaned_wikivoyage.json"
    output_csv_path = "prompt_wikidata.csv"

    if not os.path.exists(city_list_path):
        print(f"Error: '{city_list_path}' does not exist.")
        return
    if not os.path.exists(cleaned_wikivoyage_path):
        print(f"Error: '{cleaned_wikivoyage_path}' does not exist.")
        return

    titles = load_city_list(city_list_path)
    mapping = load_cleaned_wikivoyage(cleaned_wikivoyage_path)

    prompts_responses = create_prompts_responses(titles, mapping)
    save_to_csv(prompts_responses, output_csv_path)

    try:
        df_loaded = pd.read_csv(output_csv_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"CSV successfully loaded back into Pandas. Shape: {df_loaded.shape}")
    except Exception as e:
        print(f"Error loading CSV into Pandas: {e}")


if __name__ == "__main__":
    main()