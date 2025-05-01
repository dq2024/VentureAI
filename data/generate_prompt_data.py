#!/usr/bin/env python3
import json
import pandas as pd
import re
from typing import List, Dict, Set
import os
import csv
import random

# Section mappings and patterns
SECTION_MAPPING = {
    "see": "Attractions",
    "do": "Activities", 
    "buy": "Shops",
    "eat": "Restaurants",
    "drink": "Bars",
    "sleep": "Hotels"
}
IMPORTANT_SECTIONS = list(SECTION_MAPPING.keys())

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
    # Basic itinerary requests
    # TODO add this back in later after first finetune
    # "Can you create a {days}-day travel itinerary from {from_city} to {to_city}?",
    # "I'm planning a {days}-day trip from {from_city} to {to_city}. Could you build a day-by-day travel itinerary?",
    # "Design a detailed {days}-day travel itinerary for a trip from {from_city} to {to_city}.",
    # "Help me plan a {days}-day itinerary for my trip from {from_city} to {to_city}.",
    # "Please draft a {days}-day travel plan from {from_city} to {to_city}.",
    # "I'd love a day-by-day {days}-day itinerary for my trip from {from_city} to {to_city}.",
    # "What would a {days}-day itinerary look like for a trip from {from_city} to {to_city}?",
    # "Can you suggest a detailed {days}-day travel itinerary from {from_city} to {to_city}?",
    # # Budget-focused itineraries
    # "Can you create a {days}-day itinerary with a total budget of $500 for traveling from {from_city} to {to_city}?",
    # "I need a {days}-day travel itinerary from {from_city} to {to_city} with a budget of around $100 per day.",
    # "Please plan a {days}-day itinerary for a trip from {from_city} to {to_city} with a luxury budget of $500 per day.",
    # "Create a {days}-day travel itinerary from {from_city} to {to_city} with a moderate budget of $200-300 per day.",
    # # Interest-specific itineraries
    # "Create a {days}-day restaurant focused itinerary for my trip from {from_city} to {to_city}.",
    # "I need a {days}-day itinerary with the best attractions for my journey from {from_city} to {to_city}.",
    # "Please build a {days}-day itinerary highlighting the best hotels and restaurants in {to_city} for visitors from {from_city}.",
    # "Generate a {days}-day itinerary that includes must-visit bars and attractions in {to_city} for travelers from {from_city}."
    "I would like to travel from {from_city} to {to_city} for {days} days. Please create a day by day itinerary."
]

def load_city_list(file_path: str) -> List[str]:
    """Load city names from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        cities = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(cities)} cities from '{file_path}'.")
    return cities

def load_city_data_from_csv(file_path: str, allowed_cities: Set[str]) -> Dict[str, List[Dict]]:
    """Load structured city data from the CSV, filtering to only those in allowed_cities."""
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Loaded CSV with {len(df)} rows.")
    
    city_data: Dict[str, List[Dict]] = {}
    for _, row in df.iterrows():
        city_name = row.get('article', '').strip()
        if not city_name or city_name not in allowed_cities:
            continue
        city_data.setdefault(city_name, []).append(row.dropna().to_dict())
    
    print(f"Processed data for {len(city_data)} cities (filtered).")
    return city_data

def analyze_city_data(city_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Analyze if cities have sufficient data in important sections."""
    city_analysis = {}
    for city, items in city_data.items():
        section_counts = {section: 0 for section in IMPORTANT_SECTIONS}
        for item in items:
            t = item.get('type')
            if t in IMPORTANT_SECTIONS:
                section_counts[t] += 1
        sections_with_data = sum(1 for c in section_counts.values() if c > 0)
        completion = (sections_with_data / len(IMPORTANT_SECTIONS)) * 100
        city_analysis[city] = {
            'section_counts': section_counts,
            'sections_with_data': sections_with_data,
            'completion_percentage': completion,
            'has_enough_data': sections_with_data >= 3
        }
    return city_analysis

def extract_structured_data(city_items: List[Dict]) -> Dict[str, List[Dict]]:
    """Extract and structure data for a city's items."""
    structured = {section: [] for section in IMPORTANT_SECTIONS}
    for item in city_items:
        t = item.get('type')
        if t in IMPORTANT_SECTIONS:
            structured_item = {
                'title': item.get('title', ''),
                'price': item.get('price', ''),
                'description': item.get('description', '')
            }
            if structured_item['title'] or structured_item['description']:
                structured[t].append(structured_item)
    return structured

def sample_items_for_response(structured_data: Dict[str, List[Dict]],
                              min_items: int = 3, max_items: int = 5) -> Dict[str, List[Dict]]:
    """Randomly sample a subset of items from each section."""
    sampled = {}
    for section, items in structured_data.items():
        if items:
            n = min(max(min_items, random.randint(min_items, max_items)), len(items))
            sampled[section] = random.sample(items, n)
        else:
            sampled[section] = []
    return sampled

def create_prompts_responses(city_analysis: Dict[str, Dict],
                             city_data: Dict[str, List[Dict]],
                             from_cities: List[str],
                             prompts_per_city: int = 3) -> List[Dict[str, str]]:
    """Create structured prompts and responses for cities with enough data."""
    data = []
    valid = [c for c,a in city_analysis.items() if a['has_enough_data']]
    print(f"Found {len(valid)} cities with sufficient data.")
    
    for city in valid:
        structured = extract_structured_data(city_data[city])
        for _ in range(prompts_per_city):
            sampled = sample_items_for_response(structured)
            parts = []
            for sec in IMPORTANT_SECTIONS:
                items = sampled[sec]
                if not items: continue
                block = [f"[{SECTION_MAPPING[sec]}]"]
                for it in items:
                    line = f"- {it['title']}" + (f" ({it['price']})" if it['price'] else "")
                    if it['description']:
                        line += f"\n  {it['description']}"
                    block.append(line)
                parts.append("\n".join(block))
            response_text = "\n\n".join(parts).strip()
            if not response_text:
                continue
            prompt = random.choice(PROMPT_VARIATIONS).format(
                from_city=random.choice(from_cities),
                to_city=city,
                days=random.choice([3,4,5,6,7])
            )
            data.append({
                "title": city,
                "prompt": prompt + f" The context below is about {city}. Please use it in your response.\n",
                "response": "[Context]\n" + response_text
            })
    print(f"Total prompts/responses created: {len(data)}")
    return data

def save_to_csv(data: List[Dict[str, str]], output_file: str):
    """Save the processed data to a CSV file."""
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

def generate_city_report(city_analysis: Dict[str, Dict], output_file: str = "city_analysis_report.csv"):
    """Generate a report on city data quality."""
    report_rows = []
    for city, a in city_analysis.items():
        row = {
            'city': city,
            'sections_with_data': a['sections_with_data'],
            'completion_percentage': a['completion_percentage'],
            'has_enough_data': a['has_enough_data']
        }
        for sec in IMPORTANT_SECTIONS:
            row[f"{sec}_count"] = a['section_counts'][sec]
        report_rows.append(row)
    
    df = pd.DataFrame(report_rows)
    df = df.sort_values(['has_enough_data','completion_percentage'], ascending=[False,False])
    
    # Rename columns with friendly names
    rename_map = {
        sec + "_count": f"{SECTION_MAPPING[sec]}_count"
        for sec in IMPORTANT_SECTIONS
    }
    df.rename(columns=rename_map, inplace=True)
    df.to_csv(output_file, index=False)
    print(f"City analysis report saved to '{output_file}'.")
    
    valid = df[df['has_enough_data']]
    print(f"\nSummary:")
    print(f"Total cities: {len(df)}")
    print(f"With enough data: {len(valid)} ({len(valid)/len(df)*100:.1f}%)")
    print(f"Average sections: {df['sections_with_data'].mean():.1f} of {len(IMPORTANT_SECTIONS)}")
    print("\nSection coverage:")
    for sec in IMPORTANT_SECTIONS:
        cnt = (df[f"{SECTION_MAPPING[sec]}_count"] > 0).sum()
        print(f"  {SECTION_MAPPING[sec]}: {cnt} cities ({cnt/len(df)*100:.1f}%)")

def main():
    city_list_path     = "cities/city_list.txt"
    city_data_csv_path = "wikivoyage-listings-en.csv"
    output_csv_path    = "prompt_wikidata.csv"

    if not os.path.exists(city_list_path):
        print(f"Error: '{city_list_path}' not found.")
        return
    if not os.path.exists(city_data_csv_path):
        print(f"Error: '{city_data_csv_path}' not found.")
        return

    # 1) load city list
    cities = load_city_list(city_list_path)

    # 2) use those cities as origin pool
    FROM_CITIES = cities

    # 3) load & filter wikivoyage data
    city_data = load_city_data_from_csv(city_data_csv_path, set(cities))

    # 4) analyze, report, and build prompts
    analysis    = analyze_city_data(city_data)
    generate_city_report(analysis)
    prompts     = create_prompts_responses(analysis, city_data, FROM_CITIES)
    save_to_csv(prompts, output_csv_path)

    # sanity check
    try:
        df = pd.read_csv(output_csv_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print("Output CSV shape:", df.shape)
    except Exception as e:
        print("Error re-loading output CSV:", e)

if __name__ == "__main__":
    main()
