import json
import pandas as pd
import re
from typing import List, Dict, Set
import os
import csv
import random

# Random cities to simulate traveling FROM
FROM_CITIES = [
    # Major European cities
    "London", "Paris", "Rome", "Berlin", "Madrid", "Amsterdam", "Brussels", "Vienna", "Prague", "Budapest",
    "Copenhagen", "Stockholm", "Oslo", "Helsinki", "Warsaw", "Lisbon", "Athens", "Dublin", "Zurich", "Barcelona",
    
    # North American cities
    "New York", "Los Angeles", "Chicago", "Toronto", "Montreal", "Vancouver", "Boston", "San Francisco", 
    "Washington DC", "Miami", "Seattle", "Denver", "Atlanta", "Dallas", "Houston", "Mexico City", "Calgary",
    
    # Asian cities
    "Tokyo", "Beijing", "Shanghai", "Hong Kong", "Seoul", "Singapore", "Bangkok", "Kuala Lumpur", "Mumbai", 
    "Delhi", "Dubai", "Istanbul", "Tel Aviv", "Manila", "Jakarta", "Ho Chi Minh City", "Taipei",
    
    # Oceania cities
    "Sydney", "Melbourne", "Auckland", "Brisbane", "Perth", "Wellington", "Adelaide",
    
    # South American cities
    "Rio de Janeiro", "São Paulo", "Buenos Aires", "Lima", "Santiago", "Bogotá", "Caracas",
    
    # African cities
    "Cairo", "Cape Town", "Johannesburg", "Nairobi", "Casablanca", "Lagos", "Marrakesh"
]

# List of important sections we care about (with original type names as keys and display names as values)
SECTION_MAPPING = {
    "see": "Attractions",
    "do": "Activities", 
    "buy": "Shops",
    "eat": "Restaurants",
    "drink": "Bars",
    "sleep": "Hotels"
}

# Original section names to use for accessing data
IMPORTANT_SECTIONS = list(SECTION_MAPPING.keys())

# Junk patterns to remove (for wikivoyage text if still needed)
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
    "Can you create a {days}-day travel itinerary from {from_city} to {to_city}?",
    "I'm planning a {days}-day trip from {from_city} to {to_city}. Could you build a day-by-day travel itinerary?",
    "Design a detailed {days}-day travel itinerary for a trip from {from_city} to {to_city}.",
    "Help me plan a {days}-day itinerary for my trip from {from_city} to {to_city}.",
    "Please draft a {days}-day travel plan from {from_city} to {to_city}.",
    "I'd love a day-by-day {days}-day itinerary for my trip from {from_city} to {to_city}.",
    "What would a {days}-day itinerary look like for a trip from {from_city} to {to_city}?",
    "Can you suggest a detailed {days}-day travel itinerary from {from_city} to {to_city}?"
    
    # Budget-focused itineraries with specific numbers
    "Can you create a {days}-day itinerary with a total budget of $500 for traveling from {from_city} to {to_city}?",
    "I need a {days}-day travel itinerary from {from_city} to {to_city} with a budget of around $100 per day.",
    "Please plan a {days}-day itinerary for a trip from {from_city} to {to_city} with a luxury budget of $500 per day.",
    "Create a {days}-day travel itinerary from {from_city} to {to_city} with a moderate budget of $200-300 per day.",
    
    # Interest-specific itineraries
    "Create a {days}-day restaurant focused itinerary for my trip from {from_city} to {to_city}.",
    "I need a {days}-day itinerary with the best attractions for my journey from {from_city} to {to_city}.",
    "Please build a {days}-day itinerary highlighting the best hotels and restaurants in {to_city} for visitors from {from_city}.",
    "Generate a {days}-day itinerary that includes must-visit bars and attractions in {to_city} for travelers from {from_city}."
]


def load_city_list(file_path: str) -> List[str]:
    """Load city names from a text file."""
    titles = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            title = line.strip()
            if title:
                titles.append(title)
    print(f"Loaded {len(titles)} cities from '{file_path}'.")
    return titles


def load_city_data_from_csv(file_path: str) -> Dict[str, List[Dict]]:
    """Load structured city data from the CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns.")
        
        # Group the data by city
        city_data = {}
        for _, row in df.iterrows():
            city_name = row['article']
            if city_name not in city_data:
                city_data[city_name] = []
            
            # Convert row to dict and append to the city's data
            item_data = {col: row[col] for col in df.columns if pd.notna(row[col])}
            city_data[city_name].append(item_data)
        
        print(f"Processed data for {len(city_data)} cities.")
        return city_data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}


def analyze_city_data(city_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Analyze if cities have sufficient data in important sections."""
    city_analysis = {}
    
    for city, items in city_data.items():
        # Count items in each important section
        section_counts = {section: 0 for section in IMPORTANT_SECTIONS}
        
        for item in items:
            item_type = item.get('type')
            if item_type in IMPORTANT_SECTIONS:
                section_counts[item_type] += 1
        
        # Calculate total sections with data and percentage of completion
        sections_with_data = sum(1 for count in section_counts.values() if count > 0)
        completion_percentage = (sections_with_data / len(IMPORTANT_SECTIONS)) * 100
        
        # Determine if the city has enough data (at least 3-4 key sections)
        has_enough_data = sections_with_data >= 3
        
        city_analysis[city] = {
            'section_counts': section_counts,
            'sections_with_data': sections_with_data,
            'completion_percentage': completion_percentage,
            'has_enough_data': has_enough_data
        }
    
    return city_analysis


def extract_structured_data(city_name: str, city_items: List[Dict]) -> Dict[str, List[Dict]]:
    """Extract and structure data for a city."""
    structured_data = {section: [] for section in IMPORTANT_SECTIONS}
    
    for item in city_items:
        item_type = item.get('type')
        if item_type in IMPORTANT_SECTIONS:
            # Extract the relevant fields we want to keep
            structured_item = {
                'title': item.get('title', ''),
                'price': item.get('price', ''),
                'description': item.get('description', '')
            }
            
            # Only add items with at least title or description
            if structured_item['title'] or structured_item['description']:
                structured_data[item_type].append(structured_item)
    
    return structured_data


def sample_items_for_response(structured_data: Dict[str, List[Dict]], min_items: int = 3, max_items: int = 5) -> Dict[str, List[Dict]]:
    """Randomly sample a subset of items from each section."""
    sampled_data = {}
    
    for section, items in structured_data.items():
        if items:
            # Calculate how many items to include (between min_items and max_items, but not more than available)
            num_items = min(max(min_items, random.randint(min_items, max_items)), len(items))
            
            # Randomly sample without replacement
            sampled_data[section] = random.sample(items, num_items)
        else:
            sampled_data[section] = []
    
    return sampled_data


def create_prompts_responses(city_analysis: Dict[str, Dict], city_data: Dict[str, List[Dict]], prompts_per_city: int = 3) -> List[Dict[str, str]]:
    """Create structured prompts and responses for cities with enough data."""
    data = []
    valid_cities = [city for city, analysis in city_analysis.items() if analysis['has_enough_data']]
    print(f"Found {len(valid_cities)} cities with sufficient data.")
    
    for city in valid_cities:
        # Extract structured data for this city
        structured_data = extract_structured_data(city, city_data[city])
        
        # Generate multiple prompts for each city
        for _ in range(prompts_per_city):
            # Sample a random subset of items for each section to create variety
            sampled_data = sample_items_for_response(structured_data)
            
            # Format the structured data as a string for the response
            response_parts = []
            for section in IMPORTANT_SECTIONS:
                if sampled_data[section]:
                    # Use the friendly section name from the mapping
                    section_part = f"[{SECTION_MAPPING[section]}]\n"
                    for item in sampled_data[section]:
                        if item['title']:
                            section_part += f"- {item['title']}"
                            if item['price']:
                                section_part += f" ({item['price']})"
                            section_part += "\n"
                            if item['description']:
                                section_part += f"  {item['description']}\n"
                    response_parts.append(section_part)
            
            structured_text = "\n".join(response_parts).strip()
            
            if structured_text:
                from_city = random.choice(FROM_CITIES)
                days = random.choice([3, 4, 5, 6, 7])
                prompt_template = random.choice(PROMPT_VARIATIONS)
                prompt = prompt_template.format(from_city=from_city, to_city=city, days=days)

                data.append({
                    "title": city,
                    "prompt": prompt,
                    "response": "[Context]\n" + structured_text
                })

    print(f"\nTotal structured prompts and responses created: {len(data)}")
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
    report_data = []
    
    # Create column names using both original and friendly names for clarity
    column_names = {
        'see': f'attractions_count ({SECTION_MAPPING["see"]})',
        'do': f'activities_count ({SECTION_MAPPING["do"]})',
        'buy': f'shops_count ({SECTION_MAPPING["buy"]})',
        'eat': f'restaurants_count ({SECTION_MAPPING["eat"]})',
        'drink': f'bars_count ({SECTION_MAPPING["drink"]})',
        'sleep': f'hotels_count ({SECTION_MAPPING["sleep"]})'
    }
    
    for city, analysis in city_analysis.items():
        city_report = {
            'city': city,
            'sections_with_data': analysis['sections_with_data'],
            'completion_percentage': analysis['completion_percentage'],
            'has_enough_data': analysis['has_enough_data']
        }
        
        # Add section counts with friendly column names
        for section in IMPORTANT_SECTIONS:
            column_name = section + '_count'  # Keep original column names for code compatibility
            city_report[column_name] = analysis['section_counts'][section]
        
        report_data.append(city_report)
    
    df = pd.DataFrame(report_data)
    df = df.sort_values(by=['has_enough_data', 'completion_percentage'], ascending=[False, False])
    
    # Rename columns for the CSV output
    column_mapping = {f'{section}_count': column_names[section] for section in IMPORTANT_SECTIONS}
    df_output = df.rename(columns=column_mapping)
    df_output.to_csv(output_file, index=False)
    print(f"City analysis report saved to '{output_file}'.")
    
    # Print summary statistics
    valid_cities = df[df['has_enough_data'] == True]
    print(f"\nSummary Statistics:")
    print(f"Total cities: {len(df)}")
    print(f"Cities with enough data: {len(valid_cities)} ({len(valid_cities)/len(df)*100:.1f}%)")
    print(f"Average sections with data: {df['sections_with_data'].mean():.1f} of {len(IMPORTANT_SECTIONS)}")
    
    # Show section distribution with friendly names
    print("\nSection distribution (cities having at least one item):")
    for section in IMPORTANT_SECTIONS:
        count = len(df[df[f'{section}_count'] > 0])
        print(f"  {SECTION_MAPPING[section]}: {count} cities ({count/len(df)*100:.1f}%)")


def main():
    city_list_path = "city_list.txt"
    city_data_csv_path = "wikivoyage-listings-en.csv"  # Your CSV file with the structured data
    output_csv_path = "prompt_wikidata.csv"
    
    if not os.path.exists(city_list_path):
        print(f"Error: '{city_list_path}' does not exist.")
        return
    if not os.path.exists(city_data_csv_path):
        print(f"Error: '{city_data_csv_path}' does not exist.")
        return
    
    # Load city list from text file
    cities = load_city_list(city_list_path)
    
    # Load structured city data from CSV
    city_data = load_city_data_from_csv(city_data_csv_path)
    
    # Analyze city data to see which cities have enough content
    city_analysis = analyze_city_data(city_data)
    
    # Generate a report on city data quality
    generate_city_report(city_analysis)
    
    # Create structured prompts and responses for valid cities
    prompts_responses = create_prompts_responses(city_analysis, city_data)
    
    # Save results to CSV
    save_to_csv(prompts_responses, output_csv_path)
    
    try:
        df_loaded = pd.read_csv(output_csv_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"CSV successfully loaded back into Pandas. Shape: {df_loaded.shape}")
    except Exception as e:
        print(f"Error loading CSV into Pandas: {e}")


if __name__ == "__main__":
    main()