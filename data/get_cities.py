'''
READ FROM wikivoyage-listings-en/csv and generate  text file with the titles (cities)

'''
import csv

INPUT_CSV = "wikivoyage-listings-en.csv"
OUTPUT_TXT = "city_list.txt"

def extract_unique_cities(input_csv: str, output_txt: str):
    cities = set()
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            city = row.get("article", "").strip()
            if city:
                cities.add(city)

    with open(output_txt, "w", encoding="utf-8") as txtfile:
        for city in sorted(cities):
            txtfile.write(city + "\n")
    print(f"Wrote {len(cities)} unique cities to '{output_txt}'.")

if __name__ == "__main__":
    extract_unique_cities(INPUT_CSV, OUTPUT_TXT)
