import pandas as pd

# Load both CSVs
wikidata_df = pd.read_csv("prompt_wikidata.csv")
results_df = pd.read_csv("results.csv")

# Ensure clean formatting
wikidata_df['prompt'] = wikidata_df['prompt'].str.strip()
results_df['prompt'] = results_df['prompt'].str.strip()

# Create a dictionary for fast lookup: prompt -> response
prompt_to_response = dict(zip(wikidata_df['prompt'], wikidata_df['response']))

# Append the matching response to the prompt in results_df
def append_response(row):
    response = prompt_to_response.get(row['prompt'], "")
    return row['prompt'] + "\n\n" + response if response else row['prompt']

# Create new column with modified prompt
results_df['prompt'] = results_df.apply(append_response, axis=1)

# Save the merged CSV
results_df.to_csv("results_merged.csv", index=False)
print("Merged file saved as 'results_merged.csv'")
