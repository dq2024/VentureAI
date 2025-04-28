# VentureAI

This project is focused on building a high-quality travel dataset by leveraging *Wikivoyage* and *OpenAI* models.  
We are preparing training data for **fine-tuning** large language models (LLMs) like Mistral or Falcon 7B.

Below is a step-by-step breakdown of the data pipeline so far:

---

## 1. Raw Data Extraction
- **Source:** Downloaded the **Wikivoyage** XML dump.
- **Script:** `xml_to_json.py`
- **Goal:** Convert the massive Wikivoyage XML into a more manageable **JSON** format.
- **Output:** A JSON file containing all Wikivoyage pages.

---

## 2. Cleaning the Wikivoyage Data
- **Problem:** The raw Wikivoyage pages contain unwanted metadata, images, links, and HTML artifacts.
- **Solution:**
  - Identify important sections: `Understand`, `Get in`, `Get around`, `See`, `Do`, `Buy`, `Eat`, `Drink`, `Sleep`, `Go next`.
  - Remove junk patterns (regexes to strip noise).
  - Retain only the **relevant travel content**.
- **Result:** A clean, structured version of each city page, ready for prompting.

---

## 3. Prompt Generation
- **Task:** Automatically generate prompts that ask for travel itineraries.
- **Logic:**
  - Randomly select `from_city` (e.g., London, Toronto).
  - Randomly select a stay duration (5–10 days).
  - Randomize prompt templates to create diversity.
  - Generate **2–5 prompts** per destination city to increase data coverage.
- **Result:** 
  - A CSV file where each row contains:
    - `[Prompt]`
    - `[Cleaned Wikivoyage Data]`

---

## 4. Synthetic Response Generation via OpenAI
- **Goal:** Turn the prompts + Wikivoyage data into synthetic **travel itineraries**.
- **Model:** Using `gpt-4o-mini` or similar models.
- **Challenge:** 
  - **Rate limits**: 200,000 tokens per minute maximum.
  - **429 Errors**: Caused if too many tokens are sent too quickly.
- **Solution:**
  - Added controlled batching and sleep intervals.
  - Automatically **retry** if rate limit errors occur.
  - Parallelized generation with **safe limits** (2–4 concurrent requests).

---

## 5. Saving Final Results
- The generated prompt + response pairs are saved to a CSV file.
- Each response is tightly aligned to the provided Wikivoyage data (no hallucinations encouraged).

---

# Current Status
- Wikivoyage parsed and cleaned.
- Prompts randomized and generated.
- 500+ prompt + data pairs ready.
- OpenAI API integration functional.
- Dataset generation in progress (handling token limits safely).

---

# Next Steps
- [ ] Finish generating all synthetic itineraries (~500 responses).
- [ ] Clean/finalize the dataset (remove failures / empty responses).
- [ ] Format the final dataset for fine-tuning Mistral or Falcon models.
- [ ] Train or continue synthetic data generation if needed.


