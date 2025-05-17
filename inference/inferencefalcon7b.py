from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os
import torch
from transformers import BitsAndBytesConfig  # Import only if using BitsAndBytes for 8-bit inference
from peft import PeftModel  # Import PeftModel for loading LoRA adapters


# Load environment variables from .env file
load_dotenv()

# Determine the device to run the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {device}")

# Define the directory where the trained model and tokenizer are saved
model_dir = "../finetune/trained_falcon7b"  # Updated to match the training script's output directory

# Check if the model directory exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Trained model directory not found at {model_dir}")

print("Loading tokenizer...")
# Load the tokenizer from the trained model directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

print("Loading base model...")
# Load the base model without 8-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b",
    quantization_config=bnb_config,
    #torch_dtype=torch.float16,  # Changed from bfloat16 to float16 for broader GPU compatibility
    device_map="auto"           # Automatically maps the model to available devices
)

print("Loading LoRA adapters...")
# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, model_dir)

# Set the model to evaluation mode to disable dropout and other training-specific layers
model.eval()

# Define a function for generating responses with enhanced parameters
def generate_response(prompt, max_length=2000, temperature=0.2, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=6):
    """
    Generates a response from the model based on the input prompt.

    Args:
        prompt (str): The input text prompt.
        max_length (int): Maximum length of the generated response.
        temperature (float): Sampling temperature; higher values mean more random outputs.
        top_p (float): Nucleus sampling probability.
        repetition_penalty (float): Penalizes repeated tokens to reduce redundancy.
        no_repeat_ngram_size (int): Prevents the model from repeating n-grams of this size.

    Returns:
        str: The generated response text.
    """
    # Tokenize the input prompt
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # Ensure that input does not exceed model's max context length
    )
    # input_ids = encoding['input_ids'].to(model.device)
    # attention_mask = encoding['attention_mask'].to(model.device)
    input_ids = encoding.input_ids.to(model.device)
    attention_mask = encoding.attention_mask.to(model.device)

    # Generate the response using the model's generate method with specified parameters
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,                     # Increased max_length for longer responses
            temperature=temperature,                   # Controls randomness
            top_p=top_p,                               # Nucleus sampling
            repetition_penalty=repetition_penalty,     # Penalizes repetition
            no_repeat_ngram_size=no_repeat_ngram_size, # Prevents repeating n-grams
            pad_token_id=tokenizer.pad_token_id,       # Use the pad token
            eos_token_id=tokenizer.eos_token_id,       # Stop generation at EOS token
            do_sample=True,                            # Enable sampling for variability
            early_stopping=True                        # Stop generation when EOS is reached
        )

    # Decode the generated tokens back into text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Example usage of the inference script
if __name__ == "__main__":
    print("\nModel loaded. Ready for inference.\n")

    #while True:
        # prompt = input("Enter a prompt (or type 'exit' to quit): ")
    # city_from = input("Enter the city you are traveling FROM (or type 'exit' to quit): ")
    # city_to = input("Enter the city you are traveling TO (or type 'exit' to quit): ")
    # num_days = input("Enter the number of days you will be traveling as a single number (or type 'exit' to quit): ")

    #prompt = f" I would like to travel from {city_from} to {city_to} for {num_days} days. Give me a trip plan that focuses on restaurants."
    prompt = f''' I would like to travel from New York to Antalya for 7 days. Please create a day by day itinerary. The context below is about Antalya. Please use it in your response. \n 
    [Context]
[Attractions]
- Old Dam and Dilberler Seki Road
- Old Town and the Clock Tower
- Central Park
- Ethnographic Museum of Adana
  Closed for renovation.

[Restaurants]
- L'estaquade
  Situated on the Right Bank of the river (it is the building that just out over the water on stilts) you will get a great view of the Bordeaux waterfront at the same time as a delicious meal.
- Mado's
  Sweet shop. Higher class than your regular street vendor and a little pricey, but delicious food. You should eat special Turkish ice cream.
- Cafe Ora
  Has a bar on second floor. You can have a ''bici bici'' (traditional Adana sweet) for TRY3 there.
- Adana Kebab
  Delicious and famous Adana kebap and mezzes, usually accompanied by salgam (turnip) juice and/or raki (traditional Turkish alcoholic drink).

[Bars]
- Havana Club
  The grand hotel is a must-see in Abu Dhabi and the actual club is nicely decorated, comfortable, has great service, a balcony overlooking the hotel grounds, and provides a fun time with great music and very colorful laser shows.
- The Yacht Club
  offers a gorgeous view of the sunset over the marina if you sit outside. Inside has a very modern, minimalistic feel. The cocktails are delicious, but expensive.
- Lebinese Flower
  Great food and nice atmosphere.

[Hotels]
- Hilton Adana (€100+)
  The tallest building in town. Looks quite strange in a not that big town to have such a huge Hilton Hotel.
- Garajlar Hotel (TRY15)
  You will have to pay extra for the bath (TRY5).
- Konya Hotel (TRY20 non-air-con, TRY30 for rooms with air-con)
  If you insist, they can give you a cheaper, small room. The rooms are clean but there is no heater.
    '''

    
    # Generate the response based on the input prompt
    response = generate_response(prompt, max_length=2000)
    print(f"\nResponse:\n{response}\n")
