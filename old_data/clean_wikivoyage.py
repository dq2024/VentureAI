# import json
# import re

# # Load the JSON data from the file with UTF-8 encoding
# with open("wikivoyage.json", "r", encoding="utf-8") as file:
#     data = json.load(file)

# # Function to recursively remove formatting, process both {{}} and [[]] blocks, and remove specified keys
# def remove_formatting(content):
#     if isinstance(content, str):
#         # Replace all instances of &quot; with "
#         content = content.replace("&quot;", '"')

#         # Remove formatting in {{}} while keeping the description, and remove "caption="
#         # content = re.sub(
#         #     r"\{\{.*?\|(?:caption=)?([^{}]*)\}\}", r"\1", content, flags=re.DOTALL
#         # )

#         # Completely remove text inside {{}} including the braces
#         content = re.sub(r"\{\{.*?\}\}", "", content, flags=re.DOTALL)

#         # Process [[]] blocks
#         def replace_link(match):
#             text = match.group(1)
#             # Check if "File:" is in the text
#             if "File:" in text:
#                 return ""  # Remove the entire block
#             else:
#                 return text  # Keep the text, but remove the brackets

#         # Substitute all [[]] occurrences based on the condition
#         content = re.sub(r"\[\[(.*?)\]\]", replace_link, content)
#         return content

#     elif isinstance(content, dict):
#         # Remove specified keys if present, otherwise recursively apply to dictionary values
#         keys_to_remove = {
#             "contributor",
#             "sha1",
#             "model",
#             "format",
#             "@bytes",
#             "@sha1",
#             "@xml:space",
#             "timestamp",
#             "comment",
#             "minor",
#             "id",  # This may potentially be useful
#             "origin",  # duplicate of id?
#             "ns",  # No idea what this is
#             "parentid",  # Redirects to this maybe?
#             "siteinfo",
#         }
#         return {
#             key: remove_formatting(value)
#             for key, value in content.items()
#             if key not in keys_to_remove
#         }

#     elif isinstance(content, list):
#         # Recursively apply to each item in the list
#         return [remove_formatting(item) for item in content]

#     return content

# # Clean the JSON data
# cleaned_data = remove_formatting(data)

# # Optionally, save cleaned data back to a new file
# with open("cleaned_wikivoyage.json", "w", encoding="utf-8") as file:
#     json.dump(cleaned_data, file, ensure_ascii=False, indent=4)

# print(
#     "Text within double braces {{}} and links in [[]] have been processed, &quot; replaced, unncessary keys have been removed, and cleaned JSON saved to 'cleaned_wikivoyage.json'."
# )
import json
import re

# Load the JSON data from the file with UTF-8 encoding
with open("wikivoyage.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Function to recursively remove formatting, process both {{}} and [[]] blocks, and remove specified keys
def remove_formatting(content):
    if isinstance(content, str):
        content = content.replace("&quot;", '"')
        # Remove all {{...}} blocks
        content = re.sub(r"\{\{.*?\}\}", "", content, flags=re.DOTALL)

        # Remove [[File:...]] and similar, keep regular links
        def replace_link(match):
            text = match.group(1)
            if "File:" in text:
                return ""
            else:
                return text

        content = re.sub(r"\[\[(.*?)\]\]", replace_link, content)
        return content

    elif isinstance(content, dict):
        keys_to_remove = {
            "contributor",
            "sha1",
            "model",
            "format",
            "@bytes",
            "@sha1",
            "@xml:space",
            "timestamp",
            "comment",
            "minor",
            "id",
            "origin",
            "ns",
            "parentid",
            "siteinfo",
        }
        return {
            key: remove_formatting(value)
            for key, value in content.items()
            if key not in keys_to_remove
        }

    elif isinstance(content, list):
        return [remove_formatting(item) for item in content]

    return content

# Function to check if a page is "good enough" (not a redirect, and has useful content)
def is_valid_city(page):
    # Check if page is a redirect
    if "redirect" in page:
        return False

    title = page.get("title", "")

    # Exclude administrative/discussion pages
    forbidden_prefixes = ("Wikivoyage:", "User:", "Talk:", "File:", "Template:", "Category:", "Project:")
    if title.startswith(forbidden_prefixes):
        return False

    # Check text exists and is reasonably long
    text = page.get("revision", {}).get("text", {}).get("#text", "")
    text = text.strip()
    if len(text) < 300:
        return False

    # (Optional) Check if text includes important travel guide sections
    # important_sections = ["==Get in==", "==See==", "==Eat==", "==Sleep=="]
    # if not any(section in text for section in important_sections):
    #     return False

    return True

# Clean the JSON data
cleaned_data = remove_formatting(data)

# Filter out bad pages
if "mediawiki" in cleaned_data and "page" in cleaned_data["mediawiki"]:
    pages = cleaned_data["mediawiki"]["page"]
    filtered_pages = [page for page in pages if is_valid_city(page)]
    cleaned_data["mediawiki"]["page"] = filtered_pages

# Optionally, save cleaned data back to a new file
with open("cleaned_wikivoyage.json", "w", encoding="utf-8") as file:
    json.dump(cleaned_data, file, ensure_ascii=False, indent=4)

print(
    "Formatting cleaned, bad pages filtered, and saved to 'cleaned_wikivoyage.json'."
)
