# Required libraries (install using pip: pip install spacy sentence-transformers numpy torch)
# Download spacy model: python -m spacy download en_core_web_sm
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import datetime # For handling date values
import torch # Import torch to check for GPU availability

# --- Configuration & Hypothetical Schema ---

# Load a spaCy model for basic NLP (NER, POS tagging)
# Use a small model for demonstration; larger models might be better
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Load a sentence transformer model for calculating similarity
# Using a pre-trained model efficient for sentence embeddings
# Check if GPU is available and set device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Ensure 'sentence-transformers', 'torch', and 'transformers' are installed and you have internet access.")
    similarity_model = None # Set to None to avoid errors later if loading fails

# --- Hypothetical Database Schema Information ---
# In a real system, this would be loaded dynamically or from configuration
SCHEMA = {
    "tables": {
        "Sales": ["SaleID", "ProductID", "CustomerID", "Amount", "SaleDate", "Region"],
        "Products": ["ProductID", "ProductName", "Category", "Price"],
        "Customers": ["CustomerID", "CustomerName", "City", "Country"]
    },
    "columns": {
        # Map natural language terms to actual column names
        "sale": "Sales.Amount",
        "sales": "Sales.Amount",
        "amount": "Sales.Amount",
        "product": "Products.ProductName",
        "products": "Products.ProductName",
        "customer": "Customers.CustomerName",
        "customers": "Customers.CustomerName",
        "city": "Customers.City",
        "region": "Sales.Region",
        "date": "Sales.SaleDate",
        "price": "Products.Price",
        "category": "Products.Category",
    },
    "aggregations": {
        "total": "SUM",
        "sum": "SUM",
        "average": "AVG",
        "avg": "AVG",
        "count": "COUNT",
        "number": "COUNT",
        "how many": "COUNT",
        "max": "MAX",
        "maximum": "MAX",
        "highest": "MAX",
        "min": "MIN",
        "minimum": "MIN",
        "lowest": "MIN",
    },
    # Basic join paths (simplified)
    "joins": {
        ("Sales", "Products"): "Sales.ProductID = Products.ProductID",
        ("Sales", "Customers"): "Sales.CustomerID = Customers.CustomerID",
    }
}

# --- Supervised Dataset (Natural Language -> SQL Structure/Template) ---
# This is the core for the similarity approach.
# Each entry maps a natural language pattern to a parameterized SQL structure.
# Placeholders like {COL}, {VAL}, {AGG}, {TBL}, {COND_COL}, {COND_VAL} will be filled.
SUPERVISED_DATASET = [
    {
        "nl": "Show total sales",
        "sql_template": "SELECT {AGG}({COL}) FROM {TBL};",
        "params": {"AGG": "SUM", "COL": "Sales.Amount", "TBL": "Sales"}
    },
    {
        "nl": "How many products are there",
        "sql_template": "SELECT COUNT({COL}) FROM {TBL};",
        "params": {"COL": "Products.ProductID", "TBL": "Products"}
    },
    {
        "nl": "List customers in a specific city",
        "sql_template": "SELECT CustomerName FROM Customers WHERE {COND_COL} = '{COND_VAL}';",
        "params": {"COND_COL": "Customers.City"} # COND_VAL needs extraction
    },
    {
        "nl": "What is the total sales for a specific product",
        "sql_template": "SELECT SUM(s.Amount) FROM Sales s JOIN Products p ON s.ProductID = p.ProductID WHERE p.{COND_COL} = '{COND_VAL}';",
        "params": {"COND_COL": "Products.ProductName"} # COND_VAL needs extraction
    },
    {
        "nl": "Show sales amount and product name",
        "sql_template": "SELECT s.Amount, p.ProductName FROM Sales s JOIN Products p ON s.ProductID = p.ProductID;",
        "params": {}
    },
    {
        "nl": "Count customers per city",
        "sql_template": "SELECT {GROUP_COL}, COUNT({COUNT_COL}) FROM {TBL} GROUP BY {GROUP_COL};",
        "params": {"GROUP_COL": "Customers.City", "COUNT_COL": "Customers.CustomerID", "TBL": "Customers"}
    },
    # Add many more examples covering different query types (joins, group by, order by, date ranges etc.)
]

# Pre-compute embeddings for the dataset NL queries for efficiency
if similarity_model:
    dataset_nl_queries = [item["nl"] for item in SUPERVISED_DATASET]
    # Ensure embeddings are computed on the correct device
    dataset_embeddings = similarity_model.encode(dataset_nl_queries, convert_to_tensor=True, device=device)
else:
    dataset_embeddings = None
    print("Warning: Sentence similarity model not loaded. Similarity matching will be skipped.")

# --- Step 1: Find relevant data points (Entity Extraction) ---

def extract_entities(text):
    """
    Extracts potential database entities (columns, values, aggregations)
    using spaCy NER, POS tagging, and keyword matching.
    This is a simplified version.
    """
    doc = nlp(text.lower())
    entities = {
        "columns": [],
        "values": [],
        "aggregations": [],
        "tables": set(), # Track potential tables involved
        "conditions": [] # Store potential (column, operator, value) tuples
    }

    # Basic Keyword Matching for Aggregations
    for token in doc:
        if token.lemma_ in SCHEMA["aggregations"]:
            entities["aggregations"].append(SCHEMA["aggregations"][token.lemma_])

    # NER for potential values (like GPE for cities/countries, DATE for dates)
    for ent in doc.ents:
        # Simple date handling (needs more robust parsing)
        if ent.label_ == "DATE":
             # Placeholder: Real implementation needs date parsing library (e.g., dateutil)
             # to handle "last month", "yesterday", "Q2 2024" etc.
             entities["values"].append(f"'{ent.text}'") # Assume string literal for now
        elif ent.label_ in ["GPE", "ORG", "PERSON", "MONEY", "QUANTITY", "PRODUCT"]: # Geographical, Organization, Person, Money, Quantity, Product
             # Check if it's a number to avoid extra quotes
             if ent.label_ in ["MONEY", "QUANTITY"] and ent.text.replace('.', '', 1).isdigit():
                 entities["values"].append(ent.text)
             else:
                 entities["values"].append(f"'{ent.text}'") # Assume string literal otherwise
        # Add more entity types as needed

    # POS tagging and schema matching for Columns
    potential_columns = []
    for token in doc:
        # Look for nouns that might be columns or table references
        if token.pos_ == "NOUN" and token.lemma_ in SCHEMA["columns"]:
            col_name = SCHEMA["columns"][token.lemma_]
            entities["columns"].append(col_name)
            table_name = col_name.split('.')[0] # Infer table from column
            entities["tables"].add(table_name)
            potential_columns.append((token.lemma_, col_name)) # Store lemma and full name

        # Very basic condition detection (e.g., "sales > 500", "city is London")
        # Look for columns followed by ADP/VERB and then a value entity
        if token.lemma_ in SCHEMA["columns"]:
            col_to_filter = SCHEMA["columns"][token.lemma_]
            op = "=" # Default operator
            val = None

            # Check for comparison words before the column (e.g., "more than 500 sales")
            if token.i > 0:
                 prev_token = doc[token.i-1]
                 if prev_token.lemma_ in ['>', 'more', 'greater', 'over', 'above']: op = ">"
                 if prev_token.lemma_ in ['<', 'less', 'fewer', 'under', 'below']: op = "<"
                 # Check if the preceding token is a number/value entity related to this column
                 if prev_token.ent_type_ in ["MONEY", "QUANTITY", "CARDINAL", "PERCENT"] or prev_token.pos_ == "NUM":
                     val = prev_token.text # Keep numbers unquoted

            # Check for values following the column (e.g., "sales greater than 500", "city is London")
            if val is None: # Only proceed if value wasn't found before the column
                for child in token.children:
                     # Check for operators like 'is', 'equals', '>', '<'
                     if child.dep_ in ["prep", "attr", "dobj", "acomp"]: # Preposition, attribute, direct object, adjectival complement
                         # Look for value entities linked by the child or siblings of the child
                         for potential_val_token in list(child.children) + list(child.rights):
                             if potential_val_token.ent_type_ in ["GPE", "ORG", "PERSON", "DATE", "MONEY", "QUANTITY", "PRODUCT"] or potential_val_token.pos_ in ["PROPN", "NUM"]:
                                 val = f"'{potential_val_token.text}'" # Assume string literal initially
                                 if potential_val_token.pos_ == "NUM" or potential_val_token.ent_type_ in ["MONEY", "QUANTITY"]:
                                     val = potential_val_token.text # Use number directly
                                 # Refine operator based on connecting words (child or token head)
                                 if child.lemma_ in ['>', 'more', 'greater', 'over', 'above']: op = ">"
                                 if child.lemma_ in ['<', 'less', 'fewer', 'under', 'below']: op = "<"
                                 if child.lemma_ in ['in', 'on', 'at'] and potential_val_token.ent_type_ == "DATE": op = "=" # Date equality
                                 break # Found value for this child
                         if val: break # Found value for this column

            # If a value was associated, add the condition
            if val is not None:
                 entities["conditions"].append((col_to_filter, op, val))
                 # Try to remove the value if it was also captured generally
                 if val in entities["values"]:
                     try: entities["values"].remove(val)
                     except ValueError: pass
                 elif val.strip("'") in entities["values"]: # Check unquoted version too
                     try: entities["values"].remove(val.strip("'"))
                     except ValueError: pass


    # If no specific columns mentioned, but aggregation is, assume a default (e.g., COUNT(*))
    if not entities["columns"] and entities["aggregations"]:
         if "COUNT" in entities["aggregations"]:
             entities["columns"].append("*") # Default to COUNT(*)
         # Add defaults for SUM, AVG if needed based on context/schema (e.g., default numeric column)

    # If no tables explicitly found via columns, try to infer from keywords
    if not entities["tables"]:
         for token in doc:
             # Check against lowercase table names
             table_keys_lower = {name.lower(): name for name in SCHEMA["tables"]}
             if token.lemma_ in table_keys_lower:
                 entities["tables"].add(table_keys_lower[token.lemma_])


    # Remove duplicates
    entities["columns"] = list(set(entities["columns"]))
    entities["aggregations"] = list(set(entities["aggregations"]))

    print(f"--- Extracted Entities --- \n{entities}\n")
    return entities


# --- Step 2: Find relation between data points (Similarity & Template Matching) ---

def find_best_template(user_query, entities):
    """
    Finds the most similar query template from the supervised dataset
    and attempts to fill its parameters using extracted entities.
    Moves tensors to CPU before NumPy operations.
    """
    if not similarity_model or dataset_embeddings is None:
        print("Similarity model not available. Skipping template matching.")
        # Fallback: Could try rule-based matching here
        return None, {} # Return None template, empty filled params

    print(f"--- Finding Similar Template for: '{user_query}' ---")
    # Ensure user query embedding is done on the correct device
    user_embedding = similarity_model.encode(user_query, convert_to_tensor=True, device=device)

    # Calculate cosine similarities
    # The result might be on GPU if device='cuda'
    cosine_scores = util.pytorch_cos_sim(user_embedding, dataset_embeddings)[0]

    # --- FIX: Move tensor to CPU before using NumPy ---
    cosine_scores_cpu = cosine_scores.cpu()

    # Find the index of the highest score using the CPU tensor
    best_match_idx = np.argmax(cosine_scores_cpu)
    best_score = cosine_scores_cpu[best_match_idx] # Get score from CPU tensor too

    print(f"Best match: '{SUPERVISED_DATASET[best_match_idx]['nl']}' (Score: {best_score:.4f})")

    # Use a threshold - if the similarity is too low, it might be a bad match
    SIMILARITY_THRESHOLD = 0.5 # Adjust as needed
    if best_score < SIMILARITY_THRESHOLD:
        print("Similarity score below threshold. No suitable template found.")
        return None, {}

    matched_template_info = SUPERVISED_DATASET[best_match_idx]
    sql_template = matched_template_info["sql_template"]
    required_params = matched_template_info.get("params", {}) # Default params from template

    # --- Parameter Filling Logic (Simplified) ---
    filled_params = required_params.copy()

    # Fill aggregation if needed and found
    if "{AGG}" in sql_template and entities["aggregations"]:
        filled_params["AGG"] = entities["aggregations"][0] # Take the first one found

    # Fill primary column if needed and found
    if "{COL}" in sql_template and entities["columns"]:
         # Prioritize non-* columns if available
         non_star_cols = [c for c in entities["columns"] if c != '*']
         if non_star_cols:
             # Try to pick a column relevant to the query context if possible
             # Simple: just take the first non-star column found
             filled_params["COL"] = non_star_cols[0]
         elif '*' in entities["columns"]:
             filled_params["COL"] = '*'
         # If only '*' was found but template needs a specific column, this might fail later

    # Fill table if needed (can be complex if multiple tables/joins)
    if "{TBL}" in sql_template:
        if "TBL" not in filled_params: # Only fill if not predefined in template params
            if entities["tables"]:
                # Simplistic: pick the first table found or the one related to the main column
                main_col = filled_params.get("COL", entities["columns"][0] if entities["columns"] else None)
                if main_col and '.' in main_col and main_col != '*':
                    filled_params["TBL"] = main_col.split('.')[0]
                else:
                    # If multiple tables, might need join logic later or smarter selection
                    filled_params["TBL"] = list(entities["tables"])[0]
            else:
                print("Warning: Could not determine table for template.")
                # Maybe try to infer from COL if it has Table.Column format
                if "COL" in filled_params and '.' in filled_params["COL"] and filled_params["COL"] != '*':
                     filled_params["TBL"] = filled_params["COL"].split('.')[0]


    # Fill condition value if needed
    # This needs refinement: Match condition column from template to extracted condition
    if "{COND_VAL}" in sql_template:
        cond_col_template = filled_params.get("COND_COL") # Get the expected condition column from template
        found_condition = False
        if entities["conditions"]:
            for cond_col, op, cond_val in entities["conditions"]:
                # Check if the extracted condition column matches the template's expectation
                if cond_col_template and cond_col == cond_col_template:
                    filled_params["COND_VAL"] = cond_val # Use value from matched condition
                    # TODO: Incorporate the operator 'op' into the SQL template if needed
                    found_condition = True
                    break
            # Fallback: If no specific match, use the first condition's value (less reliable)
            if not found_condition:
                 filled_params["COND_VAL"] = entities["conditions"][0][2]
        elif entities["values"]:
             # Fallback if no structured conditions found, use first generic value
             filled_params["COND_VAL"] = entities["values"][0]


    # Fill condition column if needed
    if "{COND_COL}" in sql_template and "COND_COL" not in filled_params:
         # Try to find a condition matching the template's expected column type if possible
         # Or default to the first condition found
         if entities["conditions"]:
             filled_params["COND_COL"] = entities["conditions"][0][0] # Take col from first condition
             # If COND_VAL wasn't filled above, fill it now from this condition
             if "{COND_VAL}" in sql_template and "COND_VAL" not in filled_params:
                 filled_params["COND_VAL"] = entities["conditions"][0][2]
         elif entities["columns"]:
             # Fallback: use a column that seems filterable (not PK, not aggregated)
             filterable_cols = [c for c in entities["columns"] if c != '*' and "ID" not in c.upper()]
             if filterable_cols:
                 filled_params["COND_COL"] = filterable_cols[0]


    # Fill GROUP BY column
    if "{GROUP_COL}" in sql_template and "GROUP_COL" not in filled_params:
         # Try to infer from keywords like "per", "by" or use a suitable dimension column
         # Look for non-numeric, non-ID columns extracted
         group_cols = [c for c in entities["columns"] if c != '*' and "Amount" not in c and "Price" not in c and "ID" not in c.upper()]
         if group_cols:
             filled_params["GROUP_COL"] = group_cols[0]
         elif entities["columns"] and entities["columns"][0] != '*':
             # Fallback to the first non-* column if any
              non_star_cols = [c for c in entities["columns"] if c != '*']
              if non_star_cols:
                  filled_params["GROUP_COL"] = non_star_cols[0]


    # Fill COUNT column for GROUP BY
    if "{COUNT_COL}" in sql_template and "COUNT_COL" not in filled_params:
        # Often the primary key of the table being grouped, or '*'
        tbl = filled_params.get("TBL")
        count_col_found = False
        if tbl :
            table_cols = SCHEMA["tables"].get(tbl, [])
            # Look for a primary key convention (e.g., TableNameID)
            potential_pk = f"{tbl}ID"
            if potential_pk in table_cols:
                filled_params["COUNT_COL"] = f"{tbl}.{potential_pk}" # Use qualified name if possible
                count_col_found = True
        if not count_col_found:
             filled_params["COUNT_COL"] = "*" # Fallback to COUNT(*)


    print(f"Filled parameters: {filled_params}")
    return sql_template, filled_params


# --- Step 3: Make SQL query from the structure ---

def generate_sql(sql_template, params):
    """
    Generates the final SQL query string by filling the template
    with the provided parameters. Handles basic quoting.
    """
    if not sql_template:
        return "-- Could not generate SQL: No suitable template found."

    query = sql_template
    try:
        # Use regex to find placeholders to avoid partial replacements
        placeholders = re.findall(r"(\{.*?\})", query)
        filled_placeholders = set()

        for placeholder in placeholders:
            key = placeholder.strip("{}") # Get key name (e.g., "COND_VAL")
            if key in params:
                value = params[key]
                # Ensure values used in WHERE clauses are properly quoted if they are strings
                # Check if placeholder is likely within a WHERE clause context (heuristic)
                is_condition_value = "WHERE" in query.split(placeholder)[0].upper() and key == "COND_VAL"

                if is_condition_value and isinstance(value, str) and not value.startswith("'"):
                     # Add quotes only if it doesn't look like a number and isn't already quoted
                     if not value.replace('.', '', 1).isdigit():
                         query = query.replace(placeholder, f"'{value}'", 1) # Replace only once per iteration
                     else:
                         query = query.replace(placeholder, str(value), 1)
                else:
                    # Replace other placeholders directly
                    query = query.replace(placeholder, str(value), 1)
                filled_placeholders.add(placeholder)
            # else: # Placeholder key not found in params - leave it for the check below

        # Check if any placeholders remain (indicates missing info)
        # Re-run findall on the potentially modified query string
        remaining_placeholders = re.findall(r"(\{.*?\})", query)
        # Filter out any that might have been part of a value itself (unlikely but possible)
        final_remaining = [p for p in remaining_placeholders if p not in filled_placeholders and p.strip("{}") not in params]

        if final_remaining:
             print(f"Warning: Unfilled placeholders remain: {final_remaining}")
             # Try to provide a more informative error message
             missing_keys = [p.strip("{}") for p in final_remaining]
             return f"-- Could not generate SQL: Missing required parameters {missing_keys} for template."

        return query.strip() # Remove leading/trailing whitespace
    except Exception as e:
        print(f"Error during SQL generation: {e}")
        return f"-- Error generating SQL: {e}"

# --- Main Execution Flow ---

def nl_to_sql(user_query):
    """
    Main function to convert natural language query to SQL.
    """
    print(f"\nProcessing Query: '{user_query}'")

    # Step 1: Extract Entities
    entities = extract_entities(user_query)

    # Step 2: Find Best Template and Fill Parameters
    sql_template, filled_params = find_best_template(user_query, entities)

    # Step 3: Generate SQL
    final_sql = generate_sql(sql_template, filled_params)

    print("\n--- Generated SQL ---")
    print(final_sql)
    return final_sql

# --- Example Usage ---
if __name__ == "__main__":
    # Example Queries
    nl_to_sql("How many customers are in London?")
    nl_to_sql("Show total sales for product 'Laptop'")
    # nl_to_sql("List all products") # Might need a specific template for simple listings
    nl_to_sql("What is the average price of products in the 'Electronics' category?") # Needs template + condition logic + AVG aggregation
    nl_to_sql("Count sales per region") # Needs GROUP BY template
    nl_to_sql("show me sales over 1000") # Needs condition detection and template

    # Example needing more complex date handling (not fully implemented above)
    # nl_to_sql("Show sales from last month")
