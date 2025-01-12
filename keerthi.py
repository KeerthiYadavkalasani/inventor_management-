import csv
import random
from datetime import date, timedelta
import google.generativeai as genai

# Function to generate a random date within the past year
def random_date():
    start_date = date.today() - timedelta(days=365)
    time_between_dates = date.today() - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + timedelta(days=random_number_of_days)
    return random_date.strftime("%Y-%m-%d")

# List of sample product names
product_names = [
  "Laptop",
  "Smartphone",
    "Tablet",
    "Smartwatch",
    "Headphones",
    "Keyboard",
    "Mouse",
    "Monitor",
    "Printer",
     "Router",
  "Camera",
    "Speaker",
    "External Hard Drive",
   "USB Flash Drive",
  "Charger",
    "Power Bank",
  "Smart Bulb",
    "Smart Thermostat",
    "Smart Lock",
   "Wireless Earbuds"
]
def generate_sales_data(num_rows, use_gemini=False, api_key=None):
    """
    Generate synthetic sales data.

    Args:
      num_rows: The number of rows of data to generate.
      use_gemini: Whether to use the Gemini API to generate product names.
      api_key: The API key to use when use_gemini=True
    """
    local_product_names= product_names # set the local product names with global product_names

    if use_gemini and api_key:
        try:
             genai.configure(api_key=api_key)
             model = genai.GenerativeModel('gemini-pro')
             local_product_names =  get_gemini_data(model)
        except Exception as e:
           print("There was error in gemini API call, hence using local product names",e)
    sales_data = []

    for _ in range(num_rows):
        name = random.choice(local_product_names)
        sales = random.randint(10, 200) # Random sales count between 10 and 200
        date = random_date()

        sales_data.append({'date': date, 'name': name, 'sales': sales})
    return sales_data

def get_gemini_data(model):
    """
      Get synthetic product names using gemini pro api.

    Args:
      model: Gemini Pro Generative Model
    """
    try:
      response = model.generate_content("Generate 20 unique product names of retail products like grocery or electronics etc.")
      response.resolve()
      product_names = response.text.strip().split("\n")
      return product_names
    except Exception as e:
        print("Error during Gemini API call:",e)
        return []

def save_to_csv(sales_data, filename='synthetic_sales_data.csv'):
    """
    Save the sales data to a CSV file.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'name', 'sales']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(sales_data)

if __name__ == '__main__':
    num_rows = 1000  # Generate 100 rows
    use_gemini_api = False # Set to True if you want to use the Gemini API
    gemini_api_key = "AIzaSyBoHaqZ_lvjVwmhdq7sfdJkaEmpHqdNlR4" # Replace with your Gemini API key


    synthetic_data = generate_sales_data(num_rows, use_gemini_api, gemini_api_key)
    save_to_csv(synthetic_data)
    print("Synthetic CSV file 'synthetic_sales_data.csv' has been created.")