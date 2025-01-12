from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from frontend

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}


#----------------- Global Data -----------------
products = [
    {"id": 1, "name": "apple phone", "quantity": 1000, "recommended_stock":0, "sales": 0, "low_stock_threshold": 200},
    {"id": 2, "name": "xiomixi", "quantity": 900, "recommended_stock":0, "sales": 0, "low_stock_threshold": 150},
    {"id": 3, "name": "mi phone", "quantity": 200, "recommended_stock":0, "sales": 0, "low_stock_threshold": 100},
    {"id": 4, "name": "sony sled", "quantity": 500, "recommended_stock":0, "sales": 0, "low_stock_threshold": 250}
]
next_product_id = 5
# ----------------------------------------


# -----------------------Data Preprocessing -----------------------------
def preprocess_data(sales_data):
    # Handle missing values (example: filling with 0)
    sales_data.fillna(0, inplace=True)
    # Explicitly infer dtypes to avoid future warnings
    sales_data = sales_data.infer_objects(copy=False)

    # Convert date column to datetime (example: if you have date column)
    if 'date' in sales_data.columns:
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        sales_data['day_of_week'] = sales_data['date'].dt.dayofweek
        sales_data['month'] = sales_data['date'].dt.month
        sales_data['year'] = sales_data['date'].dt.year
    return sales_data
#---------------------------------------------------------------------------------

# -------------------------- Model Training --------------------------------------
def train_model(sales_data):
    sales_data = preprocess_data(sales_data) # Calling the data preprocessing function
    
    # Example feature selection (modify based on your data)
    features = ['day_of_week', 'month', 'year'] if 'date' in sales_data.columns else ['feature1', 'feature2']
    if not all(feature in sales_data.columns for feature in features):
        raise ValueError(f"Features {features} are not available in the input data")
    target = 'sales'

    # Splitting data
    X = sales_data[features]
    y = sales_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    model = LinearRegression()
    model.fit(X_train, y_train) #Training model

    # Model Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Mean Absolute Error : {mae}")
    return model
#--------------------------------------------------------------------------

# --------------------------Prediction Function ------------------------------
def make_prediction(model, new_data):
    new_data = preprocess_data(new_data)
    features = ['day_of_week', 'month', 'year'] if 'date' in new_data.columns else ['feature1', 'feature2']
    if not all(feature in new_data.columns for feature in features):
        raise ValueError(f"Features {features} are not available in the input data")
    X_new = new_data[features]
    predictions = model.predict(X_new)
    return predictions.tolist()

#----------------------------------------------------------------------------
#--------------------------Plotting function--------------------------------
def plot_predictions(sales_data, predictions):
    if 'date' not in sales_data.columns:
          raise ValueError("Date column is required for plotting")
    sales_data = preprocess_data(sales_data)
    # Creating a temporary figure to generate the image
    plt.figure(figsize=(10, 6))
    plt.plot(sales_data['date'], sales_data['sales'], label='Actual Sales')
    plt.plot(sales_data['date'], predictions, label='Predicted Sales')
    plt.xlabel("Date")
    plt.ylabel('Sales')
    plt.title("Actual vs Predicted Sales")
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a buffer and convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close() # Closing the figure to avoid displaying it
    return img_base64

#---------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#-----------------------Api endpoints-----------------------------------------
@app.route('/')
def index():
    return render_template('index.html', products=products)


@app.route('/products', methods=['GET'])
def get_products():
     return jsonify(products)

@app.route('/products', methods=['POST'])
def add_product():
    try:
        global products, next_product_id
        new_product = request.get_json()
        # Check if the product with the same name already exists
        if any(product['name'] == new_product['name'] for product in products):
            return jsonify({'error': 'Product with this name already exists'}), 400
        new_product["id"] = next_product_id;
        new_product["sales"]= 0;
        products.append(new_product)
        next_product_id += 1
        return jsonify({'message': 'Product added successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/products/<int:product_id>', methods=['PUT'])
def update_product(product_id):
    try:
        updated_data = request.get_json()
        for product in products:
            if product['id'] == product_id:
                product.update(updated_data)
                return jsonify({'message': 'Product updated successfully'}), 200
        return jsonify({'message': 'Product not found'}), 404
    except Exception as e:
       return jsonify({'error': str(e)}), 500

@app.route('/products/<int:product_id>', methods=['DELETE'])
def delete_product(product_id):
    global products
    product = next((item for item in products if item['id'] == product_id), None)
    if not product:
        return "Product not found", 404

    products = [item for item in products if item['id'] != product_id]
    return jsonify({'message': 'Product deleted successfully'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
       data = request.get_json()
       if 'sales' not in data or not isinstance(data['sales'], list):
           return jsonify({"error": "Invalid sales data format"}), 400
       
       sales_data = pd.DataFrame(data['sales'])

       if sales_data.empty:
           return jsonify({"error": "No sales data provided"}), 400
       
       model = train_model(sales_data)

       # New data for prediction
       if 'new_data' not in data or not isinstance(data['new_data'],list):
           return jsonify({"error": "New data for prediction is not found or format is not correct"}), 400
       new_data = pd.DataFrame(data['new_data'])

       predictions = make_prediction(model, new_data)
       img_base64 = plot_predictions(sales_data, predictions)

       return jsonify({"predictions": predictions, "plot": img_base64})
    except ValueError as e:
          return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": "An error occurred during prediction", "details": str(e)}), 500
@app.route('/update_sales', methods=['POST'])
def update_sales():
     try:
        data = request.get_json();
        if not data or 'sales' not in data or not isinstance(data['sales'], list):
            return jsonify({'error': 'Invalid sales data format'}), 400

        sales_data_list = data['sales']
        for sale_data in sales_data_list:
            product_name = sale_data.get('name');
            sale_value = sale_data.get('sales');
            
            if not product_name or sale_value is None:
                  return jsonify({'error': 'Invalid sales data format, product name or sale is missing'}), 400
            
            product = next((item for item in products if item['name'] == product_name), None)
            
            if not product:
                  return jsonify({'error': f'Product {product_name} not found'}), 404
            
            product['sales'] = int(product.get('sales',0)) + int(sale_value);
            product['quantity'] = int(product.get('quantity',0)) - int(sale_value) if (int(product.get('quantity',0)) - int(sale_value)) >0 else 0;

        return jsonify({'message':'Sales data updated successfully'}), 200

     except ValueError as e:
        return jsonify({'error': str(e)}), 400
     except Exception as e:
         return jsonify({'error': 'An unexpected error has occured', 'details': str(e)}), 500

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    global products, next_product_id
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            sales_data = pd.read_csv(filepath)
            
            # Prepare sales data for training
            processed_sales_data = []
            products_to_add = []
            # Handle edge cases : for empty sales data
            if sales_data.empty:
                 os.remove(filepath)
                 return jsonify({'error': 'No sales data found in the CSV'}), 400

            for index, row in sales_data.iterrows():
                product_name = row.get('name')
                if product_name:
                    product_name = product_name.strip().lower()  # Normalize product name
                    existing_product = next((item for item in products if item['name'].strip().lower() == product_name), None)
                    
                    if existing_product:
                        processed_sales_data.append({
                        'date': row.get('date'),
                         'name': product_name,
                         'sales': row.get('sales'),
                         'quantity': row.get('quantity')
                        })
                        # Update total quantity
                        for product in products:
                            if product['name'] == product_name:
                                 product['quantity'] = int(product.get('quantity',0)) + int(row.get('quantity', 0))
                                 product['sales'] = 0;
                    else:
                         if not any(p['name'].strip().lower() == product_name for p in products_to_add):
                            products_to_add.append({
                                'name': product_name,
                           })

                else:
                    print("product name not found in CSV")
                     #handle the case when the product name itself is not found
            # Add new products from the CSV to our products list
            for product in products_to_add:
                product["id"] = next_product_id
                product["quantity"]= int(row.get('quantity', 100))
                product["recommended_stock"] = 0
                product["sales"] = 0;
                product["low_stock_threshold"] = 100
                products.append(product);
                next_product_id+=1
            print(f"added products : {products_to_add}")

            
            sales_df = pd.DataFrame(processed_sales_data)
           
            
            if sales_df.empty:
                 os.remove(filepath)
                 return jsonify({'error': 'No valid sales data found for existing products'}), 400

            sales_df['sales'] = pd.to_numeric(sales_df['sales'], errors='coerce')
            sales_df = sales_df.dropna(subset=['sales']) # Drop any rows that has nan values
           
            if sales_df.empty:
                os.remove(filepath)
                return jsonify({'error': 'No valid sales data after processing'}), 400


            model = train_model(sales_df)
           
            # Create future data set
            new_prediction_data = []
            for index, row in sales_df.iterrows():
                  new_prediction_data.append({
                           "date":"2024-01-07",
                           "name": row.get('name')
                   })
            new_df= pd.DataFrame(new_prediction_data);
            predictions = make_prediction(model, new_df)
            print(f"predictions: {predictions}")

             # Create recommended_stock values for each product
            for index, row in sales_df.iterrows():
                product_name = row.get('name')
                
                # Get product with max sales
                max_sales_index = sales_df['sales'].idxmax()
                product_max_sales = sales_df.iloc[max_sales_index]
                product_name = product_max_sales.get('name', 'Unknown Product')


                predicted_next_sales = predictions[index]
                current_stock = next((item['quantity'] for item in products if item['name'] == product_name), 0) # Find if product exists, else use zero
                low_stock_threshold = next((item['low_stock_threshold'] for item in products if item['name'] == product_name), 0) # Find if product exists, else use zero
                stock_recommendation = predicted_next_sales*2 - current_stock
                for product in products:
                        if product['name'] == product_name:
                              product['recommended_stock'] = stock_recommendation if stock_recommendation >low_stock_threshold else low_stock_threshold
                              product["predicted_sales"] = predicted_next_sales; # Add predicted sales here

            os.remove(filepath) #Remove the csv file after reading it.
            return jsonify({"message": "Sales data processed successfully"})
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction", "details": str(e)}), 500
#--------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
