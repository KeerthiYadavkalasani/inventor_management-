# Enhanced Inventory Management System

This project is a comprehensive web-based **Inventory Management System** designed to help manage product inventory, track sales, predict future sales trends, and provide dynamic stock recommendations. Built with Flask for the backend and HTML/JavaScript for the frontend, this system is robust, scalable, and user-friendly.

---

## Features

### **Product Management**
- Add, edit, and delete products with ease.
- Set and update low-stock thresholds.
- Products are automatically added to inventory from uploaded sales CSV files if not already existing.

### **Sales Data Processing**
- Upload sales data through CSV files. Required columns: `date`, `name`, `sales`, and `quantity`.
- Automatically updates inventory and sales metrics.
- Tracks real-time sales data and updates quantities dynamically.

### **Dynamic Recommendations**
- Predicts next-day sales using advanced machine learning algorithms.
- Recommends stock adjustments based on sales predictions and low-stock thresholds.
- Visual indicators highlight inventory status: `Safe`, `At Risk`, or `Overstocked`.

### **Responsive and Intuitive UI**
- Responsive design ensures a seamless experience on any device.
- Clear visual elements for managing and viewing product data.
- Modals for confirmations ensure error-free operations.

### **Visualization and Predictions**
- Generates visual sales predictions using Matplotlib.
- Displays predictive data and recommended stock adjustments dynamically after each CSV upload.

---

## Getting Started

Follow these instructions to set up the project locally.

### **Prerequisites**
- Python 3.6 or higher
- `pip` (Python package manager)
- Basic knowledge of Flask, HTML, CSS, and JavaScript

### **Installation Steps**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KeerthiYadavkalasani/inventor_management-.git
   cd inventor_management-
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Required Libraries:**
   ```text
   Flask
   Flask-Cors
   pandas
   scikit-learn
   matplotlib
   werkzeug
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```
   The app will be accessible at `http://127.0.0.1:5000`.

### **Usage**

#### **Managing Products**
- View products and their details (quantity, stock status, and sales).
- Add new products via the "+ Add Product" button.
- Edit product details (e.g., quantity and low-stock threshold).
- Delete products with confirmation prompts.

#### **Uploading Sales Data**
- Navigate to the "Sales Data" section.
- Upload a CSV file with the required columns.
- Review the updated inventory and sales predictions.

#### **Viewing Predictions**
- Access predictive sales data and recommended stock adjustments immediately after uploading CSV files.

---

## File Structure
```plaintext
inventory-management/
├── app.py                 # Backend Flask application
├── index.html             # Main frontend file
├── uploads/               # Directory for uploaded CSV files
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── venv/                  # Virtual environment folder
```

---

## API Endpoints

### **Products**
- `GET /products`: Retrieve all products.
- `POST /products`: Add a new product.
- `PUT /products/<int:product_id>`: Update a product.
- `DELETE /products/<int:product_id>`: Remove a product.

### **Sales and Predictions**
- `POST /predict_sales`: Process uploaded sales data.
- `POST /predict`: Predict sales and return results.
- `POST /update_sales`: Update inventory based on sales data.

---

## Technologies Used

### **Backend**
- Flask (Python)

### **Frontend**
- HTML
- CSS
- JavaScript

### **Libraries**
- Pandas: Data manipulation
- Scikit-learn: Machine learning
- Matplotlib: Data visualization
- Flask-Cors: Cross-origin requests
- Werkzeug: WSGI utility library

---

## Future Enhancements
- Add user authentication and role-based access.
- Implement advanced analytics dashboards.
- Integrate with external APIs for real-time sales data.
- Support for multi-warehouse inventory management.

---

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue to discuss improvements.

---



## Credits
- Developed by [Keerthi Yadav]

This README ensures a professional and comprehensive guide for anyone interacting with your project.

