# 🇨🇦 Canada Express Entry Draws Dashboard

This project presents a simple yet effective Streamlit web application to explore and analyze data from Canada's Express Entry invitation rounds. Built in Python with Pandas and Streamlit, it allows users to filter draw data by various criteria, view key metrics, and observe trends in Comprehensive Ranking System (CRS) scores over time.

---

## 🚀 Features

- **Interactive Filters:** Easily narrow down data by `Draw Name`, `Draw Date Range`, `Draw Size`, and `Draw CRS Score`.
- **Customizable Display:** Choose which columns are visible in the data table for a focused view.
- **Key Metrics at a Glance:** See instant summaries like the total number of filtered draws and average CRS scores.
- **CRS Score Trend:** Visualize how CRS scores have evolved across different invitation rounds with a built-in line chart.
- **Raw Data Toggle:** Option to view the complete, unfiltered dataset for comprehensive review.

---

## 🛠️ Technologies Used

- **Python:** The core programming language.
- **Pandas:** For efficient data manipulation and analysis.
- **Streamlit:** For creating the interactive web application.

---

## 💻 How to Run the App Locally

Follow these steps to get the Express Entry Draws Dashboard running on your local machine:

1.  **Clone the Repository (or Save the File):**
    If this is part of a Git repository, clone it:

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

    Otherwise, simply save the provided Python code (e.g., `express_entry_app.py`) to a folder on your computer.

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    ```

    Activate the virtual environment:

    - **Windows:** `.\venv\Scripts\activate`
    - **macOS/Linux:** `source venv/bin/activate`

3.  **Install Dependencies:**
    With your virtual environment activated, install the necessary libraries:

    ```bash
    pip install streamlit pandas
    ```

4.  **Run the Streamlit Application:**
    Navigate to the directory where you saved `express_entry_app.py` (or the root of your cloned repository) in your terminal and run:

    ```bash
    streamlit run streamlit.py
    ```

    Your web browser should automatically open a new tab with the Streamlit application. If it doesn't, copy and paste the URL provided in your terminal (usually `http://localhost:8501`) into your browser.

---

## 📂 Project Structure

.
├── express_entry_app.py # The main Streamlit application code
└── README.md # This file

---

This project is a starting point for exploring Express Entry data. Feel free to fork the repository and enhance it!

# The MIT License (MIT)

Copyright © `<2025>` `<copyright holders>`

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
