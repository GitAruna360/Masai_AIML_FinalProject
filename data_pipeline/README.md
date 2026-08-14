# Module1- Books Web Scraping & Database Project

This project scrapes book information from **Books to Scrape**, cleans and transforms the data, saves it as a CSV file, and stores it in a normalized SQLite database.

The project also performs SQL queries and uses Pandas to analyze the stored data.

## What This Project Does

The Python script performs the following main steps:

1. **Scrapes book data**

   * Scrapes the first 5 pages of the website.
   * Collects book title, price, rating, availability, and category.

2. **Cleans and transforms data**

   * Converts GBP prices into INR using a fixed exchange rate.
   * Converts ratings from text to numbers.
   * Handles missing values.
   * Creates a Pandas DataFrame.

3. **Creates CSV output**

   * Saves the cleaned data as:
     `output/books.csv`

4. **Creates SQLite database**

   * Creates `categories` and `books` tables.
   * Uses a foreign key to connect books with their categories.
   * Stores the scraped data in `books.db`.

5. **Runs SQL queries**

   * Finds highly rated books.
   * Sorts books by price.
   * Displays categories.
   * Filters books by price.
   * Performs a SQL JOIN between books and categories.

6. **Performs Pandas analysis**

   * Reads database tables into Pandas.
   * Uses `pd.merge()` to perform a similar JOIN.
   * Compares the SQL JOIN and Pandas merge results.

7. **Saves query results**

   * Stores the query outputs in:
     `output/queries_output.txt`

## Project Structure

```text
project/
│
├── main.py
├── books.db
│
└── output/
    ├── books.csv
    └── queries_output.txt
```

## Technologies Used

* Python
* Requests
* BeautifulSoup
* Pandas
* SQLite
* SQL

## How to Run

Install the required Python libraries:

```bash
pip install requests pandas beautifulsoup4
```

Run the Python script:

```bash
python main.py
```

The script will automatically create the `output` folder and generate the CSV and query output files.

## Output Files

### books.csv

Contains the cleaned book data, including:

* Book title
* Price in GBP
* Price in INR
* Rating
* Stock availability
* Category

### books.db

SQLite database containing:

* `books` table
* `categories` table

### queries_output.txt

Contains the results of the SQL and Pandas queries performed by the script.

## Summary

This project demonstrates a basic **web scraping → data cleaning → CSV → relational database → SQL/Pandas analysis** workflow using Python.
