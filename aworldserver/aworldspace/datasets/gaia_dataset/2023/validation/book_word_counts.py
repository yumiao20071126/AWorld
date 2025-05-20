import pandas as pd
import numpy as np
from datetime import datetime

# Path to the Excel file
file_path = '/Users/yuchengyue/AWorld_mcp/gaia/gaia_dataset/2023/validation/da52d699-e8d2-4dc5-9191-a2199e0b6a9b.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# Display the complete dataframe
print("Reading list data:")
print(df)

# Define word counts for each book (since this information is not in the spreadsheet)
# These are typical word counts for these books based on common knowledge
word_counts = {
    "Fire and Blood": 300000,
    "Song of Solomon": 93400,
    "The Lost Symbol": 126000,
    "2001: A Space Odyssey": 70000,
    "American Gods": 183000,
    "Out of the Silent Planet": 58000,
    "The Andromeda Strain": 90000,
    "Brave New World": 64000,
    "Silence": 103000,
    "The Shining": 160000
}

# Add word count column to the dataframe
df['Word Count'] = df['Title'].map(word_counts)

# Calculate reading duration in days
df['Duration'] = (pd.to_datetime(df['End Date']) - pd.to_datetime(df['Start Date'])).dt.days

# Calculate reading speed (words per day)
df['Words per Day'] = df['Word Count'] / df['Duration']

# Display the dataframe with the additional calculated columns
print("\nReading list with calculated reading speed:")
print(df[['Title', 'Word Count', 'Duration', 'Words per Day']])

# Find the book with the slowest reading speed
slowest_book = df.loc[df['Words per Day'].idxmin()]

print("\nSlowest book:")
print(f"Title: {slowest_book['Title']}")
print(f"Word Count: {slowest_book['Word Count']}")
print(f"Duration: {slowest_book['Duration']} days")
print(f"Words per Day: {slowest_book['Words per Day']:.2f}")