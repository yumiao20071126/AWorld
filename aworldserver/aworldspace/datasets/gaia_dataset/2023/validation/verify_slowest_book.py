import pandas as pd
from datetime import datetime

# Path to the Excel file
file_path = '/Users/yuchengyue/AWorld_mcp/gaia/gaia_dataset/2023/validation/da52d699-e8d2-4dc5-9191-a2199e0b6a9b.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# Define word counts for each book based on published data
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

# Print date information to verify format
print("Date Information:")
for _, row in df.iterrows():
    print(f"{row['Title']}: {row['Start Date']} to {row['End Date']}")

# Calculate reading duration in days
df['Duration'] = (pd.to_datetime(df['End Date']) - pd.to_datetime(df['Start Date'])).dt.days

# Calculate reading speed (words per day)
df['Words per Day'] = df['Word Count'] / df['Duration']

# Sort by reading speed (slowest first)
df_sorted = df.sort_values('Words per Day')

# Display all books with their reading rates
print("\nAll books sorted by reading rate (words per day):")
print(df_sorted[['Title', 'Author', 'Word Count', 'Duration', 'Words per Day']].to_string())

# Get the slowest book
slowest_book = df_sorted.iloc[0]
print("\nSlowest book by reading rate (words per day):")
print(f"Title: {slowest_book['Title']}")
print(f"Author: {slowest_book['Author']}")
print(f"Word Count: {slowest_book['Word Count']} words")
print(f"Duration: {slowest_book['Duration']} days")
print(f"Words per Day: {slowest_book['Words per Day']:.2f} words/day")