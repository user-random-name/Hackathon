This script cleans temperature data from a CSV file. It reads train-input.csv, fixes incorrectly encoded station names, converts temperatures to Celsius, removes duplicate readings, and saves the cleaned data in train-output.csv.
The script works in four steps:
1. Reading the file: Removes blank lines and comments starting with #, then reads the CSV using ; as the separator.
2. Fixing station names: The fix_name() function corrects incorrectly encoded characters (mojibake) by trying to decode them using UTF-8 after encoding with cp1252 or latin-1.
3. Cleaning temperatures and duplicates: Temperatures in Fahrenheit are converted to Celsius, while existing Celsius values are kept unchanged. All temperatures are rounded to one decimal place. Duplicate records are identified by reading_id, with the original Celsius record preferred over Fahrenheit.
4. Writing the output: Records are sorted by reading_id and saved in train-output.csv with the columns reading_id, station_name, temperature_c and reading_date.
Usage:
Place train-input.csv in the same folder as the script and run the file
The cleaned data will be saved automatically in train-output.csv.
