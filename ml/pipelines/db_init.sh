#!/bin/bash

# PostgreSQL connection parameters (the rest are read through .env)
DB_HOST="db"
DB_PORT=5432

# Paths to train and test CSV files
TRAIN_CSV_PATH="$ML_DATA_DIR/raw"

# Define the table name
TABLE_NAME="raw_data"

# Path to the .pgpass file
PGPASS_FILE="$HOME/.pgpass"

# Create or overwrite the .pgpass file (prevents multiple pw prompts)
echo "$DB_HOST:$DB_PORT:$DB_NAME:$DB_USER:$DB_PASSWORD" > "$PGPASS_FILE"

# Ensure correct permissions on the .pgpass file
chmod 600 "$PGPASS_FILE"


# Connect to PostgreSQL and create the "raw_data" table
psql -U "$DB_USER" -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -c \
    "CREATE TABLE IF NOT EXISTS $TABLE_NAME (
        is_train BOOLEAN,
        type TEXT,
        sector TEXT,
        net_usable_area NUMERIC,
        net_area NUMERIC,
        n_rooms NUMERIC,
        n_bathroom NUMERIC,
        latitude NUMERIC,
        longitude NUMERIC,
        price NUMERIC
    );"

# Merge train and test CSV files with an "is_train" column
awk 'NR==1{print "is_train," $0; next} {print "1," $0}' "$TRAIN_CSV_PATH/$TRAIN_FILE_NAME" > merged_data.csv
awk 'NR==1{next} {print "0," $0}' "$TRAIN_CSV_PATH/$TEST_FILE_NAME" >> merged_data.csv


# Step 3: Load the merged data into the "raw_data" table
psql -U "$DB_USER" -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -c \
    "\\copy $TABLE_NAME FROM 'merged_data.csv' CSV HEADER;"

# Step 4: Clean up - remove the merged data file
rm merged_data.csv

# Step 5: Print the first N rows of data from the "raw_data" table
N=10  # Change this to the number of rows you want to retrieve
psql -U "$DB_USER" -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -c \
    "SELECT * FROM $TABLE_NAME LIMIT $N;"

echo "Initialization completed."
