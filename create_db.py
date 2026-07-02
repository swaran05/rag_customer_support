import sqlite3

# Connect to SQLite database (creates users.db if it doesn't exist)
conn = sqlite3.connect("users.db")

# Create a cursor object
cursor = conn.cursor()

# Create the users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    membership_tier TEXT NOT NULL
)
""")

# Remove existing data (optional, avoids duplicates if run multiple times)
cursor.execute("DELETE FROM users")

# Insert sample users
users = [
    (101, "Riya Sharma", "Gold"),
    (102, "Aman Verma", "Silver"),
    (103, "Neha Iyer", "Platinum")
]

cursor.executemany(
    "INSERT INTO users (user_id, name, membership_tier) VALUES (?, ?, ?)",
    users
)

# Save changes
conn.commit()

# Close connection
conn.close()

print("Database created successfully!")
print("Sample users inserted successfully!")