# test_connection.py
import psycopg2

# Try different passwords if you're not sure
passwords_to_try = [
    "Iamrob123$",
    "postgres",
    "password",
    "312413"  # Update with your actual password
    # Add any other passwords you might have used
]

for password in passwords_to_try:
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="energy_dashboard",
            user="postgres",
            password=password
        )
        print(f"✅ SUCCESS! Password '{password}' works!")
        conn.close()
        break
    except:
        print(f"❌ Password '{password}' failed")