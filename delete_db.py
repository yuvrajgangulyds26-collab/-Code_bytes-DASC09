import sqlite3
conn = sqlite3.connect('users.db')
cursor = conn.cursor()
cursor.execute('DROP TABLE IF EXISTS history')
conn.commit()
conn.close()
print("table dropped")