import sqlite3

# Connect to sqlite
connection = sqlite3.connect("student.db")

cursor = connection.cursor()

# create the table
table_info = """
Create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), 
SECTION VARCHAR(25), MARKS INT)
"""

cursor.execute(table_info)

# Insert some data
cursor.execute('''Insert Into STUDENT values ('A', 'Data Science', 'A', 90)''')
cursor.execute('''Insert Into STUDENT values ('B', 'Data Science', 'B', 100)''')
cursor.execute('''Insert Into STUDENT values ('C', 'Data Science', 'A', 80)''')
cursor.execute('''Insert Into STUDENT values ('D', 'MLOPS', 'C', 90)''')
cursor.execute('''Insert Into STUDENT values ('E', 'MACHINE LEARNING', 'B', 70)''')

# Display all the records
data = cursor.execute('''Select * from STUDENT''')

for row in data:
    print(row)

# close the connection
connection.commit()
connection.close()