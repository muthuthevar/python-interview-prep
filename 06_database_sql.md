# Database & SQL - Interview Questions

## SQL Fundamentals

### 1. Basic SQL Operations

#### SELECT Statements
```sql
-- Basic select
SELECT column1, column2 FROM table_name;

-- Select all
SELECT * FROM table_name;

-- Select distinct
SELECT DISTINCT column1 FROM table_name;

-- Select with conditions
SELECT * FROM table_name WHERE condition;

-- Select with ordering
SELECT * FROM table_name ORDER BY column1 ASC, column2 DESC;

-- Select with limit
SELECT * FROM table_name LIMIT 10;
```

#### JOIN Operations
```sql
-- INNER JOIN
SELECT * FROM table1
INNER JOIN table2 ON table1.id = table2.id;

-- LEFT JOIN
SELECT * FROM table1
LEFT JOIN table2 ON table1.id = table2.id;

-- RIGHT JOIN
SELECT * FROM table1
RIGHT JOIN table2 ON table1.id = table2.id;

-- FULL OUTER JOIN
SELECT * FROM table1
FULL OUTER JOIN table2 ON table1.id = table2.id;

-- CROSS JOIN
SELECT * FROM table1 CROSS JOIN table2;

-- Self JOIN
SELECT a.name, b.name FROM employees a
JOIN employees b ON a.manager_id = b.id;
```

#### Aggregation Functions
```sql
-- COUNT
SELECT COUNT(*) FROM table_name;
SELECT COUNT(DISTINCT column1) FROM table_name;

-- SUM, AVG, MIN, MAX
SELECT SUM(amount), AVG(amount), MIN(amount), MAX(amount)
FROM transactions;

-- GROUP BY
SELECT department, COUNT(*) as employee_count
FROM employees
GROUP BY department;

-- HAVING (filter after GROUP BY)
SELECT department, COUNT(*) as employee_count
FROM employees
GROUP BY department
HAVING COUNT(*) > 10;
```

#### Subqueries
```sql
-- Subquery in WHERE
SELECT * FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Subquery in SELECT
SELECT name, (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) as order_count
FROM users;

-- EXISTS
SELECT * FROM customers
WHERE EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.id);

-- IN
SELECT * FROM products
WHERE category_id IN (SELECT id FROM categories WHERE name = 'Electronics');
```

### 2. Data Modification

#### INSERT
```sql
-- Insert single row
INSERT INTO table_name (column1, column2) VALUES (value1, value2);

-- Insert multiple rows
INSERT INTO table_name (column1, column2)
VALUES (value1, value2), (value3, value4);

-- Insert from select
INSERT INTO table1 (column1, column2)
SELECT column1, column2 FROM table2;
```

#### UPDATE
```sql
-- Update with condition
UPDATE table_name
SET column1 = value1, column2 = value2
WHERE condition;

-- Update with subquery
UPDATE employees
SET salary = salary * 1.1
WHERE department_id IN (SELECT id FROM departments WHERE name = 'Engineering');
```

#### DELETE
```sql
-- Delete with condition
DELETE FROM table_name WHERE condition;

-- Delete all (be careful!)
DELETE FROM table_name;

-- Truncate (faster, cannot rollback)
TRUNCATE TABLE table_name;
```

### 3. Advanced SQL

#### Window Functions
```sql
-- ROW_NUMBER
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- RANK and DENSE_RANK
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) as rank,
       DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;

-- PARTITION BY
SELECT name, department, salary,
       AVG(salary) OVER (PARTITION BY department) as dept_avg_salary
FROM employees;

-- LAG and LEAD
SELECT date, sales,
       LAG(sales) OVER (ORDER BY date) as prev_sales,
       LEAD(sales) OVER (ORDER BY date) as next_sales
FROM daily_sales;
```

#### Common Table Expressions (CTEs)
```sql
-- Simple CTE
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 100000
)
SELECT * FROM high_earners;

-- Recursive CTE
WITH RECURSIVE hierarchy AS (
    -- Base case
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case
    SELECT e.id, e.name, e.manager_id, h.level + 1
    FROM employees e
    JOIN hierarchy h ON e.manager_id = h.id
)
SELECT * FROM hierarchy;
```

#### Indexes
```sql
-- Create index
CREATE INDEX idx_name ON table_name(column1);

-- Create composite index
CREATE INDEX idx_name ON table_name(column1, column2);

-- Create unique index
CREATE UNIQUE INDEX idx_name ON table_name(column1);

-- Drop index
DROP INDEX idx_name;
```

## Database Design

### 1. Normalization

#### First Normal Form (1NF)
- Each column contains atomic values
- No repeating groups
- Each row is unique

#### Second Normal Form (2NF)
- Must be in 1NF
- All non-key attributes fully dependent on primary key
- No partial dependencies

#### Third Normal Form (3NF)
- Must be in 2NF
- No transitive dependencies
- Non-key attributes independent of each other

### 2. Relationships

#### One-to-One
- One record in Table A relates to one record in Table B
- Example: User and UserProfile

#### One-to-Many
- One record in Table A relates to many records in Table B
- Example: User and Orders

#### Many-to-Many
- Many records in Table A relate to many records in Table B
- Requires junction table
- Example: Students and Courses

### 3. ACID Properties

#### Atomicity
- All operations in transaction succeed or all fail
- No partial updates

#### Consistency
- Database remains in valid state
- Constraints always satisfied

#### Isolation
- Concurrent transactions don't interfere
- Isolation levels: Read Uncommitted, Read Committed, Repeatable Read, Serializable

#### Durability
- Committed transactions persist
- Survives system failures

## Common Interview Questions

### Q1: Explain the difference between INNER JOIN and LEFT JOIN
- **INNER JOIN**: Returns only matching rows from both tables
- **LEFT JOIN**: Returns all rows from left table, matching rows from right table (NULL if no match)

### Q2: What is the difference between WHERE and HAVING?
- **WHERE**: Filters rows before grouping
- **HAVING**: Filters groups after GROUP BY

### Q3: Explain indexes and when to use them
- Speed up queries
- Trade-off: Slower writes, more storage
- Use on frequently queried columns
- Composite indexes for multi-column queries

### Q4: What is a transaction?
- Group of operations that execute as single unit
- All succeed or all fail
- ACID properties

### Q5: Explain database normalization
- Process of organizing data to reduce redundancy
- Normal forms (1NF, 2NF, 3NF)
- Trade-off: More tables, more joins

### Q6: What is the difference between DELETE and TRUNCATE?
- **DELETE**: Removes rows, can be rolled back, slower, can have WHERE clause
- **TRUNCATE**: Removes all rows, cannot be rolled back, faster, resets auto-increment

### Q7: Explain SQL injection and how to prevent it
- Malicious SQL code injection
- Prevention: Parameterized queries, input validation, least privilege

### Q8: What are window functions?
- Perform calculations across set of rows
- Don't collapse rows like GROUP BY
- Examples: ROW_NUMBER, RANK, LAG, LEAD

### Q9: Explain the difference between UNION and UNION ALL
- **UNION**: Combines results, removes duplicates
- **UNION ALL**: Combines results, keeps duplicates (faster)

### Q10: What is a stored procedure?
- Precompiled SQL code stored in database
- Can accept parameters
- Improves performance, security

## Python Database Integration

### 1. SQLite
```python
import sqlite3

# Connect
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Execute query
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
result = cursor.fetchone()

# Execute many
cursor.executemany("INSERT INTO users (name) VALUES (?)", [('Alice',), ('Bob',)])

# Commit
conn.commit()

# Close
conn.close()

# Context manager
with sqlite3.connect('database.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
```

### 2. PostgreSQL (psycopg2)
```python
import psycopg2
from psycopg2.extras import RealDictCursor

conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="user",
    password="password"
)

cursor = conn.cursor(cursor_factory=RealDictCursor)
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
result = cursor.fetchone()

conn.commit()
conn.close()
```

### 3. SQLAlchemy (ORM)
```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    email = Column(String(100))

# Create engine
engine = create_engine('sqlite:///database.db')
Base.metadata.create_all(engine)

# Create session
Session = sessionmaker(bind=engine)
session = Session()

# Query
users = session.query(User).filter(User.name == 'Alice').all()

# Create
new_user = User(name='Bob', email='bob@example.com')
session.add(new_user)
session.commit()

# Update
user = session.query(User).filter(User.id == 1).first()
user.name = 'Alice Updated'
session.commit()

# Delete
session.delete(user)
session.commit()
```

### 4. Database Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

## NoSQL Databases

### 1. MongoDB (Document Store)
```python
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['mydatabase']
collection = db['users']

# Insert
collection.insert_one({'name': 'Alice', 'age': 30})

# Find
user = collection.find_one({'name': 'Alice'})
users = collection.find({'age': {'$gt': 25}})

# Update
collection.update_one({'name': 'Alice'}, {'$set': {'age': 31}})

# Delete
collection.delete_one({'name': 'Alice'})
```

### 2. Redis (Key-Value Store)
```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# Set/Get
r.set('key', 'value')
value = r.get('key')

# Hash
r.hset('user:1', 'name', 'Alice')
name = r.hget('user:1', 'name')

# List
r.lpush('list', 'item1')
item = r.rpop('list')

# Set
r.sadd('set', 'member1')
members = r.smembers('set')
```

## Common SQL Problems

### Problem 1: Find Second Highest Salary
```sql
SELECT MAX(salary) as second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Using window function
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rnk
    FROM employees
) t
WHERE rnk = 2;
```

### Problem 2: Find Duplicate Emails
```sql
SELECT email, COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

### Problem 3: Employees Earning More Than Their Managers
```sql
SELECT e.name as employee
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;
```

### Problem 4: Department Highest Salary
```sql
SELECT d.name as department, e.name as employee, e.salary
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE (e.department_id, e.salary) IN (
    SELECT department_id, MAX(salary)
    FROM employees
    GROUP BY department_id
);
```

### Problem 5: Rank Scores
```sql
SELECT score,
       DENSE_RANK() OVER (ORDER BY score DESC) as rank
FROM scores;
```

### Problem 6: Consecutive Numbers
```sql
SELECT DISTINCT l1.num as ConsecutiveNums
FROM logs l1
JOIN logs l2 ON l1.id = l2.id - 1
JOIN logs l3 ON l1.id = l3.id - 2
WHERE l1.num = l2.num AND l2.num = l3.num;
```

## Performance Optimization

### 1. Query Optimization
- Use EXPLAIN to analyze query plans
- Add indexes on frequently queried columns
- Avoid SELECT *
- Use LIMIT when possible
- Optimize JOINs (proper indexes, join order)

### 2. Index Strategy
- Index foreign keys
- Index columns in WHERE, JOIN, ORDER BY
- Composite indexes for multi-column queries
- Don't over-index (slows writes)

### 3. Connection Management
- Use connection pooling
- Close connections properly
- Set appropriate timeout values
