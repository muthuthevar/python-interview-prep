-- ============================================================================
-- SQL PRACTICE - Most Common Interview Questions
-- ============================================================================

-- ============================================================================
-- PROBLEM 1: Find Second Highest Salary
-- ============================================================================
SELECT MAX(salary) as second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Using window function (better approach)
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rnk
    FROM employees
) t
WHERE rnk = 2;

-- ============================================================================
-- PROBLEM 2: Find Duplicate Emails
-- ============================================================================
SELECT email, COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- ============================================================================
-- PROBLEM 3: Employees Earning More Than Their Managers
-- ============================================================================
SELECT e.name as employee
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;

-- ============================================================================
-- PROBLEM 4: Department Highest Salary
-- ============================================================================
SELECT d.name as department, e.name as employee, e.salary
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE (e.department_id, e.salary) IN (
    SELECT department_id, MAX(salary)
    FROM employees
    GROUP BY department_id
);

-- ============================================================================
-- PROBLEM 5: Rank Scores
-- ============================================================================
SELECT score,
       DENSE_RANK() OVER (ORDER BY score DESC) as rank
FROM scores;

-- ============================================================================
-- PROBLEM 6: Consecutive Numbers
-- ============================================================================
SELECT DISTINCT l1.num as ConsecutiveNums
FROM logs l1
JOIN logs l2 ON l1.id = l2.id - 1
JOIN logs l3 ON l1.id = l3.id - 2
WHERE l1.num = l2.num AND l2.num = l3.num;

-- ============================================================================
-- PROBLEM 7: Customers Who Never Order
-- ============================================================================
SELECT c.name as Customers
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;

-- ============================================================================
-- PROBLEM 8: Delete Duplicate Emails
-- ============================================================================
DELETE p1 FROM person p1
INNER JOIN person p2
WHERE p1.id > p2.id AND p1.email = p2.email;

-- ============================================================================
-- PROBLEM 9: Rising Temperature
-- ============================================================================
SELECT w1.id
FROM weather w1
JOIN weather w2 ON DATEDIFF(w1.recordDate, w2.recordDate) = 1
WHERE w1.temperature > w2.temperature;

-- ============================================================================
-- PROBLEM 10: Nth Highest Salary
-- ============================================================================
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      SELECT DISTINCT salary
      FROM (
          SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rnk
          FROM employee
      ) t
      WHERE rnk = N
  );
END;

-- ============================================================================
-- PROBLEM 11: Top 3 Salaries Per Department
-- ============================================================================
SELECT d.name as Department, e.name as Employee, e.salary as Salary
FROM employee e
JOIN department d ON e.departmentId = d.id
WHERE (
    SELECT COUNT(DISTINCT e2.salary)
    FROM employee e2
    WHERE e2.departmentId = e.departmentId
    AND e2.salary > e.salary
) < 3;

-- ============================================================================
-- PROBLEM 12: Find Median Salary
-- ============================================================================
SELECT AVG(salary) as median_salary
FROM (
    SELECT salary,
           ROW_NUMBER() OVER (ORDER BY salary) as rn,
           COUNT(*) OVER () as total
    FROM employees
) t
WHERE rn IN (FLOOR((total + 1) / 2), CEIL((total + 1) / 2));

-- ============================================================================
-- PROBLEM 13: Employees With Missing Information
-- ============================================================================
SELECT COALESCE(e.employee_id, s.employee_id) as employee_id
FROM employees e
FULL OUTER JOIN salaries s ON e.employee_id = s.employee_id
WHERE e.name IS NULL OR s.salary IS NULL
ORDER BY employee_id;

-- ============================================================================
-- PROBLEM 14: Exchange Seats
-- ============================================================================
SELECT 
    CASE 
        WHEN id % 2 = 0 THEN id - 1
        WHEN id = (SELECT MAX(id) FROM seat) THEN id
        ELSE id + 1
    END as id,
    student
FROM seat
ORDER BY id;

-- ============================================================================
-- PROBLEM 15: Department Top Three Salaries
-- ============================================================================
WITH RankedSalaries AS (
    SELECT 
        e.name as Employee,
        e.salary as Salary,
        d.name as Department,
        DENSE_RANK() OVER (PARTITION BY e.departmentId ORDER BY e.salary DESC) as rnk
    FROM employee e
    JOIN department d ON e.departmentId = d.id
)
SELECT Department, Employee, Salary
FROM RankedSalaries
WHERE rnk <= 3;

-- ============================================================================
-- COMMON PATTERNS
-- ============================================================================

-- Self Join Pattern
SELECT a.column1, b.column2
FROM table a
JOIN table b ON a.id = b.related_id;

-- Window Functions
SELECT 
    column1,
    ROW_NUMBER() OVER (PARTITION BY column2 ORDER BY column3) as rn,
    RANK() OVER (ORDER BY column3 DESC) as rank,
    DENSE_RANK() OVER (ORDER BY column3 DESC) as dense_rank,
    LAG(column1) OVER (ORDER BY column3) as prev_value,
    LEAD(column1) OVER (ORDER BY column3) as next_value,
    SUM(column1) OVER (PARTITION BY column2) as total,
    AVG(column1) OVER (PARTITION BY column2) as avg_value
FROM table;

-- Common Table Expression (CTE)
WITH cte AS (
    SELECT column1, column2
    FROM table1
    WHERE condition
)
SELECT * FROM cte;

-- Recursive CTE
WITH RECURSIVE hierarchy AS (
    -- Base case
    SELECT id, name, parent_id, 1 as level
    FROM employees
    WHERE parent_id IS NULL
    
    UNION ALL
    
    -- Recursive case
    SELECT e.id, e.name, e.parent_id, h.level + 1
    FROM employees e
    JOIN hierarchy h ON e.parent_id = h.id
)
SELECT * FROM hierarchy;

-- ============================================================================
-- SQLALCHEMY EQUIVALENTS (Python)
-- ============================================================================

-- Basic Query
-- SQL: SELECT * FROM users WHERE age > 25
users = session.query(User).filter(User.age > 25).all()

-- Join
-- SQL: SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id
result = session.query(User.name, Order.total).join(Order).all()

-- Aggregation
-- SQL: SELECT department, AVG(salary) FROM employees GROUP BY department
from sqlalchemy import func
result = session.query(
    Employee.department,
    func.avg(Employee.salary)
).group_by(Employee.department).all()

-- Subquery
-- SQL: SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)
subquery = session.query(Order.user_id).subquery()
users = session.query(User).filter(User.id.in_(subquery)).all()

-- Window Function (using func)
from sqlalchemy import func
result = session.query(
    Employee.name,
    Employee.salary,
    func.rank().over(order_by=Employee.salary.desc()).label('rank')
).all()
