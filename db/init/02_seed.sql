-- Deterministic seed data (no random()), so evaluation gold answers are reproducible.

INSERT INTO categories (category_id, name, description) VALUES
    (1, 'Electronics', 'Phones, audio, and computer accessories'),
    (2, 'Books', 'Print and digital books'),
    (3, 'Home & Kitchen', 'Cookware, appliances, and home goods'),
    (4, 'Sports & Outdoors', 'Fitness and outdoor equipment'),
    (5, 'Toys', 'Toys and games for all ages'),
    (6, 'Beauty', 'Skincare and personal care'),
    (7, 'Grocery', 'Pantry staples and snacks'),
    (8, 'Office Supplies', 'Stationery and office equipment');

-- 80 products: 10 per category, unique names from (adjective, category noun).
INSERT INTO products (product_id, name, category_id, unit_price, stock_quantity, is_active, launched_on)
SELECT
    g,
    (ARRAY['Classic', 'Premium', 'Compact', 'Deluxe', 'Eco', 'Pro', 'Smart', 'Ultra', 'Mini', 'Max'])[((g - 1) / 8) + 1]
        || ' '
        || (ARRAY['Headphones', 'Novel', 'Blender', 'Yoga Mat', 'Puzzle', 'Face Serum', 'Granola', 'Notebook'])[((g - 1) % 8) + 1],
    ((g - 1) % 8) + 1,
    ROUND((5 + ((g * 37) % 200) + ((g * 13) % 100) / 100.0
        + CASE WHEN ((g - 1) % 8) + 1 = 1 THEN 150 ELSE 0 END)::NUMERIC, 2),
    (g * 53) % 500,
    g % 11 <> 0,
    DATE '2022-01-01' + ((g * 17) % 900)
FROM generate_series(1, 80) AS g;

-- 320 customers (orders only reference ids 1-300, so 20 customers never ordered).
INSERT INTO customers (customer_id, first_name, last_name, email, city, state, segment, signup_date)
SELECT
    g,
    fn,
    ln,
    LOWER(fn) || '.' || LOWER(ln) || g || '@example.com',
    (ARRAY['Seattle', 'Portland', 'San Francisco', 'Los Angeles', 'San Diego', 'Austin',
           'Houston', 'Chicago', 'New York', 'Boston', 'Denver', 'Atlanta'])[((g * 5) % 12) + 1],
    (ARRAY['WA', 'OR', 'CA', 'CA', 'CA', 'TX', 'TX', 'IL', 'NY', 'MA', 'CO', 'GA'])[((g * 5) % 12) + 1],
    CASE WHEN g % 10 < 6 THEN 'consumer' WHEN g % 10 < 9 THEN 'small_business' ELSE 'enterprise' END,
    DATE '2023-01-01' + ((g * 29) % 700)
FROM (
    SELECT
        g,
        (ARRAY['Olivia', 'Liam', 'Emma', 'Noah', 'Ava', 'Ethan', 'Sophia', 'Mason', 'Mia', 'Lucas',
               'Isabella', 'Logan', 'Amelia', 'James', 'Harper', 'Benjamin', 'Evelyn', 'Elijah', 'Abigail', 'Henry'])[((g * 7) % 20) + 1] AS fn,
        (ARRAY['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
               'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin'])[((g * 11) % 20) + 1] AS ln
    FROM generate_series(1, 320) AS g
) AS names;

-- 30 employees: ids 1-4 are department heads; everyone else reports to their department head.
-- Department is chosen so that employee_id % 4 = 1 -> Sales, 0 -> Support, 2 -> Engineering, 3 -> Operations.
INSERT INTO employees (employee_id, first_name, last_name, department, title, hire_date, salary, manager_id)
SELECT
    g,
    (ARRAY['Grace', 'Daniel', 'Chloe', 'Samuel', 'Zoe', 'David', 'Lily', 'Owen', 'Nora', 'Jack'])[(g % 10) + 1],
    (ARRAY['Clark', 'Lewis', 'Walker', 'Hall', 'Allen', 'Young', 'King', 'Wright', 'Scott', 'Green',
           'Baker', 'Adams', 'Nelson', 'Hill', 'Campbell'])[((g * 7) % 15) + 1],
    (ARRAY['Support', 'Sales', 'Engineering', 'Operations'])[(g % 4) + 1],
    CASE
        WHEN g <= 4 THEN 'Director'
        WHEN g % 3 = 0 THEN 'Associate'
        WHEN g % 3 = 1 THEN 'Specialist'
        ELSE 'Senior Specialist'
    END,
    DATE '2018-01-01' + ((g * 97) % 2000),
    CASE WHEN g <= 4 THEN 150000 + g * 5000 ELSE 55000 + ROUND(((g * 1373) % 60000) / 100.0) * 100 END,
    CASE WHEN g <= 4 THEN NULL WHEN g % 4 = 0 THEN 4 ELSE g % 4 END
FROM generate_series(1, 30) AS g;

-- 3000 orders. Customer ids are skewed toward low ids (some customers never order);
-- order dates never precede the customer's signup date.
INSERT INTO orders (order_id, customer_id, sales_rep_id, order_date, status, shipping_cost)
SELECT
    o.g,
    c.customer_id,
    CASE WHEN o.g % 9 = 0 THEN NULL ELSE 1 + 4 * ((o.g * 3) % 8) END,
    GREATEST(DATE '2024-01-01' + ((o.g * 11) % 640), c.signup_date + 1),
    CASE
        WHEN (o.g * 7) % 20 = 0 THEN 'cancelled'
        WHEN (o.g * 7) % 20 = 1 THEN 'returned'
        WHEN (o.g * 7) % 20 IN (2, 3) THEN 'pending'
        WHEN (o.g * 7) % 20 BETWEEN 4 AND 8 THEN 'shipped'
        ELSE 'delivered'
    END,
    CASE WHEN o.g % 5 = 0 THEN 0 ELSE 4.99 + (o.g % 4) * 2 END
FROM generate_series(1, 3000) AS o(g)
JOIN customers c
    ON c.customer_id = LEAST(300, 1 + FLOOR(300 * POWER(((o.g * 7919) % 3000) / 3000.0, 2))::INT);

-- 1-4 line items per order, priced at the product's list price.
INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount)
SELECT
    o.order_id,
    p.product_id,
    1 + ((o.order_id + s * 3) % 5),
    p.unit_price,
    CASE
        WHEN (o.order_id + s) % 7 = 0 THEN 0.10
        WHEN (o.order_id + s) % 11 = 0 THEN 0.20
        ELSE 0
    END
FROM orders o
CROSS JOIN generate_series(1, 4) AS s
JOIN products p ON p.product_id = ((o.order_id * 13 + s * 29) % 80) + 1
WHERE s <= 1 + (o.order_id % 4)
ORDER BY o.order_id, s;

-- 600 support tickets assigned to Support employees (ids 4, 8, ..., 28).
INSERT INTO support_tickets (ticket_id, customer_id, assigned_employee_id, order_id, priority, status, created_at, resolved_at)
SELECT
    g,
    ((g * 53) % 300) + 1,
    CASE WHEN g % 25 = 0 THEN NULL ELSE 4 + 4 * ((g * 5) % 7) END,
    CASE WHEN g % 3 = 0 THEN NULL ELSE ((g * 101) % 3000) + 1 END,
    (ARRAY['low', 'medium', 'high', 'urgent'])[((g * 3) % 4) + 1],
    status,
    created_at,
    CASE WHEN status IN ('resolved', 'closed') THEN created_at + (((g * 7) % 96) + 1) * INTERVAL '1 hour' END
FROM (
    SELECT
        g,
        CASE
            WHEN g % 10 IN (0, 1) THEN 'open'
            WHEN g % 10 = 2 THEN 'in_progress'
            WHEN g % 10 = 3 THEN 'closed'
            ELSE 'resolved'
        END AS status,
        TIMESTAMP '2024-01-01 08:00' + ((g * 13) % 640) * INTERVAL '1 day' + ((g * 37) % 600) * INTERVAL '1 minute' AS created_at
    FROM generate_series(1, 600) AS g
) AS t;

SELECT setval(pg_get_serial_sequence('categories', 'category_id'), (SELECT MAX(category_id) FROM categories));
SELECT setval(pg_get_serial_sequence('products', 'product_id'), (SELECT MAX(product_id) FROM products));
SELECT setval(pg_get_serial_sequence('customers', 'customer_id'), (SELECT MAX(customer_id) FROM customers));
SELECT setval(pg_get_serial_sequence('employees', 'employee_id'), (SELECT MAX(employee_id) FROM employees));
SELECT setval(pg_get_serial_sequence('orders', 'order_id'), (SELECT MAX(order_id) FROM orders));
SELECT setval(pg_get_serial_sequence('order_items', 'order_item_id'), (SELECT MAX(order_item_id) FROM order_items));
SELECT setval(pg_get_serial_sequence('support_tickets', 'ticket_id'), (SELECT MAX(ticket_id) FROM support_tickets));

ANALYZE;
