-- Demo e-commerce schema used by docker-compose, Kubernetes, integration tests, and the evaluation set.

CREATE TABLE categories (
    category_id  SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    description  TEXT
);
COMMENT ON TABLE categories IS 'Product categories in the catalog.';

CREATE TABLE products (
    product_id      SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    category_id     INT NOT NULL REFERENCES categories (category_id),
    unit_price      NUMERIC(10, 2) NOT NULL CHECK (unit_price > 0),
    stock_quantity  INT NOT NULL CHECK (stock_quantity >= 0),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    launched_on     DATE NOT NULL
);
COMMENT ON TABLE products IS 'Products for sale. unit_price is the current list price in USD.';

CREATE TABLE customers (
    customer_id  SERIAL PRIMARY KEY,
    first_name   TEXT NOT NULL,
    last_name    TEXT NOT NULL,
    email        TEXT NOT NULL UNIQUE,
    city         TEXT NOT NULL,
    state        TEXT NOT NULL,
    segment      TEXT NOT NULL CHECK (segment IN ('consumer', 'small_business', 'enterprise')),
    signup_date  DATE NOT NULL
);
COMMENT ON TABLE customers IS 'Registered customers. state is a two-letter US state code.';

CREATE TABLE employees (
    employee_id  SERIAL PRIMARY KEY,
    first_name   TEXT NOT NULL,
    last_name    TEXT NOT NULL,
    department   TEXT NOT NULL CHECK (department IN ('Support', 'Sales', 'Engineering', 'Operations')),
    title        TEXT NOT NULL,
    hire_date    DATE NOT NULL,
    salary       NUMERIC(10, 2) NOT NULL,
    manager_id   INT REFERENCES employees (employee_id)
);
COMMENT ON TABLE employees IS 'Company employees. manager_id references the employee''s manager; department heads have no manager.';

CREATE TABLE orders (
    order_id       SERIAL PRIMARY KEY,
    customer_id    INT NOT NULL REFERENCES customers (customer_id),
    sales_rep_id   INT REFERENCES employees (employee_id),
    order_date     DATE NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled', 'returned')),
    shipping_cost  NUMERIC(8, 2) NOT NULL DEFAULT 0
);
COMMENT ON TABLE orders IS 'Customer orders. sales_rep_id is the Sales employee credited with the order. Order totals are not stored; compute them from order_items.';

CREATE TABLE order_items (
    order_item_id  SERIAL PRIMARY KEY,
    order_id       INT NOT NULL REFERENCES orders (order_id),
    product_id     INT NOT NULL REFERENCES products (product_id),
    quantity       INT NOT NULL CHECK (quantity > 0),
    unit_price     NUMERIC(10, 2) NOT NULL,
    discount       NUMERIC(4, 2) NOT NULL DEFAULT 0 CHECK (discount >= 0 AND discount < 1)
);
COMMENT ON TABLE order_items IS 'Order line items. unit_price is the price at purchase time; discount is a fraction (0.10 = 10% off). Line revenue = quantity * unit_price * (1 - discount).';

CREATE TABLE support_tickets (
    ticket_id             SERIAL PRIMARY KEY,
    customer_id           INT NOT NULL REFERENCES customers (customer_id),
    assigned_employee_id  INT REFERENCES employees (employee_id),
    order_id              INT REFERENCES orders (order_id),
    priority              TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    status                TEXT NOT NULL CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
    created_at            TIMESTAMP NOT NULL,
    resolved_at           TIMESTAMP
);
COMMENT ON TABLE support_tickets IS 'Customer support tickets handled by Support employees. resolved_at is set once a ticket is resolved or closed.';

CREATE INDEX idx_products_category ON products (category_id);
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_sales_rep ON orders (sales_rep_id);
CREATE INDEX idx_orders_date ON orders (order_date);
CREATE INDEX idx_order_items_order ON order_items (order_id);
CREATE INDEX idx_order_items_product ON order_items (product_id);
CREATE INDEX idx_tickets_customer ON support_tickets (customer_id);
CREATE INDEX idx_tickets_employee ON support_tickets (assigned_employee_id);
