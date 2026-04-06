SELECT TOP 5 * FROM products; 

SELECT code, package, product_size from products ORDER BY product_name;

SELECT document AS client_code FROM clients;

SELECT * FROM products WHERE listing_price >=3.5;

SELECT * FROM products WHERE package <> 'C'

SELECT full_name, birth_date FROM clients WHERE YEAR(birth_date) >= '1800';

SELECT * FROM dealers WHERE commission_percentage > 0.08;
SELECT * FROM products WHERE listing_price < 6.00 AND package = 'C';
SELECT * FROM clients WHERE state_code <> 'RS' AND state_code <> 'SC';
SELECT * FROM clients WHERE state_code NOT IN ('RS','SC');