UPDATE products 
SET package = 'B'
WHERE code = '1040107';

UPDATE products
SET listing_price = listing_price * 0.15
WHERE package = 'C';

UPDATE products
SET product_size = '600ml'
WHERE package = 'B';