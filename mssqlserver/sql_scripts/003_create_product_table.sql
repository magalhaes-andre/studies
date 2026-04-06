/*

Product Table Definition:

- code
- name
- package
- size
- flavour
- listing_price

*/

CREATE TABLE products (

    code VARCHAR(50) NOT NULL PRIMARY KEY,
    product_name VARCHAR(150),
    package CHAR (1),
    product_size VARCHAR (5),
    flavour VARCHAR (100),
    listing_price SMALLMONEY
)