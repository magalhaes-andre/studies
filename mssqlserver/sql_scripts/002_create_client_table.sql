/*
Client Table definition:

Columns:
- document
- fullName
- address
- birthDate
- age
- gender
- creditLimit
- minimalBoughtVolume
- hasBoughtInTheCompany
*/

CREATE TABLE clients(
    document CHAR (11),
    full_name VARCHAR (150),
    address_line1 VARCHAR (150),
    address_line2 VARCHAR (150),
    neighbourhood VARCHAR (150),
    city VARCHAR (150),
    state_code VARCHAR (5),
    postal_code VARCHAR (10),
    birth_date DATE,
    age SMALLINT,
    gender CHAR (1),
    credit_limit MONEY,
    minimal_volume FLOAT,
    has_bought BIT
)