/*

Dealers Table Requirements:

- registration (text of 5 positions)
- name (100)
- commission_percentage

*/

CREATE TABLE dealers (
    registration CHAR (5) NOT NULL PRIMARY KEY,
    full_name VARCHAR (100),
    commission_percentage float

)