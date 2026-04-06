/* 
    GO here is used to send commands in batches, evicting that anything that needs
    to run sequentially tries to do something before previous commands
    are actually commited.
*/
ALTER TABLE clients ALTER COLUMN document CHAR (11) NOT NULL;
GO


ALTER TABLE clients ADD CONSTRAINT pk_clients PRIMARY KEY CLUSTERED (document)
GO