CREATE TRIGGER UpdateItems
AFTER INSERT ON orders
FOR EACH ROW
	UPDATE items
	SET items.quantity = items.quantity - NEW.number
	WHERE items.name = NEW.item_name;
