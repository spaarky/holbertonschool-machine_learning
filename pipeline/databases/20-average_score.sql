-- comment
DELIMITER $$
CREATE PROCEDURE ComputeAverageScoreForUser (
	IN new_id INT
)
BEGIN
	UPDATE users
	SET average_score=(
		SELECT AVG(score)
		FROM corrections
		WHERE user_id = new_id)
	WHERE id = new_id;
END $$
DELIMITER ;
