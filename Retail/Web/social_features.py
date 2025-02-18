class SocialTrader:  
    def __init__(self):  
        self.leaderboard = {}  

    def update_leaderboard(self, user_id: str, profit: float):  
        self.leaderboard[user_id] = self.leaderboard.get(user_id, 0.0) + profit  
        # Publish to Redis pub/sub  
        redis_client.publish("leaderboard", json.dumps(self.leaderboard))  