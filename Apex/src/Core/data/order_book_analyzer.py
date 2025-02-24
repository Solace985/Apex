class OrderBookImbalance:
    def calculate_imbalance(self, bids, asks):
        """
        Calculates order book imbalance between bids & asks.
        """
        top_bid_qty = bids[0][1]
        top_ask_qty = asks[0][1]
        return (top_bid_qty - top_ask_qty) / (top_bid_qty + top_ask_qty)

    def detect_liquidity_pools(self, order_book):
        """
        Identifies key liquidity areas based on order book depth.
        """
        significant_levels = []
        for price, qty in order_book['bids'][:5]:
            if qty > 2 * order_book['avg_bid_qty']:
                significant_levels.append(('support', price))
        return significant_levels
