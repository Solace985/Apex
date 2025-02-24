# SEC Form 4/5 analysis with NLP
class InsiderMonitor:
    def analyze_form4(self, filing):
        """Detect clustered insider activity"""
        entities = self.ner_model.extract(filing.text)
        cluster_score = self.dbscan.cluster([
            filing.executive_name,
            filing.trade_size,
            filing.price
        ])
        
        if cluster_score > 0.9:
            alert_portfolio_manager(entities)

    def realtime_alerts(self):
        # Monitor SEC EDGAR API
        while True:
            new_filings = sec_api.poll()
            for filing in new_filings:
                thread.start_new_thread(self.analyze_form4, (filing,))