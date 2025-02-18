pub async fn update_correlations(&self) {  
    let assets = self.load_asset_universe();  
    let pairs = generate_asset_pairs(&assets);  

    stream::iter(pairs)  
        .for_each_concurrent(50, |(a1, a2)| async move {  
            let corr = calculate_rolling_corr(a1, a2, 24.hours()).await;  
            if corr.abs() > 0.6 {  
                self.correlation_graph.update(a1, a2, corr);  
            }  
        })  
        .await;  
}  