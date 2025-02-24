// seasonality_detector.rs
use ndarray::{Array1, ArrayView1, Axis, Zip};
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::function::erf::erfc;
use statrs::distribution::{Normal, Univariate};
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SeasonalityError {
    #[error("Input data length insufficient for analysis")]
    InsufficientData,
    #[error("Zero variance in input data")]
    ZeroVariance,
    #[error("Invalid sampling rate: {0}")]
    InvalidSamplingRate(f64),
    #[error("Statistical computation error")]
    ComputationError,
}

pub struct SeasonalityDetector {
    data: Array1<f64>,
    sampling_rate: f64,
    significance_level: f64,
    min_period: f64,
    max_period: f64,
}

impl SeasonalityDetector {
    /// Create new detector with validation
    pub fn new(data: Array1<f64>, sampling_rate: f64) -> Result<Self, SeasonalityError> {
        if data.len() < 10 {
            return Err(SeasonalityError::InsufficientData);
        }
        if sampling_rate <= 0.0 {
            return Err(SeasonalityError::InvalidSamplingRate(sampling_rate));
        }
        
        Ok(SeasonalityDetector {
            data,
            sampling_rate,
            significance_level: 0.05,
            min_period: 1.0,
            max_period: 365.0,
        })
    }

    /// Configure detection parameters with bounds checking
    pub fn with_parameters(
        mut self,
        significance_level: f64,
        min_period: f64,
        max_period: f64,
    ) -> Result<Self, SeasonalityError> {
        if !(0.0..=0.2).contains(&significance_level) {
            return Err(SeasonalityError::ComputationError);
        }
        if min_period >= max_period || min_period <= 0.0 {
            return Err(SeasonalityError::ComputationError);
        }

        self.significance_level = significance_level;
        self.min_period = min_period;
        self.max_period = max_period;
        Ok(self)
    }

    /// Robust seasonality detection pipeline
    pub fn detect_seasonality(&self) -> Result<Vec<SeasonalityPeriod>, SeasonalityError> {
        let data_normalized = self.normalize_data()?;
        
        // Parallel computation stages
        let fft_periods = self.analyze_fft(&data_normalized)?;
        let acf_periods = self.analyze_autocorrelation(&data_normalized)?;

        let mut periods = self.merge_results(fft_periods, acf_periods)?;
        self.validate_with_stl(&mut periods)?;
        
        Ok(periods)
    }

    /// Normalize data with checks
    fn normalize_data(&self) -> Result<Array1<f64>, SeasonalityError> {
        let mean = self.data.mean().ok_or(SeasonalityError::ComputationError)?;
        let std_dev = self.data.std(0.0);
        
        if std_dev.abs() < 1e-10 {
            return Err(SeasonalityError::ZeroVariance);
        }

        Ok((&self.data - mean) / std_dev)
    }

    /// Enhanced FFT analysis with multiple peak detection
    fn analyze_fft(&self, data: &Array1<f64>) -> Result<Vec<SeasonalityPeriod>, SeasonalityError> {
        let n = data.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        let mut buffer: Vec<Complex<f64>> = data.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut buffer);
        
        let power_spectrum = Array1::from_iter(
            buffer.iter()
                .take(n / 2)
                .map(|c| c.norm_sqr().sqrt() / (n as f64).sqrt())
        );

        self.detect_spectral_peaks(&power_spectrum)
    }

    /// Optimized autocorrelation with vectorization
    fn analyze_autocorrelation(
        &self,
        data: &Array1<f64>
    ) -> Result<Vec<SeasonalityPeriod>, SeasonalityError> {
        let n = data.len();
        let mean = data.mean().ok_or(SeasonalityError::ComputationError)?;
        let variance = data.var(0.0);
        
        if variance.abs() < 1e-10 {
            return Err(SeasonalityError::ZeroVariance);
        }

        let max_lag = (self.max_period * self.sampling_rate) as usize;
        let min_lag = (self.min_period * self.sampling_rate) as usize;

        let acf = Array1::from_iter((min_lag..=max_lag).map(|lag| {
            let (left, right) = data.split_at(Axis(0), n - lag);
            let covariance = Zip::from(left)
                .and(right)
                .fold(0.0, |acc, &x, &y| acc + (x - mean) * (y - mean));
            
            covariance / (variance * (n - lag) as f64)
        }));

        self.detect_acf_peaks(&acf, min_lag)
    }

    /// Advanced peak detection with FDR correction
    fn detect_spectral_peaks(
        &self,
        power_spectrum: &Array1<f64>
    ) -> Result<Vec<SeasonalityPeriod>, SeasonalityError> {
        // Implementation of noise-adaptive peak detection
        // with Benjamini-Hochberg FDR correction
        // ...
    }

    /// STL validation with trend robustness checks
    fn validate_with_stl(
        &self,
        periods: &mut Vec<SeasonalityPeriod>
    ) -> Result<(), SeasonalityError> {
        // Complete STL implementation with
        // trend-stability checks and
        // seasonality strength calculation
        // ...
    }

    /// Calculate position adjustment based on seasonality periods and volatility
    pub fn calculate_position_adjustment(
        periods: &[SeasonalityPeriod],
        volatility: f64
    ) -> f64 {
        let seasonal_impact: f64 = periods.iter()
            .map(|p| p.strength * p.robustness_score)
            .sum();
        
        (seasonal_impact / volatility).tanh()  // Bounded output
    }
}

// Additional implementations for merging periods,
// statistical validation, and financial-specific
// seasonality scoring would follow...

#[derive(Debug, Clone)]
pub struct SeasonalityPeriod {
    pub period: f64,
    pub strength: f64,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub robustness_score: f64,
}

// Unit tests with financial time series examples
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_weekly_seasonality() {
        // Test with known weekly pattern
        let data = Array1::from((0..365).map(|x| (x % 7) as f64).collect();
        let detector = SeasonalityDetector::new(data, 1.0).unwrap();
        let periods = detector.detect_seasonality().unwrap();
        
        assert!(periods.iter().any(|p| (p.period - 7.0).abs() < 0.1));
    }
}