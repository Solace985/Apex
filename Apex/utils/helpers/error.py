// error.rs
#[derive(Debug)]
pub enum CorrelationError {
    DataUnavailable(String),
    ComputationError(String),
    AiServiceError(ReqwestError),
    StorageError(DatabaseError),
    InvalidInput(String),
}

impl From<ReqwestError> for CorrelationError {
    fn from(err: ReqwestError) -> Self {
        Self::AiServiceError(err)
    }
}