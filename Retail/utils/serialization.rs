use serde_json::Error;

pub fn parse_json<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, Error> {
    serde_json::from_str(input)
}
