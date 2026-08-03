use std::collections::BTreeMap;

use crate::{Error, Result};

/// Build an OpenAI-compatible endpoint from a base URL, path, and default
/// query parameters.  Every provider-facing subsystem uses this helper so an
/// Azure `api-version`, for example, cannot be accidentally omitted by media
/// endpoints.
pub fn endpoint_url(
    base_url: &str,
    path: &str,
    query: &BTreeMap<String, String>,
) -> Result<String> {
    let base_url = base_url.trim().trim_end_matches('/');
    if base_url.is_empty() {
        return Err(Error::MissingBaseUrl("request context".into()));
    }
    let (path, existing_query) = path
        .trim_start_matches('/')
        .split_once('?')
        .unwrap_or((path, ""));
    let mut url = format!("{base_url}/{path}");
    if !existing_query.is_empty() {
        url.push('?');
        url.push_str(existing_query);
    }
    if !query.is_empty() {
        let encoded = query
            .iter()
            .map(|(key, value)| {
                format!(
                    "{}={}",
                    percent_encoding::utf8_percent_encode(key, percent_encoding::NON_ALPHANUMERIC),
                    percent_encoding::utf8_percent_encode(
                        value,
                        percent_encoding::NON_ALPHANUMERIC
                    ),
                )
            })
            .collect::<Vec<_>>()
            .join("&");
        url.push(if existing_query.is_empty() { '?' } else { '&' });
        url.push_str(&encoded);
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::endpoint_url;

    #[test]
    fn merges_default_query_with_an_endpoint_query() {
        let query = BTreeMap::from([("api-version".into(), "2025-04-01-preview".into())]);
        assert_eq!(
            endpoint_url(
                "https://example.openai.azure.com/openai/v1/",
                "videos/example/content?variant=video",
                &query,
            )
            .unwrap(),
            "https://example.openai.azure.com/openai/v1/videos/example/content?variant=video&api%2Dversion=2025%2D04%2D01%2Dpreview",
        );
    }
}
