import os
import time
import logging
from typing import Any, Dict, Optional

import requests
import yaml
import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(BASE_DIR))
CREDENTIALS_FILE: str = os.path.join(PROJECT_ROOT, "credentials", "api.yml")


def load_yaml(filename: str) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents.
    """
    try:
        with open(filename, "r") as file:
            config = yaml.safe_load(file)
        return config or {}
    except FileNotFoundError as e:
        logger.error(f"File '{filename}' not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{filename}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unknown error loading YAML file '{filename}': {e}")
        raise


def get_google_api_key(config: Dict[str, Any]) -> str:
    """
    Extract the Google API key from the configuration.
    """
    api_key = config.get("GOOGLE_API_KEY", "")
    if not api_key:
        logger.error("Google API key is missing in the configuration.")
        raise ValueError("Google API key not found in the configuration.")
    return api_key


def follow_redirect(
    url: str,
    max_retries: int = 5,
    backoff_factor: float = 1.0,
    timeout: float = 10.0
) -> Optional[str]:
    """
    Attempt to fetch the final redirect URL for the given link.
    Retries up to `max_retries` times with exponential backoff 
    if there's a request error.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, allow_redirects=True, timeout=timeout)
            logger.info(f"Successfully followed URL: {url} -> {response.url}")
            return response.url
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Attempt {attempt + 1} of {max_retries} failed for URL '{url}': {e}"
            )
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                logger.info(f"Sleeping for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached. Could not resolve URL: {url}")
                return None
    return None


def map_chunk_confidences(grounding_metadata: Any) -> Dict[int, float]:
    """
    Parse grounding_supports to create a mapping from chunk index -> confidence score.
    If multiple supports refer to the same chunk, use the maximum confidence score.
    """
    chunk_confidences = {}
    if not grounding_metadata:
        return chunk_confidences

    supports = getattr(grounding_metadata, "grounding_supports", [])
    for support in supports:
        indices = getattr(support, "grounding_chunk_indices", [])
        scores = getattr(support, "confidence_scores", [])
        for i, chunk_idx in enumerate(indices):
            current_score = scores[i] if i < len(scores) else 0.0
            if chunk_idx not in chunk_confidences:
                chunk_confidences[chunk_idx] = current_score
            else:
                chunk_confidences[chunk_idx] = max(
                    chunk_confidences[chunk_idx], current_score
                )
    return chunk_confidences


def run_query(query: str) -> Dict[str, Any]:
    """
    Execute a query against the Gemini model with google_search_retrieval enabled.

    Returns a dictionary:
    {
      "original_query": str,
      "results": [
         {
           "link": str,
           "title": str,
           "confidence_score": float
         },
         ...
      ]
    }
    """
    try:
        # Load config and get API key (only do once in a real app; done every time here for clarity)
        config = load_yaml(CREDENTIALS_FILE)
        api_key = get_google_api_key(config)
        genai.configure(api_key=api_key)

        # Model name can vary
        model = genai.GenerativeModel("models/gemini-1.5-pro-002")

        response = model.generate_content(
            contents=query,
            tools="google_search_retrieval"
        )

        # Gather results in a list
        results = []

        # We only handle the first candidate for simplicity
        if not response.candidates:
            logger.info("No candidates returned from model.")
            return {"original_query": query, "results": results}

        candidate = response.candidates[0]

        # If you want to inspect the model text response
        # if candidate.content and candidate.content.parts:
        #    logger.info("Model response text:")
        #    logger.info(candidate.content.parts[0].text)

        # Extract grounding chunks if present
        grounding_metadata = getattr(candidate, "grounding_metadata", None)
        grounding_chunks = getattr(grounding_metadata, "grounding_chunks", [])

        # Map chunk index -> confidence score
        chunk_confidences = map_chunk_confidences(grounding_metadata)

        for idx, chunk in enumerate(grounding_chunks):
            # Each chunk might be something like chunk.web.uri, chunk.web.title, etc.
            web_obj = getattr(chunk, "web", None)
            if not web_obj:
                continue

            uri = getattr(web_obj, "uri", None)
            title = getattr(web_obj, "title", "")
            if not uri:
                continue

            final_link = follow_redirect(uri)
            confidence_score = chunk_confidences.get(idx, 0.0)

            results.append({
                "link": final_link or "",
                "title": title,
                "confidence_score": round(confidence_score, 6)  # or keep full precision
            })

        # Return the final JSON-like structure
        return {
            "original_query": query,
            "results": results
        }

    except Exception as e:
        logger.exception("Error occurred in run_query.")
        return {
            "original_query": query,
            "results": [],
            "error": str(e)
        }
