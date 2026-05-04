import logging
import json
logger = logging.getLogger(__name__)
from typing import Any, Optional


def _extract_json_content(response: str) -> str:
    """Extract the most likely JSON object/array from an LLM response."""
    text = response.strip()

    fence_marker = "```"
    if fence_marker in text:
        parts = text.split(fence_marker)
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith(("{", "[")):
                return candidate

    for opening, closing in (("{", "}"), ("[", "]")):
        start = text.find(opening)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

    return text


def parse_llm_response(response: str) -> Optional[Any]:
    """
    Parse the LLM response that contains JSON data.
    
    Args:
        response: Raw response string from LLM containing JSON
        
    Returns:
        Parsed JSON data as a list of dictionaries, or None if parsing fails
    """
    try:
        json_content = _extract_json_content(response)
        # print(json_content)
        json_content = json_content.replace("None", "null")
        # Parse the JSON content
        output_llm = json.loads(json_content)

        return output_llm
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")

        # logger.error(f"Raw response: {response}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing response: {str(e)}")
        return None
    
def parse_search_results(res,short_answer_only=False):
    short_answer = ''
    if 'answerBox' in res:
        short_answer = res['answerBox']['snippet']
        #short_answer = res['answerBox']['snippetHighlighted']
    reference = ""
    for id, item in enumerate(res['organic']):
        try:
            reference += f"Reference {id}: {item['snippet']}\n"
        except KeyError:
            print(f"KeyError: {item}")
            continue
    if short_answer_only:
        if short_answer =='':
            return reference
        else:
            return short_answer
    else:
        return f"{short_answer}\n{reference}"
