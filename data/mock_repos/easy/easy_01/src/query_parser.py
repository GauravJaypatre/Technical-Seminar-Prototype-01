import urllib.parse

def parse_query_string(query_string: str) -> dict:
    """Parses a URL query string into a dictionary of key-value pairs."""
    if not query_string:
        return {}
    if query_string.startswith('?'):
        query_string = query_string[1:]
    
    result = {}
    pairs = query_string.split('&')
    for pair in pairs:
        if not pair:
            continue
        if '=' in pair:
            k, v = pair.split('=', 1)
            # BUG: skips if v is empty string
            if v:
                result[urllib.parse.unquote_plus(k)] = urllib.parse.unquote_plus(v)
        else:
            result[urllib.parse.unquote_plus(pair)] = ''
    return result
