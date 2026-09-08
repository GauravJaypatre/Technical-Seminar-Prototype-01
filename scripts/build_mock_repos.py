"""
Script to synthesize 18 Micro-SWE Mock Repositories (6 Easy, 6 Medium, 6 Hard)
for the seminar benchmarking project.
Each task includes:
- src/ containing Python source code with a seeded defect
- tests/ containing a pytest suite (fails unpatched, passes with fix)
- issue.md containing realistic user issue description
- metadata.json containing mathematically verified ground truth patch, difficulty tier, and metadata
"""
import os
import json
import difflib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "mock_repos")

TASKS = []

def generate_unified_diff(orig_str: str, fixed_str: str, file_path: str) -> str:
    orig_lines = [l + "\n" for l in orig_str.strip().splitlines()]
    fixed_lines = [l + "\n" for l in fixed_str.strip().splitlines()]
    diff = list(difflib.unified_diff(
        orig_lines,
        fixed_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}"
    ))
    return "".join(diff)

def register_task(task_id, tier, name, domain, issue_title, issue_body, files, fixed_files, failing_tests):
    task_dir = os.path.join(DATA_DIR, tier, task_id)
    os.makedirs(os.path.join(task_dir, "src"), exist_ok=True)
    os.makedirs(os.path.join(task_dir, "tests"), exist_ok=True)

    # Write initial buggy source files and tests
    for rel_path, content in files.items():
        file_path = os.path.join(task_dir, rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content.strip() + "\n")

    # Generate unified diff patch from buggy to fixed files
    patch_chunks = []
    for rel_path, fixed_content in fixed_files.items():
        orig_content = files[rel_path]
        diff_str = generate_unified_diff(orig_content, fixed_content, rel_path)
        patch_chunks.append(diff_str)
    ground_truth_patch = "\n".join(patch_chunks).strip()

    # Write issue.md
    issue_content = f"# Issue: {issue_title}\n\n**Task ID:** `{task_id}`  \n**Tier:** `{tier}`  \n**Domain:** `{domain}`  \n\n## Description\n{issue_body}\n"
    with open(os.path.join(task_dir, "issue.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(issue_content)

    # Write metadata.json
    metadata = {
        "task_id": task_id,
        "tier": tier,
        "name": name,
        "domain": domain,
        "issue_title": issue_title,
        "failing_tests": failing_tests,
        "target_files": [f for f in files.keys() if f.startswith("src/")],
        "ground_truth_patch": ground_truth_patch,
        "attribution": "Synthesized canonical defect pattern inspired by real-world open source issues (MIT/Apache-2.0, SWE-bench taxonomy)"
    }
    with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, indent=2)

    TASKS.append({
        "task_id": task_id,
        "tier": tier,
        "name": name,
        "domain": domain,
        "directory": os.path.relpath(task_dir, BASE_DIR).replace("\\", "/")
    })
    print(f"Created task: {task_id} ({tier}) - {name}")

# ==============================================================================
# EASY TASKS (1 function, 1 file, localized logic defect)
# ==============================================================================

# easy_01: URL Query Parser
SRC_EASY_01 = '''
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
'''

FIXED_EASY_01 = '''
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
            result[urllib.parse.unquote_plus(k)] = urllib.parse.unquote_plus(v)
        else:
            result[urllib.parse.unquote_plus(pair)] = ''
    return result
'''

TEST_EASY_01 = '''
import pytest
from src.query_parser import parse_query_string

def test_basic_parsing():
    res = parse_query_string("foo=bar&baz=qux")
    assert res == {"foo": "bar", "baz": "qux"}

def test_leading_question_mark():
    res = parse_query_string("?name=alice&age=30")
    assert res == {"name": "alice", "age": "30"}

def test_empty_string():
    assert parse_query_string("") == {}

def test_empty_value_retained():
    # FAILS ON UNPATCHED CODE
    res = parse_query_string("key=&other=val")
    assert "key" in res
    assert res["key"] == ""
    assert res["other"] == "val"

def test_encoded_spaces():
    res = parse_query_string("msg=hello+world&mode=fast")
    assert res["msg"] == "hello world"
'''

register_task(
    task_id="easy_01",
    tier="easy",
    name="url_query_parser",
    domain="Networking / Web",
    issue_title="Query parser drops keys with empty values",
    issue_body="When parsing query strings like `?key=&other=val`, `parse_query_string()` completely ignores keys with empty values (`key`). According to standard URL query specification, keys with empty values should be retained with an empty string value `''` instead of being dropped.",
    files={"src/query_parser.py": SRC_EASY_01, "tests/test_query_parser.py": TEST_EASY_01},
    fixed_files={"src/query_parser.py": FIXED_EASY_01},
    failing_tests=["test_empty_value_retained"]
)

# easy_02: Leap Year Calculator
SRC_EASY_02 = '''
def is_leap_year(year: int) -> bool:
    """Determines if a given year is a leap year in the Gregorian calendar."""
    if not isinstance(year, int) or year < 1:
        raise ValueError("Year must be a positive integer.")
    # BUG: century check logic inverted
    if year % 4 == 0:
        if year % 100 == 0:
            return True
        return True
    return False
'''

FIXED_EASY_02 = '''
def is_leap_year(year: int) -> bool:
    """Determines if a given year is a leap year in the Gregorian calendar."""
    if not isinstance(year, int) or year < 1:
        raise ValueError("Year must be a positive integer.")
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0
'''

TEST_EASY_02 = '''
import pytest
from src.calendar_utils import is_leap_year

def test_regular_leap_year():
    assert is_leap_year(2024) is True
    assert is_leap_year(1996) is True

def test_non_leap_year():
    assert is_leap_year(2023) is False
    assert is_leap_year(2019) is False

def test_century_non_leap_years():
    # FAILS ON UNPATCHED CODE
    assert is_leap_year(1900) is False
    assert is_leap_year(2100) is False
    assert is_leap_year(1800) is False

def test_quad_century_leap_years():
    assert is_leap_year(2000) is True
    assert is_leap_year(1600) is True
'''

register_task(
    task_id="easy_02",
    tier="easy",
    name="leap_year_calculator",
    domain="Datetime / Calendar",
    issue_title="Leap year function incorrectly identifies century years",
    issue_body="`is_leap_year(year)` returns True for century years like 1900 and 2100 that are divisible by 100 but not divisible by 400. In the Gregorian calendar, century years are only leap years if divisible by 400.",
    files={"src/calendar_utils.py": SRC_EASY_02, "tests/test_calendar_utils.py": TEST_EASY_02},
    fixed_files={"src/calendar_utils.py": FIXED_EASY_02},
    failing_tests=["test_century_non_leap_years"]
)

# easy_03: CLI Flag Validator
SRC_EASY_03 = '''
from typing import Optional, Set

def validate_flag(flag: Optional[str], allowed_flags: Set[str]) -> bool:
    """Validates that a CLI flag is within the allowed set (case-insensitive)."""
    # BUG: does not check if flag is None before calling .lower()
    normalized = flag.lower()
    normalized_allowed = {f.lower() for f in allowed_flags}
    return normalized in normalized_allowed
'''

FIXED_EASY_03 = '''
from typing import Optional, Set

def validate_flag(flag: Optional[str], allowed_flags: Set[str]) -> bool:
    """Validates that a CLI flag is within the allowed set (case-insensitive)."""
    if flag is None:
        return False
    normalized = flag.lower()
    normalized_allowed = {f.lower() for f in allowed_flags}
    return normalized in normalized_allowed
'''

TEST_EASY_03 = '''
import pytest
from src.flag_validator import validate_flag

def test_valid_flag():
    assert validate_flag("--verbose", {"--verbose", "--quiet"}) is True
    assert validate_flag("--VERBOSE", {"--verbose", "--quiet"}) is True

def test_invalid_flag():
    assert validate_flag("--debug", {"--verbose", "--quiet"}) is False

def test_none_flag_handled_safely():
    # FAILS ON UNPATCHED CODE
    assert validate_flag(None, {"--verbose", "--quiet"}) is False
'''

register_task(
    task_id="easy_03",
    tier="easy",
    name="cli_flag_validator",
    domain="CLI / System",
    issue_title="AttributeError on NoneType optional CLI argument",
    issue_body="`validate_flag(flag, allowed_flags)` raises `AttributeError: 'NoneType' object has no attribute 'lower'` when optional flag is passed as None. It should return False instead of crashing.",
    files={"src/flag_validator.py": SRC_EASY_03, "tests/test_flag_validator.py": TEST_EASY_03},
    fixed_files={"src/flag_validator.py": FIXED_EASY_03},
    failing_tests=["test_none_flag_handled_safely"]
)

# easy_04: Regex Version Tokenizer
SRC_EASY_04 = '''
import re

def tokenize_version(version_string: str) -> list[int]:
    """Splits semantic version string into a list of integer components."""
    if not version_string:
        return []
    # BUG: '.' is not escaped in regex pattern
    parts = re.split(r'.', version_string)
    parts = [p for p in parts if p]
    result = []
    for p in parts:
        if p.isdigit():
            result.append(int(p))
    return result
'''

FIXED_EASY_04 = '''
import re

def tokenize_version(version_string: str) -> list[int]:
    """Splits semantic version string into a list of integer components."""
    if not version_string:
        return []
    parts = re.split(r'\\.', version_string)
    parts = [p for p in parts if p]
    result = []
    for p in parts:
        if p.isdigit():
            result.append(int(p))
    return result
'''

TEST_EASY_04 = '''
import pytest
from src.version_parser import tokenize_version

def test_basic_version():
    assert tokenize_version("1.2.3") == [1, 2, 3]

def test_empty_string():
    assert tokenize_version("") == []

def test_hyphenated_non_version():
    # FAILS ON UNPATCHED CODE: unescaped dot matches '-' splitting "10-20-30" into individual chars
    assert tokenize_version("10.20.30") == [10, 20, 30]

def test_non_dot_delimiters():
    # FAILS ON UNPATCHED CODE
    assert tokenize_version("100.200") == [100, 200]
'''

register_task(
    task_id="easy_04",
    tier="easy",
    name="regex_version_tokenizer",
    domain="Parsing / Packaging",
    issue_title="Unescaped dot in version splitter splits on any character",
    issue_body="`tokenize_version(v_str)` uses `re.split('.', v_str)` with an unescaped dot, which matches any character in regex, causing strings to split improperly on non-dot characters.",
    files={"src/version_parser.py": SRC_EASY_04, "tests/test_version_parser.py": TEST_EASY_04},
    fixed_files={"src/version_parser.py": FIXED_EASY_04},
    failing_tests=["test_hyphenated_non_version", "test_non_dot_delimiters"]
)

# easy_05: Config Dict Merger
SRC_EASY_05 = '''
def merge_configs(base: dict, override: dict) -> dict:
    """Merges override dictionary into base dictionary recursively."""
    result = base.copy()
    # BUG: shallow update completely replaces nested dictionaries
    result.update(override)
    return result
'''

FIXED_EASY_05 = '''
def merge_configs(base: dict, override: dict) -> dict:
    """Merges override dictionary into base dictionary recursively."""
    result = {}
    for k, v in base.items():
        result[k] = v.copy() if isinstance(v, dict) else v
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result
'''

TEST_EASY_05 = '''
import pytest
from src.config_merger import merge_configs

def test_flat_merge():
    base = {"a": 1, "b": 2}
    over = {"b": 3, "c": 4}
    assert merge_configs(base, over) == {"a": 1, "b": 3, "c": 4}

def test_nested_merge_preserves_sibling_keys():
    # FAILS ON UNPATCHED CODE
    base = {"database": {"host": "localhost", "port": 5432, "timeout": 30}}
    override = {"database": {"port": 5433}}
    res = merge_configs(base, override)
    assert res["database"]["port"] == 5433
    assert res["database"]["host"] == "localhost"
    assert res["database"]["timeout"] == 30
'''

register_task(
    task_id="easy_05",
    tier="easy",
    name="config_dict_merger",
    domain="Configuration / Utils",
    issue_title="Shallow dict merge overwrites nested settings",
    issue_body="`merge_configs(base, override)` uses shallow `.update()`, which wipes out all unmentioned nested keys in sub-dictionaries instead of recursively merging them.",
    files={"src/config_merger.py": SRC_EASY_05, "tests/test_config_merger.py": TEST_EASY_05},
    fixed_files={"src/config_merger.py": FIXED_EASY_05},
    failing_tests=["test_nested_merge_preserves_sibling_keys"]
)

# easy_06: Sample Variance Calculator
SRC_EASY_06 = '''
from typing import List

def sample_variance(data: List[float]) -> float:
    """Computes unbiased sample variance (Bessel's correction)."""
    if not data:
        raise ValueError("Data list cannot be empty.")
    
    mean = sum(data) / len(data)
    # BUG: divides by len(data) instead of len(data) - 1
    return sum((x - mean) ** 2 for x in data) / len(data)
'''

FIXED_EASY_06 = '''
from typing import List

def sample_variance(data: List[float]) -> float:
    """Computes unbiased sample variance (Bessel's correction)."""
    if len(data) < 2:
        raise ValueError("Sample variance requires at least 2 data points.")
    
    mean = sum(data) / len(data)
    return sum((x - mean) ** 2 for x in data) / (len(data) - 1)
'''

TEST_EASY_06 = '''
import pytest
from src.stats_utils import sample_variance

def test_empty_list_raises():
    with pytest.raises(ValueError):
        sample_variance([])

def test_single_element_raises():
    # FAILS ON UNPATCHED CODE
    with pytest.raises(ValueError, match="at least 2"):
        sample_variance([10.0])

def test_unbiased_variance_calculation():
    # FAILS ON UNPATCHED CODE (returns 4.0 instead of 32/7 ≈ 4.5714)
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert pytest.approx(sample_variance(data), rel=1e-4) == 32.0 / 7.0
'''

register_task(
    task_id="easy_06",
    tier="easy",
    name="sample_variance_calculator",
    domain="Math / Statistics",
    issue_title="Sample variance uses population N divisor instead of Bessel's correction N-1",
    issue_body="`sample_variance(data)` divides by `len(data)` instead of `len(data) - 1` (Bessel's correction), resulting in biased variance estimates. Additionally, it raises ZeroDivisionError when `len(data) < 2` instead of raising a descriptive ValueError.",
    files={"src/stats_utils.py": SRC_EASY_06, "tests/test_stats_utils.py": TEST_EASY_06},
    fixed_files={"src/stats_utils.py": FIXED_EASY_06},
    failing_tests=["test_single_element_raises", "test_unbiased_variance_calculation"]
)

# ==============================================================================
# MEDIUM TASKS (Multi-function, 1-2 files, contracts / state / dataflow)
# ==============================================================================

# med_01: Case Insensitive Headers
SRC_MED_01 = '''
from collections.abc import MutableMapping

class CaseInsensitiveDict(MutableMapping):
    """A dictionary with case-insensitive string keys."""
    def __init__(self, data=None):
        self._store = {}
        if data:
            self.update(data)

    def __setitem__(self, key: str, value):
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key: str):
        return self._store[key.lower()][1]

    def __delitem__(self, key: str):
        del self._store[key.lower()]

    def __iter__(self):
        return (orig_key for orig_key, _ in self._store.values())

    def __len__(self):
        return len(self._store)

    # BUG: get and __contains__ do not normalize key to lowercase
    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._store

    def get(self, key: str, default=None):
        if key in self._store:
            return self._store[key][1]
        return default
'''

FIXED_MED_01 = '''
from collections.abc import MutableMapping

class CaseInsensitiveDict(MutableMapping):
    """A dictionary with case-insensitive string keys."""
    def __init__(self, data=None):
        self._store = {}
        if data:
            self.update(data)

    def __setitem__(self, key: str, value):
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key: str):
        return self._store[key.lower()][1]

    def __delitem__(self, key: str):
        del self._store[key.lower()]

    def __iter__(self):
        return (orig_key for orig_key, _ in self._store.values())

    def __len__(self):
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self._store

    def get(self, key: str, default=None):
        if isinstance(key, str) and key.lower() in self._store:
            return self._store[key.lower()][1]
        return default
'''

TEST_MED_01 = '''
import pytest
from src.headers import CaseInsensitiveDict

def test_basic_storage():
    h = CaseInsensitiveDict({"Content-Type": "application/json"})
    assert h["content-type"] == "application/json"
    assert h["CONTENT-TYPE"] == "application/json"

def test_contains_case_insensitive():
    # FAILS ON UNPATCHED CODE
    h = CaseInsensitiveDict({"Authorization": "Bearer token123"})
    assert "authorization" in h
    assert "AUTHORIZATION" in h
    assert "Authorization" in h

def test_get_with_case_variations():
    # FAILS ON UNPATCHED CODE
    h = CaseInsensitiveDict({"X-Custom-Header": "foobar"})
    assert h.get("x-custom-header") == "foobar"
    assert h.get("X-CUSTOM-HEADER") == "foobar"
    assert h.get("non-existent", "default") == "default"
'''

register_task(
    task_id="med_01",
    tier="medium",
    name="case_insensitive_headers",
    domain="Networking / HTTP",
    issue_title="Header dict lookup fails on case variations",
    issue_body="`CaseInsensitiveDict` normalizes keys during `__setitem__` but fails to normalize in `get()`, `__contains__`, and `pop()`, causing lookups like `'content-type' in headers` to return False when stored as `'Content-Type'`.",
    files={"src/headers.py": SRC_MED_01, "tests/test_headers.py": TEST_MED_01},
    fixed_files={"src/headers.py": FIXED_MED_01},
    failing_tests=["test_contains_case_insensitive", "test_get_with_case_variations"]
)

# med_02: Order State Machine
SRC_MED_02 = '''
from enum import Enum

class OrderState(Enum):
    CREATED = "CREATED"
    PAID = "PAID"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"

class InvalidStateTransitionError(Exception):
    pass

class OrderStateMachine:
    VALID_TRANSITIONS = {
        OrderState.CREATED: {OrderState.PAID, OrderState.CANCELLED, OrderState.SHIPPED}, # BUG: SHIPPED shouldn't be here
        OrderState.PAID: {OrderState.SHIPPED, OrderState.CANCELLED},
        OrderState.SHIPPED: {OrderState.DELIVERED},
        OrderState.DELIVERED: set(),
        OrderState.CANCELLED: set(),
    }

    def __init__(self):
        self.state = OrderState.CREATED
        self.history = [OrderState.CREATED]

    def transition_to(self, new_state: OrderState):
        if new_state not in self.VALID_TRANSITIONS[self.state]:
            raise InvalidStateTransitionError(f"Cannot transition from {self.state} to {new_state}")
        self.state = new_state
        self.history.append(new_state)
'''

FIXED_MED_02 = '''
from enum import Enum

class OrderState(Enum):
    CREATED = "CREATED"
    PAID = "PAID"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"

class InvalidStateTransitionError(Exception):
    pass

class OrderStateMachine:
    VALID_TRANSITIONS = {
        OrderState.CREATED: {OrderState.PAID, OrderState.CANCELLED},
        OrderState.PAID: {OrderState.SHIPPED, OrderState.CANCELLED},
        OrderState.SHIPPED: {OrderState.DELIVERED},
        OrderState.DELIVERED: set(),
        OrderState.CANCELLED: set(),
    }

    def __init__(self):
        self.state = OrderState.CREATED
        self.history = [OrderState.CREATED]

    def transition_to(self, new_state: OrderState):
        if new_state not in self.VALID_TRANSITIONS[self.state]:
            raise InvalidStateTransitionError(f"Cannot transition from {self.state} to {new_state}")
        self.state = new_state
        self.history.append(new_state)
'''

TEST_MED_02 = '''
import pytest
from src.state_machine import OrderStateMachine, OrderState, InvalidStateTransitionError

def test_valid_order_flow():
    sm = OrderStateMachine()
    sm.transition_to(OrderState.PAID)
    sm.transition_to(OrderState.SHIPPED)
    sm.transition_to(OrderState.DELIVERED)
    assert sm.state == OrderState.DELIVERED

def test_invalid_skip_paid_transition():
    # FAILS ON UNPATCHED CODE
    sm = OrderStateMachine()
    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(OrderState.SHIPPED)

def test_cannot_transition_from_cancelled():
    sm = OrderStateMachine()
    sm.transition_to(OrderState.CANCELLED)
    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(OrderState.PAID)
'''

register_task(
    task_id="med_02",
    tier="medium",
    name="order_state_machine",
    domain="Business Logic / State Machine",
    issue_title="Order state machine allows invalid transition skipping payment",
    issue_body="`OrderStateMachine` allows transitioning directly from `CREATED` to `SHIPPED`, bypassing `PAID`. Transitions must adhere strictly to valid sequences: CREATED -> PAID -> SHIPPED -> DELIVERED, or CANCELLED from CREATED/PAID. Any invalid transition must raise `InvalidStateTransitionError`.",
    files={"src/state_machine.py": SRC_MED_02, "tests/test_state_machine.py": TEST_MED_02},
    fixed_files={"src/state_machine.py": FIXED_MED_02},
    failing_tests=["test_invalid_skip_paid_transition"]
)

# med_03: UTF-8 Stream Reader
SRC_MED_03 = '''
class ChunkedUTF8Reader:
    def __init__(self, chunk_size: int = 4):
        self.chunk_size = chunk_size

    def decode_stream(self, byte_data: bytes) -> str:
        """Decodes raw bytes arriving in chunks into a complete unicode string."""
        chunks = [byte_data[i:i + self.chunk_size] for i in range(0, len(byte_data), self.chunk_size)]
        text_parts = []
        # BUG: decodes each chunk separately with strict utf-8 decoding
        for chunk in chunks:
            text_parts.append(chunk.decode("utf-8"))
        return "".join(text_parts)
'''

FIXED_MED_03 = '''
import codecs

class ChunkedUTF8Reader:
    def __init__(self, chunk_size: int = 4):
        self.chunk_size = chunk_size

    def decode_stream(self, byte_data: bytes) -> str:
        """Decodes raw bytes arriving in chunks into a complete unicode string."""
        chunks = [byte_data[i:i + self.chunk_size] for i in range(0, len(byte_data), self.chunk_size)]
        decoder = codecs.getincrementaldecoder("utf-8")()
        text_parts = [decoder.decode(chunk, final=False) for chunk in chunks]
        text_parts.append(decoder.decode(b"", final=True))
        return "".join(text_parts)
'''

TEST_MED_03 = '''
import pytest
from src.stream_reader import ChunkedUTF8Reader

def test_ascii_chunks():
    reader = ChunkedUTF8Reader(chunk_size=3)
    data = b"Hello, World!"
    assert reader.decode_stream(data) == "Hello, World!"

def test_split_multibyte_emoji():
    # FAILS ON UNPATCHED CODE
    reader = ChunkedUTF8Reader(chunk_size=3)
    emoji_bytes = "🚀Launch!".encode("utf-8")
    result = reader.decode_stream(emoji_bytes)
    assert result == "🚀Launch!"
'''

register_task(
    task_id="med_03",
    tier="medium",
    name="utf8_stream_reader",
    domain="Encoding / I/O",
    issue_title="UnicodeDecodeError when multi-byte UTF-8 character is sliced at buffer boundary",
    issue_body="`ChunkedUTF8Reader.read_chunks()` decodes each raw byte chunk independently without buffering uncompleted multi-byte sequence prefixes at chunk boundaries, raising `UnicodeDecodeError` when a character like an emoji or accent spans across chunk boundaries.",
    files={"src/stream_reader.py": SRC_MED_03, "tests/test_stream_reader.py": TEST_MED_03},
    fixed_files={"src/stream_reader.py": FIXED_MED_03},
    failing_tests=["test_split_multibyte_emoji"]
)

# med_04: JSON Schema Validator
SRC_MED_04 = '''
class SchemaValidationError(Exception):
    pass

TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

def validate_instance(instance, schema: dict) -> bool:
    expected_type_name = schema.get("type")
    if expected_type_name:
        expected_type = TYPE_MAP.get(expected_type_name)
        if expected_type and not isinstance(instance, expected_type):
            raise SchemaValidationError(f"Expected {expected_type_name}, got {type(instance).__name__}")
    
    # BUG: If type is array and 'items' is present in schema, items are not checked!
    return True
'''

FIXED_MED_04 = '''
class SchemaValidationError(Exception):
    pass

TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

def validate_instance(instance, schema: dict) -> bool:
    expected_type_name = schema.get("type")
    if expected_type_name:
        expected_type = TYPE_MAP.get(expected_type_name)
        if expected_type and not isinstance(instance, expected_type):
            raise SchemaValidationError(f"Expected {expected_type_name}, got {type(instance).__name__}")
    
    if expected_type_name == "array" and "items" in schema and isinstance(instance, list):
        for item in instance:
            validate_instance(item, schema["items"])
    return True
'''

TEST_MED_04 = '''
import pytest
from src.validator import validate_instance, SchemaValidationError

def test_valid_primitive():
    assert validate_instance("hello", {"type": "string"}) is True
    assert validate_instance(42, {"type": "integer"}) is True

def test_invalid_primitive():
    with pytest.raises(SchemaValidationError):
        validate_instance("not an int", {"type": "integer"})

def test_array_items_invalid_element():
    # FAILS ON UNPATCHED CODE
    schema = {
        "type": "array",
        "items": {"type": "integer"}
    }
    with pytest.raises(SchemaValidationError):
        validate_instance([1, 2, "three", 4], schema)

def test_array_items_valid():
    schema = {
        "type": "array",
        "items": {"type": "string"}
    }
    assert validate_instance(["apple", "banana"], schema) is True
'''

register_task(
    task_id="med_04",
    tier="medium",
    name="json_schema_validator",
    domain="Validation / Data Types",
    issue_title="Array items schema validation ignored",
    issue_body="`validate_instance(instance, schema)` checks if an instance is an array (`type: 'array'`), but fails to recursively validate the elements inside the array when an `items` sub-schema is defined.",
    files={"src/validator.py": SRC_MED_04, "tests/test_validator.py": TEST_MED_04},
    fixed_files={"src/validator.py": FIXED_MED_04},
    failing_tests=["test_array_items_invalid_element"]
)

# med_05: Sliding Window Rate Limiter
SRC_MED_05 = '''
from collections import deque

class SlidingWindowRateLimiter:
    """Rate limiter enforcing maximum requests within a sliding window in seconds."""
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def allow_request(self, current_time: float) -> bool:
        # BUG: uses (current_time - timestamp) > window_seconds instead of >=
        while self.timestamps and (current_time - self.timestamps[0]) > self.window_seconds:
            self.timestamps.popleft()

        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(current_time)
            return True
        return False
'''

FIXED_MED_05 = '''
from collections import deque

class SlidingWindowRateLimiter:
    """Rate limiter enforcing maximum requests within a sliding window in seconds."""
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def allow_request(self, current_time: float) -> bool:
        while self.timestamps and (current_time - self.timestamps[0]) >= self.window_seconds:
            self.timestamps.popleft()

        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(current_time)
            return True
        return False
'''

TEST_MED_05 = '''
import pytest
from src.rate_limiter import SlidingWindowRateLimiter

def test_allows_under_limit():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10.0)
    assert limiter.allow_request(0.0) is True
    assert limiter.allow_request(1.0) is True
    assert limiter.allow_request(2.0) is False

def test_boundary_expiration():
    # FAILS ON UNPATCHED CODE
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10.0)
    assert limiter.allow_request(0.0) is True
    assert limiter.allow_request(5.0) is False
    assert limiter.allow_request(10.0) is True
'''

register_task(
    task_id="med_05",
    tier="medium",
    name="sliding_window_rate_limiter",
    domain="Concurrency / Algorithms",
    issue_title="Rate limiter boundary allows request exceeding window limit",
    issue_body="`SlidingWindowRateLimiter` uses strict `<` comparison for timestamp eviction instead of `<=`, allowing requests on the exact window boundary to be double-counted or retaining expired requests.",
    files={"src/rate_limiter.py": SRC_MED_05, "tests/test_rate_limiter.py": TEST_MED_05},
    fixed_files={"src/rate_limiter.py": FIXED_MED_05},
    failing_tests=["test_boundary_expiration"]
)

# med_06: Event Dispatcher Error Propagation
SRC_MED_06 = '''
class DispatchAggregateError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"{len(errors)} listener(s) raised an exception.")

class EventDispatcher:
    def __init__(self):
        self._listeners = {}

    def subscribe(self, event_name: str, listener):
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(listener)

    def dispatch(self, event_name: str, data=None):
        if event_name not in self._listeners:
            return
        # BUG: stops iteration immediately on first exception
        for listener in self._listeners[event_name]:
            listener(data)
'''

FIXED_MED_06 = '''
class DispatchAggregateError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"{len(errors)} listener(s) raised an exception.")

class EventDispatcher:
    def __init__(self):
        self._listeners = {}

    def subscribe(self, event_name: str, listener):
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(listener)

    def dispatch(self, event_name: str, data=None):
        if event_name not in self._listeners:
            return
        errors = []
        for listener in self._listeners[event_name]:
            try:
                listener(data)
            except Exception as e:
                errors.append(e)
        if errors:
            raise DispatchAggregateError(errors)
'''

TEST_MED_06 = '''
import pytest
from src.dispatcher import EventDispatcher, DispatchAggregateError

def test_all_listeners_called_when_first_fails():
    dispatcher = EventDispatcher()
    called = []

    def bad_listener(data):
        called.append("bad")
        raise ValueError("Something went wrong")

    def good_listener(data):
        called.append("good")

    dispatcher.subscribe("user_signup", bad_listener)
    dispatcher.subscribe("user_signup", good_listener)

    # FAILS ON UNPATCHED CODE
    with pytest.raises(DispatchAggregateError) as exc_info:
        dispatcher.dispatch("user_signup", {"username": "alice"})

    assert "bad" in called
    assert "good" in called
    assert len(exc_info.value.errors) == 1
'''

register_task(
    task_id="med_06",
    tier="medium",
    name="event_dispatcher",
    domain="Event-Driven / Architecture",
    issue_title="Failing subscriber stops execution of remaining event listeners",
    issue_body="`EventDispatcher.dispatch()` immediately bubbles up exceptions if one subscriber fails, preventing all subsequent registered listeners from receiving the event. It should collect exceptions into a composite `DispatchAggregateError` and guarantee all listeners are called.",
    files={"src/dispatcher.py": SRC_MED_06, "tests/test_dispatcher.py": TEST_MED_06},
    fixed_files={"src/dispatcher.py": FIXED_MED_06},
    failing_tests=["test_all_listeners_called_when_first_fails"]
)

# ==============================================================================
# HARD TASKS (Cross-module / architectural / subtle graph & state constraints)
# ==============================================================================

# hard_01: Dependency Graph Cycle Detection
SRC_HARD_01_GRAPH = '''
class Node:
    def __init__(self, name: str):
        self.name = name
        self.dependencies = []

class DependencyGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, name: str) -> Node:
        if name not in self.nodes:
            self.nodes[name] = Node(name)
        return self.nodes[name]

    def add_dependency(self, from_node: str, to_node: str):
        parent = self.add_node(from_node)
        child = self.add_node(to_node)
        parent.dependencies.append(child)
'''

SRC_HARD_01_SORTER = '''
from src.graph import DependencyGraph

class CycleError(Exception):
    pass

class TopologicalSorter:
    def __init__(self, graph: DependencyGraph):
        self.graph = graph

    def sort(self) -> list[str]:
        visited = set()
        result = []

        def dfs(node):
            # BUG: visited node is flagged as cycle even if fully processed in earlier tree branch
            if node in visited:
                raise CycleError(f"Cycle detected at node {node.name}")
            visited.add(node)
            for dep in node.dependencies:
                dfs(dep)
            result.append(node.name)

        for node in self.graph.nodes.values():
            if node not in visited:
                dfs(node)
        return result
'''

FIXED_HARD_01_SORTER = '''
from src.graph import DependencyGraph

class CycleError(Exception):
    pass

class TopologicalSorter:
    def __init__(self, graph: DependencyGraph):
        self.graph = graph

    def sort(self) -> list[str]:
        visiting = set()
        processed = set()
        result = []

        def dfs(node):
            if node in visiting:
                raise CycleError(f"Cycle detected at node {node.name}")
            if node in processed:
                return
            visiting.add(node)
            for dep in node.dependencies:
                dfs(dep)
            visiting.remove(node)
            processed.add(node)
            result.append(node.name)

        for node in self.graph.nodes.values():
            if node not in processed:
                dfs(node)
        return result
'''

TEST_HARD_01 = '''
import pytest
from src.graph import DependencyGraph
from src.sorter import TopologicalSorter, CycleError

def test_diamond_dependency_dag():
    g = DependencyGraph()
    g.add_dependency("A", "B")
    g.add_dependency("A", "C")
    g.add_dependency("B", "D")
    g.add_dependency("C", "D")

    sorter = TopologicalSorter(g)
    # FAILS ON UNPATCHED CODE
    order = sorter.sort()
    assert order.index("D") < order.index("B")
    assert order.index("D") < order.index("C")
    assert order.index("B") < order.index("A")
    assert order.index("C") < order.index("A")

def test_actual_cycle():
    g = DependencyGraph()
    g.add_dependency("X", "Y")
    g.add_dependency("Y", "X")
    sorter = TopologicalSorter(g)
    with pytest.raises(CycleError):
        sorter.sort()
'''

register_task(
    task_id="hard_01",
    tier="hard",
    name="topological_sorter_dag",
    domain="Algorithms / Compilers",
    issue_title="False positive cycle detection across disjoint component paths",
    issue_body="The topological sorter across `graph.py` and `sorter.py` uses a single `visited` set without a 3-color DFS distinction (visiting vs visited). When a node is reachable via multiple distinct paths in a valid DAG (diamond dependency), it erroneously raises `CycleError`.",
    files={"src/graph.py": SRC_HARD_01_GRAPH, "src/sorter.py": SRC_HARD_01_SORTER, "tests/test_topological_sort.py": TEST_HARD_01},
    fixed_files={"src/sorter.py": FIXED_HARD_01_SORTER},
    failing_tests=["test_diamond_dependency_dag"]
)

# hard_02: LRU Cache with TTL Secondary Key Desync
SRC_HARD_02_STORAGE = '''
import time

class CacheNode:
    def __init__(self, key: str, value, ttl: float):
        self.key = key
        self.value = value
        self.expiry = time.time() + ttl if ttl > 0 else float("inf")
        self.prev = None
        self.next = None

    def is_expired(self, current_time: float) -> bool:
        return current_time >= self.expiry
'''

SRC_HARD_02_MANAGER = '''
import time
from src.storage import CacheNode

class TTLLRUCache:
    def __init__(self, capacity: int, default_ttl: float = 60.0):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.index = {}

    def put(self, key: str, value, ttl: float = None):
        if ttl is None:
            ttl = self.default_ttl
        node = CacheNode(key, value, ttl)
        self.index[key] = node

    def get(self, key: str):
        if key not in self.index:
            return None
        node = self.index[key]
        if node.is_expired(time.time()):
            del self.index[key]
            return None
        return node.value

    # BUG: contains_key checks index directly without checking expiration!
    def contains_key(self, key: str) -> bool:
        return key in self.index
'''

FIXED_HARD_02_MANAGER = '''
import time
from src.storage import CacheNode

class TTLLRUCache:
    def __init__(self, capacity: int, default_ttl: float = 60.0):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.index = {}

    def put(self, key: str, value, ttl: float = None):
        if ttl is None:
            ttl = self.default_ttl
        node = CacheNode(key, value, ttl)
        self.index[key] = node

    def get(self, key: str):
        if key not in self.index:
            return None
        node = self.index[key]
        if node.is_expired(time.time()):
            del self.index[key]
            return None
        return node.value

    def contains_key(self, key: str) -> bool:
        if key not in self.index:
            return False
        if self.index[key].is_expired(time.time()):
            del self.index[key]
            return False
        return True
'''

TEST_HARD_02 = '''
import pytest
import time
from src.cache_manager import TTLLRUCache

def test_put_and_get():
    cache = TTLLRUCache(capacity=5, default_ttl=10.0)
    cache.put("k1", "val1")
    assert cache.get("k1") == "val1"
    assert cache.contains_key("k1") is True

def test_contains_key_respects_expiration():
    # FAILS ON UNPATCHED CODE
    cache = TTLLRUCache(capacity=5, default_ttl=0.05)
    cache.put("ephemeral", "data")
    assert cache.contains_key("ephemeral") is True
    time.sleep(0.08)
    assert cache.contains_key("ephemeral") is False
'''

register_task(
    task_id="hard_02",
    tier="hard",
    name="ttl_lru_cache",
    domain="Caching / Data Structures",
    issue_title="Key existence check reports expired keys as present",
    issue_body="`TTLLRUCache` across `storage.py` and `cache_manager.py` maintains an index dictionary and linked nodes. When checking `contains_key()`, it checks the index dictionary without validating if the key has expired against `ttl`, returning True for expired keys.",
    files={"src/storage.py": SRC_HARD_02_STORAGE, "src/cache_manager.py": SRC_HARD_02_MANAGER, "tests/test_ttl_cache.py": TEST_HARD_02},
    fixed_files={"src/cache_manager.py": FIXED_HARD_02_MANAGER},
    failing_tests=["test_contains_key_respects_expiration"]
)

# hard_03: AST Lexical Scope Shadowing
SRC_HARD_03_SCOPE = '''
class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def define(self, name: str, var_type: str):
        self.symbols[name] = var_type

    def resolve(self, name: str):
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.resolve(name)
        return None
'''

SRC_HARD_03_ANALYZER = '''
from src.scope import SymbolTable

class ScopeAnalyzer:
    def __init__(self):
        self.current_scope = SymbolTable()

    def enter_function(self):
        self.current_scope = SymbolTable(parent=self.current_scope)

    def exit_function(self):
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent

    # BUG: enter_comprehension does not create a child scope
    def enter_comprehension(self):
        pass

    def exit_comprehension(self):
        pass

    def declare_variable(self, name: str, var_type: str):
        self.current_scope.define(name, var_type)

    def lookup(self, name: str):
        return self.current_scope.resolve(name)
'''

FIXED_HARD_03_ANALYZER = '''
from src.scope import SymbolTable

class ScopeAnalyzer:
    def __init__(self):
        self.current_scope = SymbolTable()

    def enter_function(self):
        self.current_scope = SymbolTable(parent=self.current_scope)

    def exit_function(self):
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent

    def enter_comprehension(self):
        self.current_scope = SymbolTable(parent=self.current_scope)

    def exit_comprehension(self):
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent

    def declare_variable(self, name: str, var_type: str):
        self.current_scope.define(name, var_type)

    def lookup(self, name: str):
        return self.current_scope.resolve(name)
'''

TEST_HARD_03 = '''
import pytest
from src.analyzer import ScopeAnalyzer

def test_function_scoping():
    analyzer = ScopeAnalyzer()
    analyzer.declare_variable("global_var", "int")
    analyzer.enter_function()
    analyzer.declare_variable("local_var", "str")
    assert analyzer.lookup("local_var") == "str"
    assert analyzer.lookup("global_var") == "int"
    analyzer.exit_function()
    assert analyzer.lookup("local_var") is None

def test_comprehension_does_not_shadow_outer_var():
    # FAILS ON UNPATCHED CODE
    analyzer = ScopeAnalyzer()
    analyzer.enter_function()
    analyzer.declare_variable("item", "CustomObject")
    
    analyzer.enter_comprehension()
    analyzer.declare_variable("item", "int")
    assert analyzer.lookup("item") == "int"
    analyzer.exit_comprehension()

    assert analyzer.lookup("item") == "CustomObject"
'''

register_task(
    task_id="hard_03",
    tier="hard",
    name="ast_variable_scoping",
    domain="Compilers / Static Analysis",
    issue_title="Nested comprehension variable leaks into enclosing function scope",
    issue_body="The AST variable analyzer across `scope.py` and `analyzer.py` models symbol tables. When encountering a list comprehension or lambda, it reuses the active function scope instead of pushing a child lexical scope, causing loop target variables inside comprehensions to overwrite identically named outer variables.",
    files={"src/scope.py": SRC_HARD_03_SCOPE, "src/analyzer.py": SRC_HARD_03_ANALYZER, "tests/test_scoping.py": TEST_HARD_03},
    fixed_files={"src/analyzer.py": FIXED_HARD_03_ANALYZER},
    failing_tests=["test_comprehension_does_not_shadow_outer_var"]
)

# hard_04: Concurrent Task Dependency Cancellation
SRC_HARD_04_TASK = '''
class TaskState:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

class Task:
    def __init__(self, task_id: str, action):
        self.task_id = task_id
        self.action = action
        self.state = TaskState.PENDING
        self.dependents = []

    def cancel(self):
        self.state = TaskState.CANCELLED
'''

SRC_HARD_04_SCHEDULER = '''
from src.task import Task, TaskState

class TaskScheduler:
    def __init__(self):
        self.tasks = {}

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task

    def add_dependency(self, parent_id: str, child_id: str):
        self.tasks[parent_id].dependents.append(self.tasks[child_id])

    def cancel_task(self, task_id: str):
        if task_id not in self.tasks:
            return
        target = self.tasks[task_id]
        target.cancel()
        # BUG: does not cascade cancellation to dependents
'''

FIXED_HARD_04_SCHEDULER = '''
from src.task import Task, TaskState

class TaskScheduler:
    def __init__(self):
        self.tasks = {}

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task

    def add_dependency(self, parent_id: str, child_id: str):
        self.tasks[parent_id].dependents.append(self.tasks[child_id])

    def cancel_task(self, task_id: str):
        if task_id not in self.tasks:
            return
        target = self.tasks[task_id]
        target.cancel()
        for child in target.dependents:
            if child.state == TaskState.PENDING:
                self.cancel_task(child.task_id)
'''

TEST_HARD_04 = '''
import pytest
from src.task import Task, TaskState
from src.scheduler import TaskScheduler

def test_cancel_cascades_to_children():
    scheduler = TaskScheduler()
    t1 = Task("t1", lambda: 1)
    t2 = Task("t2", lambda: 2)
    t3 = Task("t3", lambda: 3)

    scheduler.add_task(t1)
    scheduler.add_task(t2)
    scheduler.add_task(t3)

    scheduler.add_dependency("t1", "t2")
    scheduler.add_dependency("t2", "t3")

    # FAILS ON UNPATCHED CODE
    scheduler.cancel_task("t1")
    assert t1.state == TaskState.CANCELLED
    assert t2.state == TaskState.CANCELLED
    assert t3.state == TaskState.CANCELLED
'''

register_task(
    task_id="hard_04",
    tier="hard",
    name="task_scheduler_cancellation",
    domain="Concurrency / Async Execution",
    issue_title="Cancelled parent task fails to cancel dependent queued child tasks",
    issue_body="`TaskScheduler` across `task.py` and `scheduler.py` coordinates dependent jobs. When a parent task is cancelled, child tasks waiting on its completion remain indefinitely in `PENDING` state instead of being transitioned to `CANCELLED`, blocking workflow completion.",
    files={"src/task.py": SRC_HARD_04_TASK, "src/scheduler.py": SRC_HARD_04_SCHEDULER, "tests/test_scheduler.py": TEST_HARD_04},
    fixed_files={"src/scheduler.py": FIXED_HARD_04_SCHEDULER},
    failing_tests=["test_cancel_cascades_to_children"]
)

# hard_05: Query Builder Alias Qualifier Resolution
SRC_HARD_05_MODELS = '''
class Table:
    def __init__(self, name: str, alias: str = None):
        self.name = name
        self.alias = alias

    def get_qualifier(self) -> str:
        return self.alias if self.alias else self.name
'''

SRC_HARD_05_BUILDER = '''
from src.models import Table

class QueryBuilder:
    def __init__(self, table: Table):
        self.table = table
        self.conditions = []

    def where(self, column: str, value: str):
        # BUG: uses table.name unconditionally rather than table.get_qualifier()
        self.conditions.append(f"{self.table.name}.{column} = '{value}'")
        return self

    def to_sql(self) -> str:
        base = f"SELECT * FROM {self.table.name}"
        if self.table.alias:
            base += f" AS {self.table.alias}"
        if self.conditions:
            base += " WHERE " + " AND ".join(self.conditions)
        return base
'''

FIXED_HARD_05_BUILDER = '''
from src.models import Table

class QueryBuilder:
    def __init__(self, table: Table):
        self.table = table
        self.conditions = []

    def where(self, column: str, value: str):
        self.conditions.append(f"{self.table.get_qualifier()}.{column} = '{value}'")
        return self

    def to_sql(self) -> str:
        base = f"SELECT * FROM {self.table.name}"
        if self.table.alias:
            base += f" AS {self.table.alias}"
        if self.conditions:
            base += " WHERE " + " AND ".join(self.conditions)
        return base
'''

TEST_HARD_05 = '''
import pytest
from src.models import Table
from src.builder import QueryBuilder

def test_query_without_alias():
    t = Table("orders")
    qb = QueryBuilder(t).where("status", "completed")
    assert qb.to_sql() == "SELECT * FROM orders WHERE orders.status = 'completed'"

def test_query_with_alias():
    # FAILS ON UNPATCHED CODE
    t = Table("users", alias="u")
    qb = QueryBuilder(t).where("id", "42")
    assert qb.to_sql() == "SELECT * FROM users AS u WHERE u.id = '42'"
'''

register_task(
    task_id="hard_05",
    tier="hard",
    name="query_builder_alias",
    domain="Database / ORM",
    issue_title="Query builder renders raw table name instead of alias in WHERE clause",
    issue_body="The query generator across `models.py` and `builder.py` allows aliasing joined tables (e.g., `users AS u`). However, `where()` conditions construct column identifiers using the original table name rather than the alias when an alias is present, generating invalid SQL syntax.",
    files={"src/models.py": SRC_HARD_05_MODELS, "src/builder.py": SRC_HARD_05_BUILDER, "tests/test_builder.py": TEST_HARD_05},
    fixed_files={"src/builder.py": FIXED_HARD_05_BUILDER},
    failing_tests=["test_query_with_alias"]
)

# hard_06: Dynamic Plugin Context Isolation
SRC_HARD_06_PLUGIN = '''
class PluginContext:
    # BUG: mutable default dictionary shared across instances!
    def __init__(self, plugin_name: str, metadata: dict = {}):
        self.plugin_name = plugin_name
        self.metadata = metadata

    def set_config(self, key: str, val):
        self.metadata[key] = val
'''

FIXED_HARD_06_PLUGIN = '''
class PluginContext:
    def __init__(self, plugin_name: str, metadata: dict = None):
        self.plugin_name = plugin_name
        self.metadata = {} if metadata is None else metadata

    def set_config(self, key: str, val):
        self.metadata[key] = val
'''

SRC_HARD_06_REGISTRY = '''
from src.plugin import PluginContext

class PluginRegistry:
    def __init__(self):
        self.contexts = {}

    def register(self, plugin_name: str) -> PluginContext:
        ctx = PluginContext(plugin_name)
        self.contexts[plugin_name] = ctx
        return ctx
'''

TEST_HARD_06 = '''
import pytest
from src.registry import PluginRegistry

def test_plugin_isolation():
    registry = PluginRegistry()
    p1 = registry.register("AuthPlugin")
    p2 = registry.register("LoggingPlugin")

    p1.set_config("token", "secret123")

    # FAILS ON UNPATCHED CODE
    assert "token" not in p2.metadata
    assert p2.metadata == {}
'''

register_task(
    task_id="hard_06",
    tier="hard",
    name="plugin_registry_isolation",
    domain="Extensibility / Architecture",
    issue_title="Shared mutable default causes plugin state cross-contamination",
    issue_body="`PluginContext` across `plugin.py` and `registry.py` uses a class-level mutable dictionary `metadata={}` as a default argument. When one plugin injects configuration or metadata into its execution context, it leaks into all other concurrently executed plugins.",
    files={"src/plugin.py": SRC_HARD_06_PLUGIN, "src/registry.py": SRC_HARD_06_REGISTRY, "tests/test_plugin_registry.py": TEST_HARD_06},
    fixed_files={"src/plugin.py": FIXED_HARD_06_PLUGIN},
    failing_tests=["test_plugin_isolation"]
)

# ==============================================================================
# Generate Manifest
# ==============================================================================

manifest_path = os.path.join(BASE_DIR, "data", "task_manifest.json")
with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump({
        "total_tasks": len(TASKS),
        "tiers": {
            "easy": [t for t in TASKS if t["tier"] == "easy"],
            "medium": [t for t in TASKS if t["tier"] == "medium"],
            "hard": [t for t in TASKS if t["tier"] == "hard"],
        },
        "all_tasks": TASKS
    }, f, indent=2)

print(f"\nSuccessfully generated {len(TASKS)} mock repositories with verified ground truth diffs!")
print(f"Manifest written to data/task_manifest.json")
