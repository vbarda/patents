"""This client allows querying REST API provided by patentsview.org.
It also implements a simple domain-specific language that allows composing complext queries
using primitive building blocks.

Author: Vadym Barda vadim.barda@gmail.com
"""
import abc
import copy
import datetime
import enum
import logging
import pprint
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Set

import arrow
import funcy
import pandas as pd
import requests


logger = logging.getLogger(__name__)


# Constant names
PATENT_TYPE_FIELD = "patent_type"
PATENT_DATE_FIELD = "patent_date"
INVENTOR_LAST_NAME_FIELD = "inventor_last_name"

# Defaults
DEFAULT_URL = "https://www.patentsview.org/api/patents/query"
DEFAULT_START_DATE = datetime.date(2000, 1, 1)
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 1000
DEFAULT_MAX_RESULTS = 1000
DEFAULT_TIMEOUT_SECONDS = 10


# Helpers

def _is_flat_dictionary(d: Dict[str, Any]) -> bool:
    """Check if a dictionary is flat."""
    return all(not isinstance(value, (dict, list)) for value in d.values())


def _get_next_states(d: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Generate next states to visit when flattening a dictionary."""
    for key, value in d.items():
        if isinstance(value, list):
            for child_value in value:
                new_state = {
                    **d,
                    **{key: child_value}
                }
                yield new_state
        elif isinstance(value, dict):
            # Create a copy
            new_state: Dict[str, Any] = copy.deepcopy(d)
            new_state.pop(key)  # remove parent key
            for child_key, child_value in value.items():
                joined_key = "{}__{}".format(key, child_key)
                new_state[joined_key] = child_value
            yield new_state
        else:
            continue


def _flatten(d: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Produce multiple flattened dictionaries from a single nested dictionary."""
    if _is_flat_dictionary(d):
        yield d
        stack = []
    else:
        stack = [d]

    seen_state_hashes: Set[str] = set()
    while stack:
        state = stack.pop()
        next_states = _get_next_states(state)
        for state in next_states:
            state_hash = pprint.pformat(state)
            if state_hash in seen_state_hashes:
                continue

            seen_state_hashes.add(state_hash)
            if _is_flat_dictionary(state):
                yield state
            else:
                stack.append(state)

# DSL


@enum.unique
class Operator(enum.Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


def _get_operator_key(operator: Operator) -> str:
    """Get a string key corresponding to an Operator instance."""
    operator_to_key_mapping = {
        Operator.EQ: "_eq",
        Operator.NEQ: "_neq",
        Operator.GT: "_gt",
        Operator.GTE: "_gte",
        Operator.LT: "_lt",
        Operator.LTE: "_lte",
    }
    if operator not in operator_to_key_mapping:
        raise AssertionError("Operator '{}' is not supported".format(operator))
    return operator_to_key_mapping[operator]


class Criterion(abc.ABC):
    """An abstract interface for a generic criterion."""

    def to_dict(self) -> dict:
        """Serialize a criterion to dict."""
        raise NotImplementedError()


class Filter(Criterion):
    """A generic representation of a filter."""

    def __init__(self, field: str, operator: Operator, value: Any) -> None:
        """Initialize."""
        self.field = field
        self.operator = operator
        self.value = value

    def to_dict(self) -> dict:
        """Serialize filter to dict."""
        return {
            _get_operator_key(self.operator): {
                self.field: self.value
            }
        }


# Logical operators


class LogicalOperator(Criterion):
    """An interface for logical operator on filters."""
    KEY = None

    def __init__(self, *criteria: Criterion) -> None:
        """Initialize."""
        if not criteria:
            raise AssertionError(
                "Cannot use LogicalOperator on an empty set. Please provide at least one criterion"
            )
        self.criteria = criteria

    def to_dict(self) -> dict:
        """Serialize logical operator with other criteria to dict."""
        if self.KEY is None:
            raise AssertionError("Cannot serialize a LogicalOperator without a corresponding key.")

        return {
            self.KEY: [c.to_dict() for c in self.criteria]
        }


class Or(LogicalOperator):
    """A logical Or operator."""
    KEY = "_or"


class And(LogicalOperator):
    """A logical And operator."""
    KEY = "_and"


class Not(LogicalOperator):
    """A logical Not operator."""
    KEY = "_not"


class Query:
    """A final query object."""

    def __init__(self, criterion: Criterion) -> None:
        """Initialize."""
        self.criterion = criterion


@enum.unique
class SortDirection(enum.Enum):
    """A generic representation of a SortDirection."""
    ASC = "asc"
    DESC = "desc"


class SortOption(NamedTuple):
    """A generic representation of a SortOption."""
    field: str
    direction: SortDirection

    def to_dict(self) -> dict:
        """Serialize a SortOption to dict."""
        return {self.field: self.direction.value}


class PatentSearchRequest(NamedTuple):
    """An internal representation of patent search request."""
    query: Query
    field_names: List[str]
    sort_options: List[SortOption]
    page: Optional[int] = None
    page_size: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert PatentSearchRequest to dict."""
        d = {"q": self.query.criterion.to_dict()}
        options = {}
        if self.field_names:
            d["f"] = self.field_names
        if self.sort_options:
            d["s"] = [s.to_dict() for s in self.sort_options]
        if self.page is not None:
            options["page"] = self.page
        if self.page_size is not None:
            options["per_page"] = self.page_size
        if options:
            d["o"] = options
        return d


class PatentSearchResponse(NamedTuple):
    """An internal representation of patent search response."""
    hits: List[Dict[str, Any]]
    count: int
    total_patent_count: int

    def to_dataframe(self, flatten: bool = True) -> pd.DataFrame:
        """Export search results into a pandas dataframe."""
        transformer = _flatten if flatten else funcy.identity
        transformed_hits = funcy.flatten(transformer(hit) for hit in self.hits)
        return pd.DataFrame(transformed_hits)

    def __repr__(self):
        """Override display to ignore hits."""
        return (
            "{} (count: {}, total_patent_count: {})"
            .format(self.__class__.__name__, self.count, self.total_patent_count)
        )


# Client


class PatentsViewClient:

    def __init__(
        self,
        url: Optional[str] = None,
        timeout_seconds: Optional[int] = None
    ) -> None:
        """Initialize with a URL."""
        if url is None:
            url = DEFAULT_URL
        if timeout_seconds is None:
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS

        self._url = url
        self._timeout_seconds = timeout_seconds

    def _fetch_results(
        self,
        patent_search_request: PatentSearchRequest
    ) -> PatentSearchResponse:
        """Get results of a query."""
        response = requests.post(
            self._url,
            json=patent_search_request.to_dict(),
            timeout=self._timeout_seconds
        )
        response_as_json = response.json()
        # Convert json to a PatentSearchResponse object
        hits = response_as_json.get("patents", [])
        count = response_as_json.get("count", 0)
        total_patent_count = response_as_json.get("total_patent_count", 0)
        return PatentSearchResponse(
            hits=hits, count=count, total_patent_count=total_patent_count
        )

    @staticmethod
    def _update_search_request(
        patent_search_request: PatentSearchRequest,
        page: int,
        page_size: int
    ) -> PatentSearchRequest:
        """Get search request with updated pagination info."""
        return PatentSearchRequest(
            query=patent_search_request.query,
            sort_options=patent_search_request.sort_options,
            field_names=patent_search_request.field_names,
            page=page,
            page_size=page_size,
        )

    def get_query_results(
        self,
        patent_search_request: PatentSearchRequest,
        max_results: Optional[int] = None,
    ) -> PatentSearchResponse:
        """Get full query results.

        Args:
            patent_search_request: PatentSearchRequest to be executed
            max_results: int, max # of results to return. Defaults to DEFAULT_MAX_RESULTS

        Returns:
            PatentSearchResponse
        """
        page_size = patent_search_request.page_size
        page = patent_search_request.page
        # Set some defaults.
        if page is None:
            page = DEFAULT_PAGE
        if page_size is None:
            page_size = DEFAULT_PAGE_SIZE
        # Adjust page size according to max # of results, if specified
        if max_results is not None:
            page_size = min(page_size, max_results)

        # Start by fetching a first batch of results.
        request = self._update_search_request(patent_search_request, page, page_size)
        logger.info(
            "Fetching first batch of results to estimate total # of results matching criteria."
        )
        response = self._fetch_results(request)
        logger.info("Fetched %s results.", page * page_size)

        # Find how many results in total we need to fetch.
        n_results_to_fetch = response.total_patent_count
        if max_results is not None:
            n_results_to_fetch = min(n_results_to_fetch, max_results)

        logger.info(
            "Found %s total results, with maximum # of results to fetch set to %s "
            "(None = all results will be fetched)", n_results_to_fetch, max_results
        )

        # How many results do we have left to fetch.
        remaining_n_results_to_fetch = n_results_to_fetch - response.count

        # Container for storing paginated responses. Store first set of responses.
        responses = [response]

        logger.info("Beginning to paginate in batches of %s", page_size)
        # Begin to paginate now.
        while remaining_n_results_to_fetch > 0:
            # Increment the page #.
            page += 1
            request = self._update_search_request(request, page, page_size)
            response = self._fetch_results(request)
            logger.info("Fetched %s results.", page * page_size)
            # Decrement # of results to fetch to exit the while loop.
            remaining_n_results_to_fetch -= response.count
            responses.append(response)

        logger.info("Successfully fetched all of the batches.")

        # Now combine the responses.
        hits = funcy.lflatten(response.hits for response in responses)
        count = sum(response.count for response in responses)
        if count != n_results_to_fetch:
            raise AssertionError(
                "Expected the # of combined paginated responses {} to be "
                "the same as the # of results to fetch {}"
                .format(count, n_results_to_fetch)
            )
        return PatentSearchResponse(
            hits=hits, count=count, total_patent_count=response.total_patent_count
        )


def make_date_range_criterion(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Criterion:
    """Make date range criterion."""
    if start_date is None:
        start_date = DEFAULT_START_DATE.isoformat()
    if end_date is None:
        end_date = arrow.get().date().isoformat()
    date_criteria = (
        Filter(PATENT_DATE_FIELD, Operator.GTE, start_date),
        Filter(PATENT_DATE_FIELD, Operator.LTE, end_date)
    )
    return And(*date_criteria)
