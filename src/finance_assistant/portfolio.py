"""Portfolio holdings (CAP-3): parse a holdings CSV and analyse diversification.

Pure and deterministic, with no LLM imports, so the chat agent and the UI's
Portfolio tab share one implementation. Nothing here is persisted: a parsed
``Portfolio`` lives only as long as the caller keeps it.

CSV format: a header row with a ``ticker`` column (alias ``symbol``) and a
``shares`` (alias ``quantity``) and/or ``value`` (alias ``market value``)
column; headers are case-insensitive. Each row gives exactly one of shares or
value (market value in dollars). ``$`` and thousands commas are allowed.
Duplicate tickers are merged. At most ``MAX_HOLDINGS`` rows and
``MAX_FILE_BYTES`` bytes.

Analysis (thresholds are the constants below; weights are compared as shown,
rounded to 0.1 %):

- concentration risk: one company (any ticker not on the bundled fund list) at
  ``SINGLE_COMPANY_LIMIT`` or more of the priced value; the (up to) three
  largest companies together at ``TOP_COMPANIES_LIMIT`` or more; or one
  narrow/sector fund at ``NARROW_FUND_LIMIT`` or more;
- diversification level: ``low`` if any concentration flag; ``high`` if broad
  US stock, international stock and bond funds together are
  ``BROAD_SHARE_FOR_HIGH`` or more and nothing is flagged; otherwise
  ``moderate``.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import cache
from importlib import resources
from typing import Literal

from .market_data import normalize_symbol

MAX_HOLDINGS = 50
MAX_FILE_BYTES = 1024 * 1024  # 1 MB
# Shares or dollar values above this are rejected (keeps every later Decimal operation in range).
MAX_AMOUNT = Decimal("1e12")

SINGLE_COMPANY_LIMIT = Decimal(20)
TOP_COMPANIES_LIMIT = Decimal(50)
TOP_COMPANIES_COUNT = 3
NARROW_FUND_LIMIT = Decimal(40)
BROAD_SHARE_FOR_HIGH = Decimal(50)

Category = Literal["company", "broad_us_stock", "international_stock", "bond", "narrow"]
FUND_CATEGORIES: tuple[Category, ...] = ("broad_us_stock", "international_stock", "bond", "narrow")
BROAD_CATEGORIES: frozenset[str] = frozenset({"broad_us_stock", "international_stock", "bond"})
STOCK_CATEGORIES: frozenset[str] = frozenset({"company", "broad_us_stock", "international_stock", "narrow"})

CATEGORY_LABELS: dict[str, str] = {
    "company": "Single company",
    "broad_us_stock": "Broad US stock fund",
    "international_stock": "International stock fund",
    "bond": "Bond fund",
    "narrow": "Narrow/sector fund",
}

Level = Literal["low", "moderate", "high"]
FlagKind = Literal["single_company", "top_companies", "narrow_fund"]

_CENT = Decimal("0.01")
_TENTH = Decimal("0.1")

# Header aliases, matched case-insensitively after collapsing spaces/underscores/hyphens.
_HEADER_ALIASES: dict[str, str] = {
    "ticker": "ticker",
    "symbol": "ticker",
    "shares": "shares",
    "quantity": "shares",
    "value": "value",
    "market value": "value",
}

_FUNDS_FILE = "funds.json"


# -- bundled fund list -------------------------------------------------------------


@dataclass(frozen=True)
class FundInfo:
    """One fund on the bundled list."""

    ticker: str
    name: str
    category: Category


@cache
def fund_list() -> dict[str, FundInfo]:
    """The bundled fund list, by ticker."""
    text = resources.files(__package__).joinpath(_FUNDS_FILE).read_text(encoding="utf-8")
    funds = json.loads(text)["funds"]
    result: dict[str, FundInfo] = {}
    for ticker, row in funds.items():
        if row["category"] not in FUND_CATEGORIES:
            raise ValueError(f"unknown fund category for {ticker}: {row['category']!r}")
        result[ticker] = FundInfo(ticker, row["name"], row["category"])
    return result


def category_of(ticker: str) -> Category:
    """A fund's category from the bundled list; any other ticker is treated as a single company."""
    fund = fund_list().get(ticker)
    return fund.category if fund is not None else "company"


# -- holdings ----------------------------------------------------------------------


def _positive(value: Decimal | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if value <= 0:
        raise ValueError(f"{name} must be more than 0")
    if value > MAX_AMOUNT:
        raise ValueError(f"{name} must be at most {MAX_AMOUNT:,f}")


@dataclass(frozen=True)
class Holding:
    """One ticker and how much of it: a number of shares, a market value in dollars, or both
    (both only after merging duplicate rows)."""

    ticker: str
    shares: Decimal | None = None
    value: Decimal | None = None

    def __post_init__(self) -> None:
        if normalize_symbol(self.ticker) != self.ticker:
            raise ValueError(f"ticker must be a normalized symbol, got {self.ticker!r}")
        if self.shares is None and self.value is None:
            raise ValueError("a holding needs shares or a value")
        _positive(self.shares, "shares")
        _positive(self.value, "value")


def _add(a: Decimal | None, b: Decimal | None) -> Decimal | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def merge_holdings(holdings: Iterable[Holding]) -> tuple[Holding, ...]:
    """Merge holdings with the same ticker (shares added to shares, values to values), first order kept."""
    merged: dict[str, Holding] = {}
    for h in holdings:
        prior = merged.get(h.ticker)
        merged[h.ticker] = h if prior is None else Holding(
            h.ticker, _add(prior.shares, h.shares), _add(prior.value, h.value)
        )
    return tuple(merged.values())


@dataclass(frozen=True)
class Portfolio:
    """An uploaded or typed portfolio: 1 to ``MAX_HOLDINGS`` holdings with distinct tickers.

    Session-only: never persisted.
    """

    holdings: tuple[Holding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.holdings, tuple) or not all(isinstance(h, Holding) for h in self.holdings):
            raise ValueError("holdings must be a tuple of Holding")
        if not self.holdings:
            raise ValueError("a portfolio needs at least one holding")
        if len(self.holdings) > MAX_HOLDINGS:
            raise ValueError(f"a portfolio can have at most {MAX_HOLDINGS} holdings")
        if len({h.ticker for h in self.holdings}) != len(self.holdings):
            raise ValueError("tickers must be distinct (use merge_holdings)")


# -- CSV parsing -------------------------------------------------------------------


@dataclass(frozen=True)
class HoldingsProblem:
    """One problem with a holdings file: the file row (1 = header) or None for the whole file."""

    row: int | None
    message: str

    def __str__(self) -> str:
        return self.message if self.row is None else f"Row {self.row}: {self.message}"


class HoldingsFormatError(ValueError):
    """Raised by ``parse_holdings_csv`` with every problem found, by row."""

    def __init__(self, problems: Sequence[HoldingsProblem]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(str(p) for p in self.problems))


def _header_key(name: str) -> str:
    return " ".join(name.replace("_", " ").replace("-", " ").lower().split())


def _shown(text: str) -> str:
    text = text.strip()
    return text if len(text) <= 20 else text[:20] + "…"


def parse_amount(text: str, name: str) -> Decimal:
    """A positive amount from a cell such as ``"$1,234.50"``; ValueError with a user-facing message."""
    cleaned = text.strip().replace("$", "").replace(",", "").replace(" ", "")
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        raise ValueError(f"{name} '{_shown(text)}' isn't a number") from None
    if not amount.is_finite():
        raise ValueError(f"{name} '{_shown(text)}' isn't a number")
    if amount < 0:
        raise ValueError(f"{name} can't be negative ('{_shown(text)}')")
    if amount == 0:
        raise ValueError(f"{name} must be more than 0")
    if amount > MAX_AMOUNT:
        raise ValueError(f"{name} is too large ('{_shown(text)}')")
    return amount


def parse_ticker(text: str) -> str:
    """A normalized ticker from a cell (a leading ``$`` is allowed); ValueError with a user-facing message."""
    cleaned = text.strip().lstrip("$").strip()
    if not cleaned:
        raise ValueError("the ticker is empty")
    try:
        return normalize_symbol(cleaned)
    except ValueError:
        raise ValueError(f"'{_shown(text)}' isn't a valid ticker symbol") from None


def _decode(data: str | bytes) -> str:
    if isinstance(data, bytes):
        if len(data) > MAX_FILE_BYTES:
            raise HoldingsFormatError([HoldingsProblem(None, "The file is larger than 1 MB.")])
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HoldingsFormatError(
                [HoldingsProblem(None, "The file isn't a UTF-8 text (CSV) file.")]
            ) from None
    if isinstance(data, str):
        if len(data.encode("utf-8")) > MAX_FILE_BYTES:
            raise HoldingsFormatError([HoldingsProblem(None, "The file is larger than 1 MB.")])
        return data.removeprefix("﻿")
    raise TypeError("data must be str or bytes")


def parse_holdings_csv(data: str | bytes) -> Portfolio:
    """Parse a holdings CSV into a ``Portfolio``; raise ``HoldingsFormatError`` listing every problem."""
    text = _decode(data)
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        rows = [(reader.line_num, row) for row in reader]
    except csv.Error as exc:
        raise HoldingsFormatError([HoldingsProblem(None, f"The file isn't valid CSV ({exc}).")]) from None
    rows = [(n, r) for n, r in rows if any(cell.strip() for cell in r)]
    if not rows:
        raise HoldingsFormatError([HoldingsProblem(None, "The file is empty.")])

    header_line, header = rows[0]
    columns: dict[str, int] = {}
    problems: list[HoldingsProblem] = []
    for index, name in enumerate(header):
        key = _HEADER_ALIASES.get(_header_key(name))
        if key is None:
            continue  # extra columns are ignored
        if key in columns:
            problems.append(HoldingsProblem(header_line, f"more than one {key} column ('{_shown(name)}')"))
        else:
            columns[key] = index
    if "ticker" not in columns:
        problems.append(HoldingsProblem(header_line, "no ticker column (name it 'ticker' or 'symbol')"))
    if "shares" not in columns and "value" not in columns:
        problems.append(
            HoldingsProblem(header_line, "no shares or value column (name them 'shares' and/or 'value')")
        )
    data_rows = rows[1:]
    if not data_rows and not problems:
        problems.append(HoldingsProblem(None, "The file has a header but no holdings rows."))
    if len(data_rows) > MAX_HOLDINGS:
        problems.append(
            HoldingsProblem(None, f"The file has {len(data_rows)} holdings rows; the limit is {MAX_HOLDINGS}.")
        )
    if problems:
        raise HoldingsFormatError(problems)

    def cell(row: list[str], key: str) -> str:
        index = columns.get(key)
        return row[index].strip() if index is not None and index < len(row) else ""

    holdings: list[Holding] = []
    for line, row in data_rows:
        row_problems: list[str] = []
        ticker = shares = value = None
        try:
            ticker = parse_ticker(cell(row, "ticker"))
        except ValueError as exc:
            row_problems.append(str(exc))
        shares_text, value_text = cell(row, "shares"), cell(row, "value")
        if shares_text and value_text:
            row_problems.append("give either shares or value, not both")
        elif not shares_text and not value_text:
            row_problems.append("give shares or value")
        else:
            try:
                if shares_text:
                    shares = parse_amount(shares_text, "shares")
                else:
                    value = parse_amount(value_text, "value")
            except ValueError as exc:
                row_problems.append(str(exc))
        if row_problems:
            problems.append(HoldingsProblem(line, "; ".join(row_problems)))
        else:
            assert ticker is not None
            holdings.append(Holding(ticker, shares, value))
    if problems:
        raise HoldingsFormatError(problems)
    try:
        merged = merge_holdings(holdings)
    except ValueError:
        raise HoldingsFormatError(
            [HoldingsProblem(None, f"The rows for one ticker add up to more than {MAX_AMOUNT:,f}.")]
        ) from None
    return Portfolio(merged)


# -- analysis ------------------------------------------------------------------------


@dataclass(frozen=True)
class PricedHolding:
    """A holding with its dollar value. ``price``/``price_source`` are set when shares were priced."""

    ticker: str
    value: Decimal
    shares: Decimal | None = None
    price: Decimal | None = None
    price_source: str | None = None  # a market-data QuoteSource; None for a value given as is
    given_value: Decimal | None = None  # the part of ``value`` given directly in dollars


def to_cents(value: Decimal) -> Decimal:
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def _pct(value: Decimal) -> Decimal:
    return value.quantize(_TENTH, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class AnalyzedHolding:
    """One priced holding with its weight (percent of priced value, rounded to 0.1)."""

    holding: PricedHolding
    category: Category
    weight: Decimal

    @property
    def ticker(self) -> str:
        return self.holding.ticker

    @property
    def is_fund(self) -> bool:
        return self.category != "company"


@dataclass(frozen=True)
class ConcentrationFlag:
    """One concentration-risk finding: what kind, which tickers, and their share (percent)."""

    kind: FlagKind
    tickers: tuple[str, ...]
    share: Decimal


@dataclass(frozen=True)
class PortfolioAnalysis:
    """The deterministic analysis of the priced holdings (all percentages rounded to 0.1)."""

    rows: tuple[AnalyzedHolding, ...]  # largest value first
    total_value: Decimal
    largest: AnalyzedHolding
    top_companies: tuple[str, ...]  # up to TOP_COMPANIES_COUNT largest companies
    top_companies_share: Decimal
    category_shares: dict[str, Decimal] = field(default_factory=dict)  # every Category, 0 if absent
    broad_share: Decimal = Decimal(0)  # broad US stock + international stock + bond funds
    stock_share: Decimal = Decimal(0)  # companies + stock funds
    bond_share: Decimal = Decimal(0)
    flags: tuple[ConcentrationFlag, ...] = ()
    level: Level = "moderate"

    @property
    def companies(self) -> tuple[AnalyzedHolding, ...]:
        return tuple(r for r in self.rows if r.category == "company")


def analyze(priced: Sequence[PricedHolding]) -> PortfolioAnalysis:
    """Weights, top holdings, fund breakdown, concentration flags, and diversification level.

    Raises ValueError when there is nothing with a positive value to analyse.
    """
    if not priced:
        raise ValueError("nothing to analyse: no priced holdings")
    if len({p.ticker for p in priced}) != len(priced):
        raise ValueError("priced holdings must have distinct tickers")
    for p in priced:
        if not isinstance(p.value, Decimal) or not p.value.is_finite() or p.value <= 0:
            raise ValueError(f"{p.ticker}: value must be a positive Decimal")
    total = sum((p.value for p in priced), Decimal(0))

    raw = {p.ticker: p.value / total * 100 for p in priced}
    order = sorted(range(len(priced)), key=lambda i: (-priced[i].value, i))
    rows = tuple(
        AnalyzedHolding(priced[i], category_of(priced[i].ticker), _pct(raw[priced[i].ticker])) for i in order
    )

    def share(categories: Iterable[str]) -> Decimal:
        wanted = set(categories)
        return _pct(sum((raw[r.ticker] for r in rows if r.category in wanted), Decimal(0)))

    category_shares = {c: share([c]) for c in ("company", *FUND_CATEGORIES)}
    companies = [r for r in rows if r.category == "company"]
    top = companies[:TOP_COMPANIES_COUNT]
    top_share = _pct(sum((raw[r.ticker] for r in top), Decimal(0)))

    flags: list[ConcentrationFlag] = []
    for r in companies:
        if r.weight >= SINGLE_COMPANY_LIMIT:
            flags.append(ConcentrationFlag("single_company", (r.ticker,), r.weight))
    if len(top) >= 2 and top_share >= TOP_COMPANIES_LIMIT:
        flags.append(ConcentrationFlag("top_companies", tuple(r.ticker for r in top), top_share))
    for r in rows:
        if r.category == "narrow" and r.weight >= NARROW_FUND_LIMIT:
            flags.append(ConcentrationFlag("narrow_fund", (r.ticker,), r.weight))

    broad = share(BROAD_CATEGORIES)
    level: Level
    if flags:
        level = "low"
    elif broad >= BROAD_SHARE_FOR_HIGH:
        level = "high"
    else:
        level = "moderate"

    return PortfolioAnalysis(
        rows=rows,
        total_value=total,
        largest=rows[0],
        top_companies=tuple(r.ticker for r in top),
        top_companies_share=top_share,
        category_shares=category_shares,
        broad_share=broad,
        stock_share=share(STOCK_CATEGORIES),
        bond_share=share(["bond"]),
        flags=tuple(flags),
        level=level,
    )


__all__ = [
    "BROAD_SHARE_FOR_HIGH",
    "CATEGORY_LABELS",
    "AnalyzedHolding",
    "ConcentrationFlag",
    "FundInfo",
    "Holding",
    "HoldingsFormatError",
    "HoldingsProblem",
    "MAX_AMOUNT",
    "MAX_FILE_BYTES",
    "MAX_HOLDINGS",
    "NARROW_FUND_LIMIT",
    "Portfolio",
    "PortfolioAnalysis",
    "PricedHolding",
    "SINGLE_COMPANY_LIMIT",
    "TOP_COMPANIES_COUNT",
    "TOP_COMPANIES_LIMIT",
    "analyze",
    "category_of",
    "fund_list",
    "merge_holdings",
    "parse_amount",
    "parse_holdings_csv",
    "parse_ticker",
    "to_cents",
]
