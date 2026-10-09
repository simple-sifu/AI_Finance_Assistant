"""Portfolio parsing and analysis: CSV matrix rows, aliases, merging, limits, threshold edges."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from finance_assistant.portfolio import (
    MAX_FILE_BYTES,
    MAX_HOLDINGS,
    Holding,
    HoldingsFormatError,
    Portfolio,
    PricedHolding,
    analyze,
    category_of,
    fund_list,
    merge_holdings,
    parse_holdings_csv,
)

SAMPLE = Path(__file__).parent / "data" / "sample_holdings.csv"


def problems(data: str | bytes) -> list[str]:
    with pytest.raises(HoldingsFormatError) as info:
        parse_holdings_csv(data)
    return [str(p) for p in info.value.problems]


def priced(**values: float | int | str) -> list[PricedHolding]:
    return [PricedHolding(t, Decimal(str(v))) for t, v in values.items()]


# --- parsing ------------------------------------------------------------------------


def test_sample_file_parses_mixing_shares_and_values() -> None:
    portfolio = parse_holdings_csv(SAMPLE.read_bytes())
    assert portfolio.holdings == (
        Holding("VOO", shares=Decimal(10)),
        Holding("AAPL", value=Decimal("7250.00")),
        Holding("BND", shares=Decimal(40)),
        Holding("VXUS", value=Decimal(3000)),
        Holding("MSFT", shares=Decimal(5)),
        Holding("NVDA", value=Decimal(1200)),
    )
    assert parse_holdings_csv(SAMPLE.read_text()) == portfolio


@pytest.mark.parametrize(
    "header",
    ["ticker,shares,value", "Ticker,Shares,Value", "SYMBOL,Quantity,Market Value", "symbol,quantity,market_value"],
)
def test_header_aliases_are_case_insensitive(header: str) -> None:
    portfolio = parse_holdings_csv(f"{header}\nvoo,3,\n$aapl,,\"$1,234.50\"\n")
    assert portfolio.holdings == (Holding("VOO", shares=Decimal(3)), Holding("AAPL", value=Decimal("1234.50")))


def test_only_one_amount_column_and_extra_columns_are_fine() -> None:
    assert parse_holdings_csv("Account,Ticker,Value,Notes\nIRA,VTI,500,x\n").holdings == (
        Holding("VTI", value=Decimal(500)),
    )
    assert parse_holdings_csv("ticker,shares\nVTI,1.5\n").holdings == (Holding("VTI", shares=Decimal("1.5")),)


def test_bom_blank_lines_and_crlf_are_tolerated() -> None:
    data = "﻿ticker,shares\r\n\r\nVTI,2\r\n,\r\n".encode()
    assert parse_holdings_csv(data).holdings == (Holding("VTI", shares=Decimal(2)),)


def test_duplicate_tickers_are_merged_in_first_order() -> None:
    data = "ticker,shares,value\nVOO,10,\nAAPL,,100\nvoo,5,\nVOO,,50\nAAPL,,25\n"
    assert parse_holdings_csv(data).holdings == (
        Holding("VOO", shares=Decimal(15), value=Decimal(50)),
        Holding("AAPL", value=Decimal(125)),
    )


def test_missing_ticker_column() -> None:
    assert problems("name,shares\nVOO,1\n") == ["Row 1: no ticker column (name it 'ticker' or 'symbol')"]


def test_missing_amount_columns() -> None:
    assert problems("ticker,price\nVOO,1\n") == [
        "Row 1: no shares or value column (name them 'shares' and/or 'value')"
    ]


def test_every_bad_row_is_listed_by_row() -> None:
    data = (
        "ticker,shares,value\n"
        "VOO,10,\n"  # row 2 ok
        "AAPL,5,500\n"  # both
        "MSFT,,\n"  # neither
        "BND,-3,\n"  # negative
        "VTI,,abc\n"  # non-numeric
        ",4,\n"  # no ticker
        "BAD TICKER!,1,\n"  # invalid ticker
        "QQQ,0,\n"  # zero
    )
    assert problems(data) == [
        "Row 3: give either shares or value, not both",
        "Row 4: give shares or value",
        "Row 5: shares can't be negative ('-3')",
        "Row 6: value 'abc' isn't a number",
        "Row 7: the ticker is empty",
        "Row 8: 'BAD TICKER!' isn't a valid ticker symbol",
        "Row 9: shares must be more than 0",
    ]


def test_huge_amounts_are_rejected_per_row() -> None:
    assert problems("ticker,value\nVOO,1e30\nVTI,1000000000001\nBND,1000000000000\n") == [
        "Row 2: value is too large ('1e30')",
        "Row 3: value is too large ('1000000000001')",
    ]
    assert problems("ticker,shares\nVOO,1e13\n") == ["Row 2: shares is too large ('1e13')"]


def test_duplicate_rows_adding_up_past_the_limit_are_rejected() -> None:
    assert problems("ticker,value\nVOO,900000000000\nVOO,900000000000\n") == [
        "The rows for one ticker add up to more than 1,000,000,000,000."
    ]


def test_non_finite_amount_is_not_a_number() -> None:
    assert problems("ticker,value\nVOO,NaN\nVTI,Infinity\n") == [
        "Row 2: value 'NaN' isn't a number",
        "Row 3: value 'Infinity' isn't a number",
    ]


def test_row_limit_at_the_edge() -> None:
    rows = "".join(f"T{i},1\n" for i in range(MAX_HOLDINGS))
    assert len(parse_holdings_csv("ticker,shares\n" + rows).holdings) == MAX_HOLDINGS
    assert problems("ticker,shares\n" + rows + "EXTRA,1\n") == [
        f"The file has {MAX_HOLDINGS + 1} holdings rows; the limit is {MAX_HOLDINGS}."
    ]


def test_size_limit_at_the_edge() -> None:
    header = "ticker,shares\nVOO,1\n"
    padded = header + "\n" * (MAX_FILE_BYTES - len(header))
    assert parse_holdings_csv(padded.encode()).holdings == (Holding("VOO", shares=Decimal(1)),)
    assert problems((padded + "\n").encode()) == ["The file is larger than 1 MB."]
    assert problems(padded + "\n") == ["The file is larger than 1 MB."]


def test_empty_and_header_only_and_non_utf8() -> None:
    assert problems("") == ["The file is empty."]
    assert problems("ticker,shares\n") == ["The file has a header but no holdings rows."]
    assert problems(b"\xff\xfe\x00bad") == ["The file isn't a UTF-8 text (CSV) file."]


def test_duplicate_columns_are_reported() -> None:
    assert problems("ticker,symbol,shares\nVOO,VOO,1\n") == ["Row 1: more than one ticker column ('symbol')"]


def test_portfolio_and_holding_validate_themselves() -> None:
    with pytest.raises(ValueError):
        Holding("voo", shares=Decimal(1))  # not normalized
    with pytest.raises(ValueError):
        Holding("VOO")
    with pytest.raises(ValueError):
        Holding("VOO", value=Decimal(-1))
    with pytest.raises(ValueError):
        Portfolio(())
    with pytest.raises(ValueError):
        Portfolio((Holding("VOO", value=Decimal(1)), Holding("VOO", value=Decimal(2))))
    assert merge_holdings([Holding("VOO", value=Decimal(1)), Holding("VOO", shares=Decimal(2))]) == (
        Holding("VOO", shares=Decimal(2), value=Decimal(1)),
    )


# --- fund list ------------------------------------------------------------------------


def test_bundled_fund_list_categories() -> None:
    funds = fund_list()
    assert 30 <= len(funds) <= 40
    assert {f.category for f in funds.values()} == {"broad_us_stock", "international_stock", "bond", "narrow"}
    assert category_of("VOO") == "broad_us_stock"
    assert category_of("VXUS") == "international_stock"
    assert category_of("BND") == "bond"
    assert category_of("XLE") == "narrow"
    assert category_of("AAPL") == "company"  # unknown tickers are single companies


@pytest.mark.parametrize(
    ("ticker", "category"),
    [
        ("VFIAX", "broad_us_stock"),
        ("FZROX", "broad_us_stock"),
        ("FSKAX", "broad_us_stock"),
        ("SWTSX", "broad_us_stock"),
        ("VTIAX", "international_stock"),
        ("FTIHX", "international_stock"),
        ("VBTLX", "bond"),
        ("FXNAX", "bond"),
        ("IWM", "narrow"),
        ("SCHD", "narrow"),
        ("VUG", "narrow"),
        ("VTV", "narrow"),
    ],
)
def test_common_mutual_funds_and_etfs_are_not_companies(ticker: str, category: str) -> None:
    assert category_of(ticker) == category
    analysis = analyze(priced(**{ticker: 39, "VTI": 61}))  # 39 %: under the narrow-fund limit too
    assert {r.ticker: r.category for r in analysis.rows}[ticker] == category
    assert not any(f.kind in ("single_company", "top_companies") for f in analysis.flags)
    assert analysis.top_companies == ()


# --- analysis -----------------------------------------------------------------------


def test_sample_analysis_flags_the_thirty_percent_stock() -> None:
    analysis = analyze(
        [
            PricedHolding("VOO", Decimal("7025.50")),
            PricedHolding("AAPL", Decimal("7250")),
            PricedHolding("BND", Decimal("2984.80")),
            PricedHolding("VXUS", Decimal("3000")),
            PricedHolding("MSFT", Decimal("2693.75")),
            PricedHolding("NVDA", Decimal("1200")),
        ]
    )
    assert analysis.total_value == Decimal("24154.05")
    assert [r.ticker for r in analysis.rows] == ["AAPL", "VOO", "VXUS", "BND", "MSFT", "NVDA"]
    assert analysis.largest.ticker == "AAPL" and analysis.largest.weight == Decimal("30.0")
    assert analysis.top_companies == ("AAPL", "MSFT", "NVDA")
    assert analysis.top_companies_share == Decimal("46.1")
    assert [(f.kind, f.tickers) for f in analysis.flags] == [("single_company", ("AAPL",))]
    assert analysis.level == "low"
    assert analysis.bond_share == Decimal("12.4") and analysis.stock_share == Decimal("87.6")
    assert analysis.broad_share == Decimal("53.9")
    assert analysis.category_shares["company"] == Decimal("46.1")
    assert analysis.category_shares["narrow"] == 0


def test_broad_fund_only_is_high_and_not_single_company_risk() -> None:
    analysis = analyze(priced(VOO=10000))
    assert analysis.flags == ()
    assert analysis.level == "high"
    assert analysis.largest.weight == Decimal("100.0") and analysis.largest.is_fund
    assert analysis.top_companies == ()


@pytest.mark.parametrize(
    ("aapl", "flagged"),
    [(20, True), ("19.96", True), ("19.94", False)],  # compared as shown, at 0.1 %
)
def test_single_company_threshold_edge(aapl, flagged: bool) -> None:
    rest = 100 - Decimal(str(aapl))
    analysis = analyze(priced(AAPL=aapl, VTI=rest))
    assert any(f.kind == "single_company" for f in analysis.flags) is flagged
    assert analysis.level == ("low" if flagged else "high")


def test_top_three_companies_threshold_edge() -> None:
    at = analyze(priced(AAPL=19, MSFT=19, NVDA=12, AMZN=10, VTI=40))
    assert [(f.kind, f.tickers, f.share) for f in at.flags] == [
        ("top_companies", ("AAPL", "MSFT", "NVDA"), Decimal("50.0"))
    ]
    below = analyze(priced(AAPL=19, MSFT=19, NVDA="11.9", AMZN="10.1", VTI=40))
    assert below.flags == ()
    assert below.level == "moderate"  # broad funds 40 % < 50 %


def test_top_companies_needs_two_companies() -> None:
    analysis = analyze(priced(AAPL=60, VTI=40))
    assert [f.kind for f in analysis.flags] == ["single_company"]


def test_narrow_fund_threshold_edge() -> None:
    at = analyze(priced(QQQ=40, VTI=60))
    assert [(f.kind, f.tickers) for f in at.flags] == [("narrow_fund", ("QQQ",))]
    assert at.level == "low"
    below = analyze(priced(QQQ="39.9", VTI="60.1"))
    assert below.flags == () and below.level == "high"


def test_level_high_needs_broad_funds_at_half() -> None:
    assert analyze(priced(VTI=50, AAPL=10, MSFT=10, NVDA=10, AMZN=10, GOOGL=10)).level == "high"
    assert analyze(priced(VTI="49.9", AAPL="10.1", MSFT=10, NVDA=10, AMZN=10, GOOGL=10)).level == "moderate"
    many = {f"C{i}": 1 for i in range(10)}
    assert analyze(priced(**many)).level == "moderate"  # spread over companies, but no broad funds


def test_bonds_count_as_broad_and_bond_share() -> None:
    analysis = analyze(priced(BND=60, VXUS=40))
    assert analysis.bond_share == Decimal("60.0") and analysis.stock_share == Decimal("40.0")
    assert analysis.level == "high"


def test_analyze_rejects_empty_or_bad_input() -> None:
    with pytest.raises(ValueError):
        analyze([])
    with pytest.raises(ValueError):
        analyze([PricedHolding("VOO", Decimal(0))])
    with pytest.raises(ValueError):
        analyze(priced(VOO=1) + priced(VOO=2))
