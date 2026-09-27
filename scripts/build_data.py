from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
SITE_DATA = ROOT / "site" / "data"
CONFIG_PATH = ROOT / "config" / "source_config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = json.load(f)

MONTHS = CONFIG["month_order"]
MONTH_INDEX = {m: i + 1 for i, m in enumerate(MONTHS)}
DAILY_UNRELIABLE_YEARS = set(CONFIG.get("daily_unreliable_years", []))

CORE_COLUMNS = [
    "سطح سه کالاها",
    "سطح چهار کالاها",
    "برند",
    "نام کالا",
    "بارکد",
    "سال",
    "ماه",
    "روز",
    "مبلغ فروش خالص پس از برگشتی",
    "قیمت فروش به مشتری واحد کالا",
    "تعداد کالای فروش پس از برگشتی",
    "وزن فروش رفته پس از برگشتی",
]

OPTIONAL_COLUMNS = [
    "مبلغ فروش خالص",
    "تعداد کارتن فروش رفته",
    "مبلغ تخفیفات فروش",
    "مجموع تعداد کالای فروش",
    "وزن فروش",
    "مبلغ خالص برگشتی",
    "تعداد کالاهای برگشتی",
    "وزن برگشتی",
    "تعداد فروشگاه",
]

NUMERIC_COLUMNS = [
    "مبلغ فروش خالص",
    "مبلغ فروش خالص پس از برگشتی",
    "قیمت فروش به مشتری واحد کالا",
    "تعداد کارتن فروش رفته",
    "تعداد کالای فروش پس از برگشتی",
    "وزن فروش رفته پس از برگشتی",
    "مبلغ تخفیفات فروش",
    "مجموع تعداد کالای فروش",
    "وزن فروش",
    "مبلغ خالص برگشتی",
    "تعداد کالاهای برگشتی",
    "وزن برگشتی",
    "تعداد فروشگاه",
]

SUM_COLUMNS = [c for c in NUMERIC_COLUMNS if c != "قیمت فروش به مشتری واحد کالا"]


def clean_text(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def normalize_barcode(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip()
    if not s or s.lower() == "nan" or s == "سرجمع":
        return ""
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def json_value(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def read_qv_xlsx(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_excel(path, sheet_name=0, dtype=object)
    df.columns = [clean_text(c) for c in df.columns]

    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: ستون‌های ضروری وجود ندارند: {missing}")

    present_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]

    for c in ["سال", "روز"] + NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["سطح سه کالاها", "سطح چهار کالاها", "برند", "نام کالا", "ماه"]:
        df[c] = df[c].map(clean_text)
    df["بارکد"] = df["بارکد"].map(normalize_barcode)

    mask = (
        df["سال"].notna()
        & df["روز"].notna()
        & df["ماه"].isin(MONTHS)
        & df["بارکد"].ne("")
        & df["برند"].ne("")
        & df["نام کالا"].ne("")
    )
    detail = df.loc[mask].copy()
    detail["سال"] = detail["سال"].astype(int)
    detail["روز"] = detail["روز"].astype(int)
    detail["شماره ماه"] = detail["ماه"].map(MONTH_INDEX).astype(int)

    bad_day = detail.loc[~detail["روز"].between(1, 31)]
    if not bad_day.empty:
        raise ValueError(f"{path.name}: {len(bad_day)} ردیف دارای روز نامعتبر است")

    dup_cols = ["سال", "ماه", "روز", "بارکد"]
    dup_count = int(detail.duplicated(dup_cols, keep=False).sum())
    if dup_count:
        sample = detail.loc[detail.duplicated(dup_cols, keep=False), dup_cols].head(10).to_dict("records")
        raise ValueError(f"{path.name}: {dup_count} ردیف تکراری در Grain روز × SKU. نمونه: {sample}")

    for c in NUMERIC_COLUMNS:
        if c not in detail.columns:
            detail[c] = pd.NA

    reconciliation = {}
    if all(c in present_optional for c in ["مبلغ فروش خالص", "مبلغ خالص برگشتی"]):
        diff = (
            detail["مبلغ فروش خالص"].fillna(0)
            - detail["مبلغ خالص برگشتی"].fillna(0)
            - detail["مبلغ فروش خالص پس از برگشتی"].fillna(0)
        )
        reconciliation["sales"] = {
            "max_abs_diff": float(diff.abs().max() or 0),
            "failed_rows": int((diff.abs() > 0.1).sum()),
        }
    if all(c in present_optional for c in ["مجموع تعداد کالای فروش", "تعداد کالاهای برگشتی"]):
        diff = (
            detail["مجموع تعداد کالای فروش"].fillna(0)
            - detail["تعداد کالاهای برگشتی"].fillna(0)
            - detail["تعداد کالای فروش پس از برگشتی"].fillna(0)
        )
        reconciliation["quantity"] = {
            "max_abs_diff": float(diff.abs().max() or 0),
            "failed_rows": int((diff.abs() > 1e-8).sum()),
        }
    if all(c in present_optional for c in ["وزن فروش", "وزن برگشتی"]):
        diff = (
            detail["وزن فروش"].fillna(0)
            - detail["وزن برگشتی"].fillna(0)
            - detail["وزن فروش رفته پس از برگشتی"].fillna(0)
        )
        reconciliation["weight"] = {
            "max_abs_diff": float(diff.abs().max() or 0),
            "failed_rows": int((diff.abs() > 1e-7).sum()),
        }

    info = {
        "file": path.name,
        "source_rows": int(len(df)),
        "detail_rows": int(len(detail)),
        "removed_summary_rows": int(len(df) - len(detail)),
        "columns": list(df.columns),
        "optional_columns_present": present_optional,
        "reconciliation": reconciliation,
    }
    return detail, info


def aggregate_monthly(detail: pd.DataFrame) -> pd.DataFrame:
    dims = [
        "سال", "شماره ماه", "ماه", "سطح سه کالاها", "سطح چهار کالاها",
        "برند", "نام کالا", "بارکد"
    ]
    agg_spec = {c: "sum" for c in SUM_COLUMNS}
    grouped = detail.groupby(dims, dropna=False, as_index=False).agg(agg_spec)

    qty = grouped["تعداد کالای فروش پس از برگشتی"].astype(float)
    net = grouped["مبلغ فروش خالص پس از برگشتی"].astype(float)
    grouped["قیمت فروش به مشتری واحد کالا"] = (net / qty.where(qty != 0)).replace([math.inf, -math.inf], pd.NA)
    return grouped


def aggregate_brand_monthly(monthly: pd.DataFrame) -> list[dict[str, Any]]:
    g = monthly.groupby(["سال", "شماره ماه", "ماه", "برند"], as_index=False)["مبلغ فروش خالص پس از برگشتی"].sum()
    return [
        {
            "سال": int(r["سال"]),
            "شماره ماه": int(r["شماره ماه"]),
            "ماه": r["ماه"],
            "برند": r["برند"],
            "مبلغ فروش خالص پس از برگشتی": float(r["مبلغ فروش خالص پس از برگشتی"] or 0),
        }
        for _, r in g.iterrows()
    ]


def frame_records(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    use = df if columns is None else df[columns]
    out = []
    for rec in use.to_dict("records"):
        out.append({k: json_value(v) for k, v in rec.items()})
    return out


def year_meta(detail: pd.DataFrame, monthly: pd.DataFrame, source_info: dict[str, Any]) -> dict[str, Any]:
    year = int(detail["سال"].mode().iloc[0])
    latest_month_no = int(detail["شماره ماه"].max())
    latest_month = MONTHS[latest_month_no - 1]
    latest_day = int(detail.loc[detail["شماره ماه"] == latest_month_no, "روز"].max())
    return {
        "year": year,
        "latest_month_no": latest_month_no,
        "latest_month": latest_month,
        "latest_day": latest_day,
        "daily_trend_available": year not in DAILY_UNRELIABLE_YEARS,
        "detail_rows": int(len(detail)),
        "monthly_rows": int(len(monthly)),
        "brand_count": int(detail["برند"].nunique()),
        "sku_count": int(detail["بارکد"].nunique()),
        "category_l3_count": int(detail["سطح سه کالاها"].nunique()),
        "category_l4_count": int(detail["سطح چهار کالاها"].nunique()),
        "net_sales": float(detail["مبلغ فروش خالص پس از برگشتی"].fillna(0).sum()),
        "source": source_info,
    }


def main() -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.xlsx"))
    if not files:
        raise SystemExit("هیچ فایل XLSX در raw/ پیدا نشد")

    all_monthly = []
    all_daily = []
    years_meta = []

    for path in files:
        detail, info = read_qv_xlsx(path)
        if detail.empty:
            raise ValueError(f"{path.name}: هیچ ردیف Detail معتبر پیدا نشد")

        years = sorted(detail["سال"].unique().tolist())
        if len(years) != 1:
            raise ValueError(f"{path.name}: انتظار یک سال در هر فایل داریم، اما سال‌ها: {years}")
        year = int(years[0])

        monthly = aggregate_monthly(detail)
        all_monthly.append(monthly)

        daily_cols = [
            "سال", "شماره ماه", "ماه", "روز", "سطح سه کالاها", "سطح چهار کالاها",
            "برند", "نام کالا", "بارکد",
        ] + NUMERIC_COLUMNS
        daily = detail[daily_cols].copy()
        all_daily.append(daily)
        years_meta.append(year_meta(detail, monthly, info))

    monthly_all = pd.concat(all_monthly, ignore_index=True).sort_values(["سال", "شماره ماه", "برند", "نام کالا"])
    daily_all = pd.concat(all_daily, ignore_index=True).sort_values(["سال", "شماره ماه", "روز", "برند", "نام کالا"])

    years_meta.sort(key=lambda x: x["year"])
    default_year = CONFIG.get("default_year")
    if default_year not in [x["year"] for x in years_meta]:
        default_year = years_meta[-1]["year"]

    dashboard = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_currency": CONFIG.get("source_currency", "IRR"),
            "default_year": default_year,
            "months": MONTHS,
            "years": years_meta,
        },
        "skuMonthly": frame_records(monthly_all),
        "brandMonthly": aggregate_brand_monthly(monthly_all),
    }

    out = SITE_DATA / "dashboard.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(dashboard, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    with (SITE_DATA / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(dashboard["meta"], f, ensure_ascii=False, indent=2, allow_nan=False)

    daily_dir = SITE_DATA / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    short_map = {
        "سال":"y", "شماره ماه":"mi", "ماه":"m", "روز":"d",
        "سطح سه کالاها":"c3", "سطح چهار کالاها":"c4", "برند":"b", "نام کالا":"n", "بارکد":"bc",
        "مبلغ فروش خالص":"sales", "مبلغ فروش خالص پس از برگشتی":"net", "قیمت فروش به مشتری واحد کالا":"price",
        "تعداد کارتن فروش رفته":"cartons", "تعداد کالای فروش پس از برگشتی":"netQty",
        "وزن فروش رفته پس از برگشتی":"netWeight", "مبلغ تخفیفات فروش":"discount",
        "مجموع تعداد کالای فروش":"salesQty", "وزن فروش":"salesWeight", "مبلغ خالص برگشتی":"returns",
        "تعداد کالاهای برگشتی":"returnQty", "وزن برگشتی":"returnWeight", "تعداد فروشگاه":"stores"
    }
    for y in sorted(daily_all["سال"].unique().tolist()):
        y = int(y)
        if y in DAILY_UNRELIABLE_YEARS:
            continue
        yd = daily_all[daily_all["سال"] == y].rename(columns=short_map)
        cols = [short_map[c] for c in daily_cols]
        records = frame_records(yd, cols)
        with (daily_dir / f"{y}.json").open("w", encoding="utf-8") as f:
            json.dump({"year": y, "rows": records}, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    print("\n✓ Build completed")
    print(f"  dashboard.json: {out.stat().st_size / 1024 / 1024:.2f} MB")
    for y in years_meta:
        print(
            f"  {y['year']}: {y['detail_rows']:,} daily rows | {y['monthly_rows']:,} monthly rows | "
            f"{y['sku_count']} SKUs | latest {y['latest_day']} {y['latest_month']}"
        )
        for name, rec in y["source"].get("reconciliation", {}).items():
            print(f"    reconciliation {name}: failed={rec['failed_rows']} max_abs_diff={rec['max_abs_diff']}")
            if rec["failed_rows"]:
                raise ValueError(f"{y['year']}: reconciliation failed for {name}")


if __name__ == "__main__":
    main()
