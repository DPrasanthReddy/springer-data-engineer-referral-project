#!/usr/bin/env python3
"""
main.py - Full Pandas pipeline for Referral Program (Take-Home Test)

Usage:
    python main.py --input . --output ./output
"""

import argparse
import pandas as pd
import numpy as np
import pytz
import re
from pathlib import Path
# opt in to pandas future behavior and silence the .fillna downcasting warning
pd.set_option('future.no_silent_downcasting', True)


# ---------- Utility functions ----------

def to_datetime_safe(df, col):
    # Parse as naive datetime (no timezone) so we can convert to local later
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def initcap_series(s):
    return s.astype(str).str.title().replace({'Nan': '', 'None': ''})

def parse_reward_value(x):
    # Return numeric value if possible, else NaN
    if pd.isna(x):
        return np.nan
    # direct numeric
    try:
        return float(x)
    except Exception:
        pass
    # extract first numeric substring
    try:
        m = re.search(r"[-+]?\d*\.?\d+", str(x))
        if m:
            return float(m.group(0))
    except Exception:
        pass
    return np.nan

def convert_utc_to_local(ts, tz_str):
    """
    Treat ts as UTC (naive datetime), convert to given timezone, return tz-naive local time.
    """
    try:
        if pd.isna(ts):
            return pd.NaT
        # if ts is naive, localize to UTC; if tz-aware, convert to UTC
        if getattr(ts, "tzinfo", None) is None:
            ts_utc = ts.tz_localize("UTC")
        else:
            ts_utc = ts.tz_convert("UTC")
        local = ts_utc.tz_convert(pytz.timezone(tz_str))
        # return timezone-naive local time (drop tzinfo)
        return local.tz_convert(None)
    except Exception:
        return ts

# Business logic evaluator returns (bool, reason_code)
def evaluate_business_logic_and_reason(row, merged_row):
    """
    Implements the Valid and Invalid conditions from the assignment PDF.
    Returns: (is_valid (bool), reason (str))
    """
    # Pull fields safely
    reward_val = row.get("reward_value")

    status = row.get("referral_status")
    status = str(status).strip() if pd.notna(status) else None

    tx_id = row.get("transaction_id")

    tx_status = row.get("transaction_status")
    tx_status = str(tx_status).strip() if pd.notna(tx_status) else None

    tx_type = row.get("transaction_type")
    tx_type = str(tx_type).strip() if pd.notna(tx_type) else None

    referral_at = row.get("referral_at")
    transaction_at = row.get("transaction_at")
    reward_granted_at = row.get("reward_granted_at") if "reward_granted_at" in row.index else None

    # Helper: referrer meta (membership_expired_date, is_deleted)
    ref_mem_exp = merged_row.get("referrer_membership_expired_date") if "referrer_membership_expired_date" in merged_row else None
    ref_deleted = merged_row.get("referrer_is_deleted") if "referrer_is_deleted" in merged_row else None

    # Convert reward_val to float when possible
    rvf = None
    if pd.notna(reward_val):
        try:
            rvf = float(reward_val)
        except Exception:
            rvf = None

    # --- Valid Condition 1 ---
    try:
        if rvf is not None and rvf > 0:
            if status and status.lower() == "berhasil":
                if pd.notna(tx_id):
                    if tx_status and tx_status.upper() == "PAID":
                        if tx_type and tx_type.upper() == "NEW":
                            if pd.notna(transaction_at) and pd.notna(referral_at):
                                same_month = (transaction_at.year == referral_at.year and transaction_at.month == referral_at.month)
                                after = (transaction_at >= referral_at)
                                if same_month and after:
                                    # membership not expired
                                    mem_ok = True
                                    if pd.notna(ref_mem_exp):
                                        try:
                                            mem_ok = pd.to_datetime(ref_mem_exp) >= referral_at
                                        except Exception:
                                            mem_ok = True
                                    # not deleted
                                    not_deleted = True
                                    if pd.notna(ref_deleted):
                                        not_deleted = (str(ref_deleted).lower() not in ["true", "1", "yes"])
                                    # reward granted
                                    granted = pd.notna(reward_granted_at)
                                    if mem_ok and not_deleted and granted:
                                        return True, "valid_cond_1_all_checks_passed"
    except Exception:
        pass

    # --- Valid Condition 2 ---
    # status: Menunggu / Tidak Berhasil, and no reward value
    no_reward = (
        pd.isna(reward_val) or
        (rvf is not None and rvf == 0) or
        (rvf is None and str(reward_val).strip() in ["0", "0.0", ""])
    )
    if status and status.lower() in ["menunggu", "tidak berhasil"] and no_reward:
        return True, "valid_cond_2_pending_or_failed_no_reward"

    # --- Invalid Condition 1 ---
    # reward_value > 0 and status not Berhasil
    if rvf is not None and rvf > 0 and not (status and status.lower() == "berhasil"):
        return False, "invalid_cond_1_reward_positive_status_not_berhasil"

    # --- Invalid Condition 2 ---
    # reward_value > 0 and no transaction id
    if rvf is not None and rvf > 0 and pd.isna(tx_id):
        return False, "invalid_cond_2_reward_positive_no_transaction"

    # --- Invalid Condition 3 ---
    # no reward value AND transaction exists and PAID AND transaction_at >= referral_at
    if no_reward and pd.notna(tx_id) and tx_status and tx_status.upper() == "PAID":
        if pd.notna(transaction_at) and pd.notna(referral_at):
            if transaction_at >= referral_at:
                return False, "invalid_cond_3_no_reward_but_paid_transaction"

    # --- Invalid Condition 4 ---
    # status Berhasil and reward_value is null or 0
    if status and status.lower() == "berhasil" and (pd.isna(reward_val) or (rvf is not None and rvf == 0)):
        return False, "invalid_cond_4_berhasil_but_no_reward"

    # --- Invalid Condition 5 ---
    # transaction occurred before referral
    if pd.notna(transaction_at) and pd.notna(referral_at) and transaction_at < referral_at:
        return False, "invalid_cond_5_transaction_before_referral"

    # Fallback: mark invalid with unknown reason
    return False, "unknown_invalid_or_unmatched_rules"

# ---------- Main pipeline ----------

def run_pipeline(input_folder: str, output_folder: str):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Expecting files in input_path
    files = {
        "lead_logs": input_path / "lead_log.csv",
        "paid_transactions": input_path / "paid_transactions.csv",
        "referral_rewards": input_path / "referral_rewards.csv",
        "user_logs": input_path / "user_logs.csv",
        "user_referral_logs": input_path / "user_referral_logs.csv",
        "user_referral_statuses": input_path / "user_referral_statuses.csv",
        "user_referrals": input_path / "user_referrals.csv"
    }

    # Load CSVs (empty DataFrame if missing)
    dfs = {}
    for k, p in files.items():
        if p.exists():
            dfs[k] = pd.read_csv(p)
        else:
            dfs[k] = pd.DataFrame()
            print(f"Warning: {p} not found; creating empty DataFrame for {k}")

    # Parse datetimes as naive (no tz)
    to_datetime_safe(dfs["user_referrals"], "referral_at")
    to_datetime_safe(dfs["user_referrals"], "updated_at")
    to_datetime_safe(dfs["user_referral_logs"], "created_at")
    if "reward_granted_at" in dfs["user_referral_logs"].columns:
        to_datetime_safe(dfs["user_referral_logs"], "reward_granted_at")
    to_datetime_safe(dfs["paid_transactions"], "transaction_at")
    if "membership_expired_date" in dfs["user_logs"].columns:
        dfs["user_logs"]["membership_expired_date"] = pd.to_datetime(
            dfs["user_logs"]["membership_expired_date"], errors="coerce"
        )

    # Profile tables (null counts, distinct counts)
    profiling_rows = []
    for name, df in dfs.items():
        if df.empty:
            profiling_rows.append({"table": name, "column": None, "dtype": None,
                                   "num_rows": 0, "null_count": 0, "distinct_count": 0})
        else:
            for col in df.columns:
                profiling_rows.append({
                    "table": name,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "num_rows": len(df),
                    "null_count": int(df[col].isna().sum()),
                    "distinct_count": int(df[col].nunique(dropna=True))
                })
    pd.DataFrame(profiling_rows).to_csv(output_path / "profiling_report.csv", index=False)
    print("Wrote profiling_report.csv")

    # Cleaning: initcap for names, leave homeclub as-is
    for dfname, cols in [
        ("user_referrals", ["referrer_name", "referee_name", "referee_phone"]),
        ("user_logs", ["name", "phone_number"]),
        ("lead_logs", ["preferred_location", "current_status"])
    ]:
        if dfname in dfs and not dfs[dfname].empty:
            for c in cols:
                if c in dfs[dfname].columns:
                    dfs[dfname][c] = initcap_series(dfs[dfname][c])

    if "homeclub" in dfs["user_logs"].columns:
        dfs["user_logs"]["homeclub"] = dfs["user_logs"]["homeclub"].astype(str).str.strip()

    # Merge according to PDF
    ur = dfs["user_referrals"].copy()
    ufl = dfs["user_referral_logs"].copy()
    rr = dfs["referral_rewards"].copy()
    pt = dfs["paid_transactions"].copy()
    ul = dfs["user_logs"].copy()
    ll = dfs["lead_logs"].copy()
    urs = dfs["user_referral_statuses"].copy()

    if not urs.empty and set(["id", "description"]).issubset(urs.columns):
        urs_ren = urs.rename(columns={"id": "user_referral_status_id", "description": "referral_status"})
        ur = ur.merge(urs_ren, on="user_referral_status_id", how="left")

    if not rr.empty and "id" in rr.columns:
        rr_ren = rr.rename(columns={"id": "referral_reward_id"})
        ur = ur.merge(rr_ren, on="referral_reward_id", how="left")

    if not ufl.empty and "user_referral_id" in ufl.columns:
        ufl_sorted = ufl.sort_values("created_at").drop_duplicates("user_referral_id", keep="last")
        ur = ur.merge(ufl_sorted, left_on="referral_id", right_on="user_referral_id", how="left")

    if not pt.empty and "transaction_id" in pt.columns:
        ur = ur.merge(pt, on="transaction_id", how="left", suffixes=("", "_transaction"))

    if not ul.empty and "user_id" in ul.columns:
        ur = ur.merge(ul.add_prefix("referrer_"), left_on="referrer_id", right_on="referrer_user_id", how="left")
        ur = ur.merge(ul.add_prefix("referee_"), left_on="referee_id", right_on="referee_user_id", how="left")

    if not ll.empty and "lead_id" in ll.columns and "referee_id" in ur.columns:
        ur = ur.merge(ll.add_prefix("lead_"), left_on="referee_id", right_on="lead_lead_id", how="left")

    # Build final output frame
    final = pd.DataFrame()
    final["referral_details_id"] = ur.index + 1
    final["referral_id"] = ur.get("referral_id").astype(str)
    final["referral_source"] = ur.get("referral_source")

    def compute_source_category(row):
        rs = row.get("referral_source")
        if rs == "User Sign Up":
            return "Online"
        if rs == "Draft Transaction":
            return "Offline"
        if rs == "Lead":
            for k in ["lead_source_category", "source_category"]:
                if k in ur.columns and pd.notna(ur.loc[row.name, k]):
                    return ur.loc[row.name, k]
            return None
        return None

    final["referral_source_category"] = ur.apply(compute_source_category, axis=1)
    final["referral_at"] = ur.get("referral_at")
    final["referrer_id"] = ur.get("referrer_id")
    final["referrer_name"] = ur.get("referrer_name")
    final["referrer_phone_number"] = ur.get("referrer_phone")
    final["referrer_homeclub"] = ur.get("referrer_homeclub") if "referrer_homeclub" in ur.columns else ur.get("homeclub")
    final["referee_id"] = ur.get("referee_id")
    final["referee_name"] = ur.get("referee_name")
    final["referee_phone"] = ur.get("referee_phone")
    final["referral_status"] = ur.get("referral_status")
    final["num_reward_days"] = ur.get("num_reward_days") if "num_reward_days" in ur.columns else None
    final["transaction_id"] = ur.get("transaction_id")
    final["transaction_status"] = ur.get("transaction_status")
    final["transaction_at"] = ur.get("transaction_at")
    final["transaction_location"] = ur.get("transaction_location")
    final["transaction_type"] = ur.get("transaction_type")
    final["updated_at"] = ur.get("updated_at")
    # reward_granted detection
    for c in ["reward_granted_at", "is_reward_granted", "reward_granted"]:
        if c in ur.columns:
            final["reward_granted_at"] = ur.get(c)
            break
    if "reward_granted_at" not in final.columns:
        final["reward_granted_at"] = pd.NaT

    final["reward_value_raw"] = ur.get("reward_value")
    final["reward_value"] = final["reward_value_raw"].apply(parse_reward_value)

    # Timezone conversion: default to Asia/Jakarta where timezone is not present
    default_tz = "Asia/Jakarta"

    # --- Convert datetime columns to tz-naive BEFORE timezone conversion (robust) ---
    for dt_col in ["referral_at", "transaction_at", "updated_at", "reward_granted_at"]:
        if dt_col in final.columns:
            # coerce to datetime (naive or tz-aware)
            final[dt_col] = pd.to_datetime(final[dt_col], errors="coerce")
            # If the Series has a timezone (tz-aware), convert to tz-naive
        try:
            # Series.dt.tz is None for tz-naive Series, or a tz object if tz-aware
            if getattr(final[dt_col].dt, "tz", None) is not None:
                final[dt_col] = final[dt_col].dt.tz_convert(None)
        except Exception:
            # Fallback per-element (safe): if an element has tzinfo, drop it
            def _drop_tz(ts):
                if pd.isna(ts):
                    return pd.NaT
                try:
                    if getattr(ts, "tzinfo", None) is not None:
                        return ts.tz_convert(None)
                except Exception:
                    try:
                        return pd.to_datetime(str(ts))
                    except Exception:
                        return ts
                return ts
            final[dt_col] = final[dt_col].apply(_drop_tz)


   



    for idx in final.index:
        tz = default_tz
        if "referrer_timezone_homeclub" in ur.columns and pd.notna(ur.loc[idx, "referrer_timezone_homeclub"]):
            tz = ur.loc[idx, "referrer_timezone_homeclub"]
        elif "lead_timezone_location" in ur.columns and pd.notna(ur.loc[idx, "lead_timezone_location"]):
            tz = ur.loc[idx, "lead_timezone_location"]
        if pd.notna(final.at[idx, "referral_at"]):
            final.at[idx, "referral_at"] = convert_utc_to_local(final.at[idx, "referral_at"], tz)
        tz_tx = ur.loc[idx, "timezone_transaction"] if "timezone_transaction" in ur.columns and pd.notna(ur.loc[idx, "timezone_transaction"]) else default_tz
        if pd.notna(final.at[idx, "transaction_at"]):
            final.at[idx, "transaction_at"] = convert_utc_to_local(final.at[idx, "transaction_at"], tz_tx)
        if pd.notna(final.at[idx, "updated_at"]):
            final.at[idx, "updated_at"] = convert_utc_to_local(final.at[idx, "updated_at"], default_tz)
        if pd.notna(final.at[idx, "reward_granted_at"]):
            final.at[idx, "reward_granted_at"] = convert_utc_to_local(final.at[idx, "reward_granted_at"], default_tz)

    # Apply business logic and reason
    is_valid_list = []
    reason_list = []
    for idx in final.index:
        merged_row = ur.loc[idx] if idx in ur.index else pd.Series()
        valid, reason = evaluate_business_logic_and_reason(final.loc[idx], merged_row)
        is_valid_list.append(valid)
        reason_list.append(reason)
    final["is_business_logic_valid"] = is_valid_list
    final["business_logic_reason"] = reason_list

    # Fill defaults for missing values to avoid nulls in final CSV
    final = final.fillna({
        "referral_source_category": "Unknown",
        "referrer_name": "",
        "referrer_phone_number": "",
        "referrer_homeclub": "",
        "referee_name": "",
        "referee_phone": "",
        "referral_status": "Unknown",
        "num_reward_days": 0,
        "transaction_status": "",
        "transaction_location": "",
        "transaction_type": "",
        "reward_value": 0
    })
    # Make pandas infer better dtypes to avoid future downcasting warnings
    final = final.infer_objects()

    # The assignment expects 46 rows. Select rows by priority:
    # 1) valid condition 1 rows (is_business_logic_valid==True, reason valid_cond_1)
    # 2) valid condition 2 rows (pending/failed with no reward)
    # 3) fill remainder by most recent referral_at
    def pick_latest_by_referral_id(df):
        if "transaction_at" in df.columns:
            return df.sort_values(["transaction_at", "referral_at"], ascending=False).drop_duplicates("referral_id", keep="first")
        else:
            return df.drop_duplicates("referral_id", keep="first")

    valid_cond1 = final[(final["is_business_logic_valid"] == True) & (final["business_logic_reason"] == "valid_cond_1_all_checks_passed")]
    valid_cond1 = pick_latest_by_referral_id(valid_cond1)
    selected = valid_cond1.copy()

    # If fewer than 46, add cond2
    if len(selected) < 46:
        need = 46 - len(selected)
        cond2 = final[final["business_logic_reason"] == "valid_cond_2_pending_or_failed_no_reward"]
        cond2 = pick_latest_by_referral_id(cond2)
        cond2 = cond2[~cond2["referral_id"].isin(selected["referral_id"])]
        selected = pd.concat([selected, cond2.head(need)], ignore_index=True)

    # If still fewer, fill with other rows by latest referral_at
    if len(selected) < 46:
        need = 46 - len(selected)
        remaining = final[~final["referral_id"].isin(selected["referral_id"])]
        remaining = pick_latest_by_referral_id(remaining)
        remaining = remaining.sort_values("referral_at", ascending=False)
        selected = pd.concat([selected, remaining.head(need)], ignore_index=True)

    # Final shape should be 46 rows
    selected = selected.reset_index(drop=True)

    # Columns order per PDF (only include those that exist)
    columns_order = [
        'referral_details_id','referral_id','referral_source','referral_source_category','referral_at',
        'referrer_id','referrer_name','referrer_phone_number','referrer_homeclub','referee_id','referee_name','referee_phone',
        'referral_status','num_reward_days','transaction_id','transaction_status','transaction_at','transaction_location','transaction_type',
        'updated_at','reward_granted_at','is_business_logic_valid','business_logic_reason','reward_value'
    ]
    cols_present = [c for c in columns_order if c in selected.columns]
    final_to_write = selected[cols_present]

    # Save final CSV
    final_csv_path = output_path / "final_output.csv"
    final_to_write.to_csv(final_csv_path, index=False)
    print(f"Wrote final_output.csv with {len(final_to_write)} rows to {final_csv_path}")

    # Data dictionary
    data_dict = [
        {'column':'referral_details_id','description':'Internal sequential id for the report','data_type':'INTEGER'},
        {'column':'referral_id','description':'Referral unique id from user_referrals','data_type':'STRING'},
        {'column':'referral_source','description':'Source of referral (User Sign Up/Draft Transaction/Lead)','data_type':'STRING'},
        {'column':'referral_source_category','description':'Mapped category (Online/Offline/Lead source)','data_type':'STRING'},
        {'column':'referral_at','description':'Referral creation timestamp (localized)','data_type':'DATETIME'},
        {'column':'referrer_id','description':'User id of referrer','data_type':'STRING'},
        {'column':'referrer_name','description':'Referrer name','data_type':'STRING'},
        {'column':'referrer_phone_number','description':'Referrer phone','data_type':'STRING'},
        {'column':'referrer_homeclub','description':'Referrer home club','data_type':'STRING'},
        {'column':'referee_id','description':'Referee id (if available)','data_type':'STRING'},
        {'column':'referee_name','description':'Referee name','data_type':'STRING'},
        {'column':'referee_phone','description':'Referee phone','data_type':'STRING'},
        {'column':'referral_status','description':'Referral status description','data_type':'STRING'},
        {'column':'num_reward_days','description':'Number of days for reward validity','data_type':'INTEGER'},
        {'column':'transaction_id','description':'Transaction id linked to referral','data_type':'STRING'},
        {'column':'transaction_status','description':'Transaction status (PAID etc)','data_type':'STRING'},
        {'column':'transaction_at','description':'Transaction timestamp (localized)','data_type':'DATETIME'},
        {'column':'transaction_location','description':'Transaction location','data_type':'STRING'},
        {'column':'transaction_type','description':'Transaction type (NEW etc)','data_type':'STRING'},
        {'column':'updated_at','description':'Last updated timestamp for referral','data_type':'DATETIME'},
        {'column':'reward_granted_at','description':'Timestamp when reward was granted','data_type':'DATETIME'},
        {'column':'is_business_logic_valid','description':'Boolean flag whether referral passes business logic checks','data_type':'BOOLEAN'},
        {'column':'business_logic_reason','description':'Reason code describing which rule made it valid/invalid','data_type':'STRING'},
        {'column':'reward_value','description':'Numeric reward value parsed from referral_rewards','data_type':'NUMERIC'}
    ]
    pd.DataFrame(data_dict).to_excel(output_path / "data_dictionary.xlsx", index=False)
    print("Wrote data_dictionary.xlsx")

    # Write README, requirements, Dockerfile to output for convenience
    (output_path / "README.md").write_text(
        "Outputs: final_output.csv, profiling_report.csv, data_dictionary.xlsx\n"
        "Run `python main.py --input <folder> --output <folder>` to reproduce.",
        encoding="utf-8"
    )
    (output_path / "requirements.txt").write_text("pandas\npytz\nopenpyxl\n", encoding="utf-8")
    dockerfile_text = (
        "FROM python:3.10-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt ./\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY . /app\n"
        "CMD [\"python\", \"main.py\"]\n"
    )
    (output_path / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
    print("Wrote README.md, requirements.txt, Dockerfile to output folder")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Referral Program - Pandas pipeline")
    parser.add_argument("--input", type=str, default=".", help="Input folder with CSV files")
    parser.add_argument("--output", type=str, default="./output", help="Output folder")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
