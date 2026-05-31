import os
import numpy as np
import pandas as pd
import streamlit as st

from epdf import ePDFCalculator


# ============================================================
# Streamlit page config
# ============================================================

st.set_page_config(
    page_title="Optimal Price Generator",
    layout="wide"
)

st.title("Optimal Price Generator")


# ============================================================
# Fixed config
# ============================================================

data_path = "Data/"

symbol_dict = {
    "VG": "EuroStoxx",
    "BP": "GBP - British Pound",
    "RX": "German Bunds - German Government Bonds",
    "GC": "Gold",
    "HO": "HeatingOil",
    "JY": "JPY - Japanese Yen",
    "NQ": "Nasdaq"
}

tick_dict = {
            'NQ': 0.25,
            'HO': 0.01,
            'GC': 0.10,
            'BP': 0.01,
            'JY': 0.005,
            'RX': 0.01,
            'VG': 0.50
        }

st.info(f"Fixed data path: `{data_path}`")


# ============================================================
# Sidebar inputs
# ============================================================

st.sidebar.header("Trading Inputs")

instrument = st.sidebar.selectbox(
    "Future contract",
    list(symbol_dict.keys()),
    format_func=lambda x: f"{x} - {symbol_dict[x]}"
)

trade_side = st.sidebar.selectbox(
    "Trade side",
    ["Buy", "Sell"]
)

# Use text input instead of date_input to avoid calendar popup cutoff issue
trade_date_str = st.sidebar.text_input(
    "Trade date",
    value="2026/05/30",
    help="Format: YYYY/MM/DD"
)

trade_time_str = st.sidebar.text_input(
    "Trade time, minute level",
    value="21:04",
    help="Format: HH:MM"
)

try:
    trade_timestamp = pd.to_datetime(
        trade_date_str + " " + trade_time_str,
        format="%Y/%m/%d %H:%M"
    ).floor("min")
except Exception:
    st.sidebar.error("Invalid date or time format. Please use YYYY/MM/DD and HH:MM.")
    st.stop()

tau = st.sidebar.number_input(
    "tau (minutes)",
    min_value=1,
    value=5,
    step=1,
    help="Unit: minutes"
)

risk_percentage = st.sidebar.number_input(
    "risk_percentage, 0 < x < 1",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="Must be strictly between 0 and 1"
)


# ============================================================
# Advanced parameters
# ============================================================

with st.sidebar.expander("Advanced Parameters"):
    M = st.number_input(
        "M",
        min_value=1,
        value=3,
        step=1
    )

    N = st.number_input(
        "N",
        min_value=1,
        value=3,
        step=1
    )

    K = st.number_input(
        "K",
        min_value=1,
        value=2,
        step=1
    )

    ewma_halflife = st.number_input(
        "ewma_halflife",
        min_value=1,
        value=10,
        step=1
    )

    estimation_method = st.selectbox(
        "estimation_method",
        ["smoothed", "empirical"],
        index=0
    )

    smoothing_alpha = st.number_input(
        "smoothing_alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )


# ============================================================
# Main request summary
# ============================================================

st.header("Selected Trading Request")

col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

with col1:
    st.metric("Instrument", instrument)

with col2:
    st.metric("Side", trade_side)

with col3:
    st.write("Trade time")
    st.markdown(f"### `{trade_timestamp.strftime('%Y-%m-%d %H:%M')}`")

with col4:
    st.metric("tau", f"{tau} min")

st.write(
    {
        "data_path": data_path,
        "instrument_folder": symbol_dict[instrument],
        "risk_percentage": f"{risk_percentage:.2f}  (0 < x < 1)",
        "M": M,
        "N": N,
        "K": K,
        "ewma_halflife": ewma_halflife,
        "estimation_method": estimation_method,
        "smoothing_alpha": smoothing_alpha
    }
)


# ============================================================
# Generate optimal price
# ============================================================

if st.button("Generate Optimal Price"):

    # ========================================================
    # Step 1: Locate data folder
    # ========================================================

    folder_path = os.path.join(data_path, symbol_dict[instrument])

    st.subheader("Step 1: Locate Data Folder")
    st.write(f"Looking for data in: `{folder_path}`")

    if not os.path.isdir(folder_path):
        st.error(f"Data folder not found: {folder_path}")
        st.stop()

    files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]

    if len(files) == 0:
        st.error(f"No CSV files found in folder: {folder_path}")
        st.stop()

    st.success(f"Found {len(files)} contract files.")


    # ========================================================
    # Step 2: Find main contract at selected time
    # ========================================================

    st.subheader("Step 2: Find Main Contract at Selected Time")

    candidate_records = []

    for file in files:
        file_path = os.path.join(folder_path, file)

        try:
            temp = pd.read_csv(file_path)

            if "time" not in temp.columns:
                continue

            required_cols = ["time", "close", "volume"]

            if not all(col in temp.columns for col in required_cols):
                continue

            temp["time"] = pd.to_datetime(temp["time"])
            temp = temp.sort_values("time").reset_index(drop=True)

            # Find the latest row at or before the selected minute
            hist_for_current_minute = temp[temp["time"] <= trade_timestamp]

            if hist_for_current_minute.empty:
                continue

            current_row = hist_for_current_minute.iloc[-1]

            # Only accept data exactly in the selected minute
            if current_row["time"].floor("min") != trade_timestamp:
                continue

            candidate_records.append(
                {
                    "contract": file,
                    "path": file_path,
                    "last_time": current_row["time"],
                    "close": current_row["close"],
                    "volume": current_row["volume"]
                }
            )

        except Exception:
            continue

    if len(candidate_records) == 0:
        st.error(f"No data found for {instrument} at {trade_timestamp}.")
        st.stop()

    candidate_df = pd.DataFrame(candidate_records)

    st.write("Contracts with data at selected minute:")
    st.dataframe(candidate_df)

    # Default main contract logic:
    # among all contracts with data at this minute, choose the one with largest volume
    main_contract_row = (
        candidate_df
        .sort_values("volume", ascending=False)
        .iloc[0]
    )

    main_contract = main_contract_row["contract"]
    main_contract_path = main_contract_row["path"]

    st.success(f"Main contract at this time: `{main_contract}`")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Main contract", main_contract)

    with col2:
        st.metric("Last bar time", str(main_contract_row["last_time"]))

    with col3:
        st.metric("Volume", main_contract_row["volume"])


    # ========================================================
    # Internal calculation, hidden from UI
    # ========================================================

    try:
        df = pd.read_csv(main_contract_path)

        required_cols = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:
            st.error(f"Main contract data is missing columns: {missing_cols}")
            st.stop()

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        calc = ePDFCalculator(
            instrument=instrument,
            tau=tau,
            M=M,
            N=N,
            K=K,
            ewma_halflife=ewma_halflife,
            estimation_method=estimation_method,
            smoothing_alpha=smoothing_alpha
        )

        calc.fit(
            main_contract_path,
            train_end_date=trade_timestamp
        )

        tick = tick_dict[instrument]

        # volume strict lag EWMA
        df["volume_ewma"] = (
            df["volume"]
            .shift(1)
            .ewm(halflife=ewma_halflife, adjust=False)
            .mean()
        )

        # range R = (H - L) / tick, strict lag EWMA as volatility
        df["range_R"] = (df["high"] - df["low"]) / tick

        df["volatility_ewma"] = (
            df["range_R"]
            .shift(1)
            .ewm(halflife=ewma_halflife, adjust=False)
            .mean()
        )

        # open first difference, strict lag EWMA as price change
        df["open_delta"] = df["open"].diff()

        df["ewma_delta_x"] = (
            df["open_delta"]
            .shift(1)
            .ewm(halflife=ewma_halflife, adjust=False)
            .mean()
        )

        # Use strictly previous row before trade_timestamp
        hist = df[df["time"] < trade_timestamp]

        if hist.empty:
            st.error("No historical row before selected trade time.")
            st.stop()

        pre_row = hist.iloc[-1]

        if (
            pd.isna(pre_row["volume_ewma"])
            or pd.isna(pre_row["volatility_ewma"])
            or pd.isna(pre_row["ewma_delta_x"])
        ):
            st.error("State feature contains NaN. Cannot calculate optimal price.")
            st.stop()

        state = calc.get_current_state(
            pre_row["volume_ewma"],
            pre_row["volatility_ewma"],
            pre_row["ewma_delta_x"]
        )

        if trade_side == "Buy":
            direction = "range_dn"
        else:
            direction = "range_up"

    except Exception as e:
        st.error(f"Internal calculation failed: {e}")
        st.stop()


    # ========================================================
    # Step 3: Query CDF
    # ========================================================

    st.subheader("Step 3: Query CDF")

    placement = (0, np.nan)
    cdf_records = []

    try:
        for level in range(10):
            cdf = calc.query_cdf(
                level,
                direction,
                state
            )

            passed = cdf >= risk_percentage

            cdf_records.append(
                {
                    "level": level,
                    "cdf": cdf,
                    "risk_percentage": risk_percentage,
                    "passed": passed
                }
            )

            if passed:
                placement = (level, cdf)
            else:
                break

    except Exception as e:
        st.error(f"CDF query failed: {e}")
        st.stop()

    cdf_df = pd.DataFrame(cdf_records)

    st.dataframe(cdf_df)

    placement_level = placement[0]
    placement_cdf = placement[1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Selected level", placement_level)

    with col2:
        st.metric("Selected CDF", placement_cdf)


    # ========================================================
    # Step 4: Generate optimal price
    # ========================================================

    st.subheader("Step 4: Generate Optimal Price")

    if direction == "range_dn":
        opti_price = pre_row["close"] - placement_level * tick
    else:
        opti_price = pre_row["close"] + placement_level * tick

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Previous close", pre_row["close"])

    with col2:
        st.metric("Tick size", tick)

    with col3:
        st.metric("Placement level", placement_level)

    with col4:
        st.metric("Optimal price", opti_price)

    st.success(f"Optimal price: {opti_price}")


    # ========================================================
    # Step 5: Final result
    # ========================================================

    st.subheader("Step 5: Final Result")

    result = {
        "instrument": instrument,
        "instrument_name": symbol_dict[instrument],
        "trade_side": trade_side,
        "trade_timestamp": trade_timestamp,
        "main_contract": main_contract,
        "tau_min": tau,
        "risk_percentage": risk_percentage,
        "direction": direction,
        "placement_level": placement_level,
        "placement_cdf": placement_cdf,
        "previous_close": pre_row["close"],
        "previous_row_time": pre_row["time"],
        "tick": tick,
        "opti_price": opti_price
    }

    result_df = pd.DataFrame([result])

    st.dataframe(result_df)

    st.download_button(
        label="Download Result CSV",
        data=result_df.to_csv(index=False),
        file_name="optimal_price_result.csv",
        mime="text/csv"
    )