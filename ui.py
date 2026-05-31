import os
import numpy as np
import pandas as pd
import streamlit as st

from epdf import ePDFCalculator


st.set_page_config(
    page_title="Optimal Price Generator",
    layout="wide"
)

st.title("Optimal Price Generator")


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
    "VG": 1,
    "BP": 0.0001,
    "RX": 0.01,
    "GC": 0.1,
    "HO": 0.0001,
    "JY": 0.000001,
    "NQ": 0.25
}

st.info(f"Fixed data path: `{data_path}`")


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

trade_date_str = st.sidebar.text_input(
    "Trade date",
    value="2025/04/08",
    help="Format: YYYY/MM/DD"
)

trade_time_str = st.sidebar.text_input(
    "Trade time, minute level",
    value="09:00",
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


if st.button("Generate Optimal Price"):

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


    st.subheader("Step 2: Find Main Contract at Selected Time")

    candidate_records = []

    for file in files:
        file_path = os.path.join(folder_path, file)

        try:
            temp = pd.read_csv(file_path)

            required_cols = ["time", "close", "volume"]

            if not all(col in temp.columns for col in required_cols):
                continue

            temp["time"] = pd.to_datetime(temp["time"])
            temp = temp.sort_values("time").reset_index(drop=True)

            hist_for_current_minute = temp[temp["time"] <= trade_timestamp]

            if hist_for_current_minute.empty:
                continue

            current_row = hist_for_current_minute.iloc[-1]

            if current_row["time"].floor("min") != trade_timestamp:
                continue

            candidate_records.append(
                {
                    "contract": file,
                    "path": file_path,
                    "last_time": current_row["time"].strftime("%Y-%m-%d %H:%M"),
                    "close": current_row["close"],
                    "volume": current_row["volume"]
                }
            )

        except Exception:
            continue

    if len(candidate_records) == 0:
        st.error(f"No data found for {instrument} at {trade_timestamp.strftime('%Y-%m-%d %H:%M')}.")
        st.stop()

    candidate_df = pd.DataFrame(candidate_records)

    st.write("Contracts with data at selected minute:")
    st.dataframe(candidate_df)

    main_contract_row = (
        candidate_df
        .sort_values("volume", ascending=False)
        .iloc[0]
    )

    main_contract = main_contract_row["contract"]
    main_contract_path = main_contract_row["path"]

    st.success(f"Main contract at this time: `{main_contract}`")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.metric("Main contract", main_contract)

    with col2:
        st.write("Last bar time")
        st.markdown(f"### `{main_contract_row['last_time']}`")

    with col3:
        st.metric("Volume", main_contract_row["volume"])


    try:
        tick = tick_dict[instrument]

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

        df_proc = calc.data_processor.process_pipeline(
            filepath=main_contract_path,
            tick_size=tick,
            tau=tau,
            min_completeness=0.9,
            train_end_date=None
        )

        df_proc = calc.state_classifier.compute_all_ewma_features(
            df_proc,
            ewma_halflife
        )

        if not isinstance(df_proc.index, pd.DatetimeIndex):
            if "time" in df_proc.columns:
                df_proc["time"] = pd.to_datetime(df_proc["time"])
                df_proc = df_proc.set_index("time")
            elif "timestamp" in df_proc.columns:
                df_proc["timestamp"] = pd.to_datetime(df_proc["timestamp"])
                df_proc = df_proc.set_index("timestamp")
            else:
                st.error("Processed dataframe has no datetime index, time column, or timestamp column.")
                st.stop()

        df_proc = df_proc.sort_index()

        hist = df_proc[df_proc.index < trade_timestamp]

        if hist.empty:
            st.error("No tau-min bar before selected trade time.")
            st.stop()

        pre_row = hist.iloc[-1]

        state_cols = ["v_ewma", "sigma_ewma", "delta_x_ewma"]

        missing_state_cols = [
            col for col in state_cols
            if col not in df_proc.columns
        ]

        if missing_state_cols:
            st.error(f"Processed dataframe is missing state columns: {missing_state_cols}")
            st.stop()

        if pre_row[state_cols].isna().any():
            st.error("State feature contains NaN. Cannot calculate optimal price.")
            st.stop()

        state = calc.get_current_state(
            pre_row["v_ewma"],
            pre_row["sigma_ewma"],
            pre_row["delta_x_ewma"]
        )

        if trade_side == "Buy":
            direction = "range_dn"
        else:
            direction = "range_up"

    except Exception as e:
        st.error(f"Internal calculation failed: {e}")
        st.stop()


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


    st.subheader("Step 4: Generate Optimal Price")

    if direction == "range_dn":
        opti_price = pre_row["close"] - placement_level * tick
    else:
        opti_price = pre_row["close"] + placement_level * tick

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Previous tau-bar close", pre_row["close"])

    with col2:
        st.metric("Tick size", tick)

    with col3:
        st.metric("Placement level", placement_level)

    with col4:
        st.metric("Optimal price", opti_price)

    st.success(f"Optimal price: {opti_price}")


    st.subheader("Step 5: Backtest Fill Check")

    try:
        raw_df = pd.read_csv(main_contract_path)
        raw_df["time"] = pd.to_datetime(raw_df["time"])
        raw_df = raw_df.sort_values("time").reset_index(drop=True)

        placement_end = trade_timestamp + pd.Timedelta(minutes=tau)

        placement_df = raw_df[
            (raw_df["time"] >= trade_timestamp) &
            (raw_df["time"] < placement_end)
        ]

        next_row = raw_df[raw_df["time"] >= placement_end].head(1)

        if len(placement_df) == 0:
            if len(next_row) == 0:
                filled = False
                filled_price = np.nan
                fill_reason = "No data during placement window and no next row after placement window."
            else:
                filled = False
                filled_price = next_row["open"].iloc[0]
                fill_reason = "No data during placement window. Filled price uses next row open."
        else:
            mask = (
                (placement_df["low"] <= opti_price) &
                (placement_df["high"] >= opti_price)
            )

            filled = mask.any()

            if filled:
                filled_price = opti_price
                first_fill_row = placement_df[mask].iloc[0]
                fill_reason = (
                    "Optimal price covered by OHLC range at "
                    f"{pd.Timestamp(first_fill_row['time']).strftime('%Y-%m-%d %H:%M')}."
                )
            else:
                filled_price = placement_df["close"].iloc[-1]
                fill_reason = "Optimal price not covered. Filled price uses last close in placement window."

        fill_status = "Filled" if filled else "Not Filled"

        if filled:
            st.success(f"Fill status: {fill_status}")
        else:
            st.warning(f"Fill status: {fill_status}")

        col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])

        with col1:
            st.write("Placement start")
            st.markdown(f"### `{trade_timestamp.strftime('%Y-%m-%d %H:%M')}`")

        with col2:
            st.write("Placement end")
            st.markdown(f"### `{placement_end.strftime('%Y-%m-%d %H:%M')}`")

        with col3:
            st.metric("Filled price", filled_price)

        with col4:
            st.metric("Rows checked", len(placement_df))

        st.write(fill_reason)

        if len(placement_df) > 0:
            display_placement_df = placement_df[
                ["time", "open", "high", "low", "close", "volume"]
            ].copy()

            display_placement_df["time"] = display_placement_df["time"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )

            st.write("Placement window data:")
            st.dataframe(display_placement_df)

    except Exception as e:
        st.error(f"Backtest fill check failed: {e}")
        st.stop()


    st.subheader("Step 6: Final Result")

    result = {
        "instrument": instrument,
        "instrument_name": symbol_dict[instrument],
        "trade_side": trade_side,
        "trade_timestamp": trade_timestamp.strftime("%Y-%m-%d %H:%M"),
        "main_contract": main_contract,
        "tau_min": tau,
        "risk_percentage": risk_percentage,
        "direction": direction,
        "state": state,
        "placement_level": placement_level,
        "placement_cdf": placement_cdf,
        "previous_tau_bar_time": pd.Timestamp(pre_row.name).strftime("%Y-%m-%d %H:%M"),
        "previous_tau_bar_close": pre_row["close"],
        "tick": tick,
        "opti_price": opti_price,
        "placement_start": trade_timestamp.strftime("%Y-%m-%d %H:%M"),
        "placement_end": placement_end.strftime("%Y-%m-%d %H:%M"),
        "fill_status": fill_status,
        "filled": filled,
        "filled_price": filled_price,
        "fill_reason": fill_reason
    }

    result_df = pd.DataFrame([result])

    st.dataframe(result_df)

    st.download_button(
        label="Download Result CSV",
        data=result_df.to_csv(index=False),
        file_name="optimal_price_result.csv",
        mime="text/csv"
    )