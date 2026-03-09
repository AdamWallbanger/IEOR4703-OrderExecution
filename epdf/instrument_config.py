"""
Instrument Configuration Module

Manages instrument-specific parameters including tick sizes and symbol parsing.
"""

import re
from typing import Optional


class InstrumentConfig:
    """
    Configuration manager for futures instruments.

    Handles tick size mappings and automatic symbol detection from filenames.
    """

    # Tick size mapping table (from official exchange specifications)
    # TICK_SIZES = {
    #     # CME Equity Index Futures
    #     'ES': 0.25,      # S&P 500 E-mini
    #                      # Source: https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.html
    #
    #     'NQ': 0.25,      # Nasdaq 100 E-mini
    #                      # Source: https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html
    #
    #     # NYMEX Energy Futures
    #     'CL': 0.01,      # Crude Oil WTI (dollars per barrel)
    #                      # Source: https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html
    #
    #     'HO': 0.0001,    # Heating Oil (NY Harbor ULSD, dollars per gallon)
    #                      # Source: https://www.cmegroup.com/markets/energy/refined-products/heating-oil.contractSpecs.html
    #
    #     # COMEX Metals Futures
    #     'GC': 0.10,      # Gold (dollars per troy ounce)
    #                      # Source: https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html
    #
    #     'SI': 0.005,     # Silver (dollars per troy ounce)
    #                      # Source: https://www.cmegroup.com/markets/metals/precious/silver.contractSpecs.html
    #
    #     'HG': 0.0005,    # Copper (dollars per pound)
    #                      # Source: https://www.cmegroup.com/markets/metals/base/copper.contractSpecs.html
    #
    #     # CBOT Interest Rate Futures
    #     'TY': 0.015625,  # 10-Year T-Note (1/2 of 1/32 point, or 0.5/32)
    #                      # Source: https://www.cmegroup.com/markets/interest-rates/us-treasury/10-year-us-treasury-note.contractSpecs.html
    #
    #     # Eurex Fixed Income Futures
    #     'RX': 0.01,      # Euro Bund (German 10-year government bond, percent of par value)
    #                     # Source: https://www.eurex.com/ex-en/markets/int/fix/government-bonds/Euro-Bund-Futures-137298
    #
    #     # CBOT Agricultural Futures
    #     'S': 0.25,       # Soybeans (ZS) (1/4 cent per bushel)
    #     'ZS': 0.25,      # Soybeans (alternative code)
    #                      # Source: https://www.cmegroup.com/markets/agriculture/oilseeds/soybean.contractSpecs.html
    #
    #     # CME FX Futures
    #     'EC': 0.0001,    # Euro FX (6E) (dollars per euro)
    #     '6E': 0.0001,    # Euro FX (alternative code)
    #                      # Source: https://www.cmegroup.com/markets/fx/g10/euro-fx.contractSpecs.html
    #
    #     'BP': 0.0001,    # British Pound (6B) (dollars per GBP)
    #                      # Source: https://www.cmegroup.com/markets/fx/g10/british-pound.contractSpecs.html
    #
    #     'JY': 0.0000005, # Japanese Yen (6J) (dollars per yen)
    #                      # Source: https://www.cmegroup.com/markets/fx/g10/japanese-yen.contractSpecs.html
    #
    #     # Eurex Equity Index Futures
    #     'VG': 0.50,      # EuroStoxx 50 (FESX) (index points, tick size valid for 2021-06-21 to 2022-03-20)
    #                      # Note: Changed from 1.0 to 0.5 on 2021-06-21, then back to 1.0 on 2022-03-21
    #                      # Source: https://www.eurex.com/ex-en/markets/idx/stx/euro-stoxx-50-derivatives/products/EURO-STOXX-50-Index-Futures-160088
    #                      # Circular: https://www.eurex.com/ex-en/find/circulars/circular-2407638
    # }
    TICK_SIZES = {
        # CME Equity Index Futures
        'ES': 0.25,  # S&P 500 E-mini
        # Source: https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.html

        'NQ': 0.25,  # Nasdaq 100 E-mini
        # Source: https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html

        # NYMEX Energy Futures
        'CL': 0.01,  # Crude Oil WTI (dollars per barrel)
        # Source: https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html

        'HO': 0.01,  # Heating Oil (NY Harbor ULSD, dollars per gallon)
        # Source: https://www.cmegroup.com/markets/energy/refined-products/heating-oil.contractSpecs.html

        # COMEX Metals Futures
        'GC': 0.10,  # Gold (dollars per troy ounce)
        # Source: https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html

        'SI': 0.005,  # Silver (dollars per troy ounce)
        # Source: https://www.cmegroup.com/markets/metals/precious/silver.contractSpecs.html

        'HG': 0.0005,  # Copper (dollars per pound)
        # Source: https://www.cmegroup.com/markets/metals/base/copper.contractSpecs.html

        # CBOT Interest Rate Futures
        'TY': 0.015625,  # 10-Year T-Note (1/2 of 1/32 point, or 0.5/32)
        # Source: https://www.cmegroup.com/markets/interest-rates/us-treasury/10-year-us-treasury-note.contractSpecs.html

        # Eurex Fixed Income Futures
        'RX': 0.01,  # Euro Bund (German 10-year government bond, percent of par value)
        # Source: https://www.eurex.com/ex-en/markets/int/fix/government-bonds/Euro-Bund-Futures-137298

        # CBOT Agricultural Futures
        'S': 0.25,  # Soybeans (ZS) (1/4 cent per bushel)
        'ZS': 0.25,  # Soybeans (alternative code)
        # Source: https://www.cmegroup.com/markets/agriculture/oilseeds/soybean.contractSpecs.html

        # CME FX Futures
        'EC': 0.0001,  # Euro FX (6E) (dollars per euro)
        '6E': 0.0001,  # Euro FX (alternative code)
        # Source: https://www.cmegroup.com/markets/fx/g10/euro-fx.contractSpecs.html

        'BP': 0.01,  # British Pound (6B) (dollars per GBP)
        # Source: https://www.cmegroup.com/markets/fx/g10/british-pound.contractSpecs.html

        'JY': 0.005,  # Japanese Yen (6J) (dollars per yen)
        # Source: https://www.cmegroup.com/markets/fx/g10/japanese-yen.contractSpecs.html

        # Eurex Equity Index Futures
        'VG': 0.50,  # EuroStoxx 50 (FESX) (index points, tick size valid for 2021-06-21 to 2022-03-20)
        # Note: Changed from 1.0 to 0.5 on 2021-06-21, then back to 1.0 on 2022-03-21
        # Source: https://www.eurex.com/ex-en/markets/idx/stx/euro-stoxx-50-derivatives/products/EURO-STOXX-50-Index-Futures-160088
        # Circular: https://www.eurex.com/ex-en/find/circulars/circular-2407638
    }

    # Month code mapping
    MONTH_CODES = {
        'F': 1,  # January
        'G': 2,  # February
        'H': 3,  # March
        'J': 4,  # April
        'K': 5,  # May
        'M': 6,  # June
        'N': 7,  # July
        'Q': 8,  # August
        'U': 9,  # September
        'V': 10, # October
        'X': 11, # November
        'Z': 12  # December
    }

    @classmethod
    def get_tick_size(cls, symbol: str) -> float:
        """
        Get tick size for a given instrument symbol.

        Args:
            symbol: Instrument symbol (e.g., 'ES', 'VG', 'NQ')

        Returns:
            Tick size (epsilon) for the instrument

        Raises:
            ValueError: If symbol is not recognized
        """
        symbol = symbol.upper()
        if symbol not in cls.TICK_SIZES:
            raise ValueError(
                f"Unknown instrument symbol: {symbol}. "
                f"Supported symbols: {', '.join(sorted(cls.TICK_SIZES.keys()))}"
            )
        return cls.TICK_SIZES[symbol]

    @classmethod
    def parse_symbol_from_filename(cls, filepath: str) -> str:
        """
        Extract instrument symbol from futures contract filename.

        Futures filenames typically follow the format: [Symbol][Month][Year]
        Examples:
            - VGH22.csv → VG (EuroStoxx 50, March 2022)
            - ESM20.csv → ES (S&P 500 Mini, June 2020)
            - NQU20.csv → NQ (Nasdaq 100, September 2020)

        Args:
            filepath: Path to the data file

        Returns:
            Instrument symbol (e.g., 'VG', 'ES', 'NQ')

        Raises:
            ValueError: If symbol cannot be parsed from filename
        """
        # Extract filename from path
        import os
        filename = os.path.basename(filepath)

        # Pattern: 1-2 uppercase letters + month code + 2-digit year
        # Month codes: F, G, H, J, K, M, N, Q, U, V, X, Z
        pattern = r'([A-Z]{1,2})[FGHJKMNQUVXZ]\d{2}'
        match = re.search(pattern, filename)

        if match:
            symbol = match.group(1)
            # Verify it's a known symbol
            if symbol in cls.TICK_SIZES:
                return symbol
            else:
                raise ValueError(
                    f"Parsed symbol '{symbol}' from filename '{filename}' "
                    f"is not in the known instruments list. "
                    f"Supported symbols: {', '.join(sorted(cls.TICK_SIZES.keys()))}"
                )

        raise ValueError(
            f"Could not parse instrument symbol from filename: {filename}. "
            f"Expected format: [Symbol][MonthCode][Year] (e.g., VGH22.csv, ESM20.csv)"
        )

    @classmethod
    def register_custom_instrument(cls, symbol: str, tick_size: float):
        """
        Register a custom instrument with its tick size.

        Args:
            symbol: Instrument symbol
            tick_size: Tick size (epsilon) for the instrument
        """
        symbol = symbol.upper()
        cls.TICK_SIZES[symbol] = tick_size

    @classmethod
    def parse_contract_details(cls, filepath: str) -> dict:
        """
        Parse full contract details from filename.

        Args:
            filepath: Path to the data file

        Returns:
            Dictionary with keys: symbol, month_code, month, year, tick_size
        """
        import os
        filename = os.path.basename(filepath)

        pattern = r'([A-Z]{1,2})([FGHJKMNQUVXZ])(\d{2})'
        match = re.search(pattern, filename)

        if not match:
            raise ValueError(f"Could not parse contract details from: {filename}")

        symbol = match.group(1)
        month_code = match.group(2)
        year = match.group(3)

        return {
            'symbol': symbol,
            'month_code': month_code,
            'month': cls.MONTH_CODES.get(month_code),
            'year': int(year),
            'tick_size': cls.get_tick_size(symbol),
            'contract_code': f"{symbol}{month_code}{year}"
        }
