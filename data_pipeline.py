import pandas as pd
import os

RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"

COUNTRY_NAME_MAP = {
    "US": "United States",
    "Korea, South": "South Korea",
    "Russian Federation": "Russia",
    "Taiwan*": "Taiwan",
    "Czechia": "Czech Republic",
    "Iran (Islamic Republic of)": "Iran",
    "Viet Nam": "Vietnam",
    "Hong Kong": "Hong Kong SAR",
    "Macao": "Macao SAR",
    "Bahamas": "The Bahamas",
    "Gambia": "The Gambia",
    "Congo (Brazzaville)": "Congo",
    "Congo (Kinshasa)": "Democratic Republic of the Congo",
    "Burma": "Myanmar",
    "Eswatini": "Swaziland",
    "Cabo Verde": "Cape Verde",
    "Saint Kitts and Nevis": "St Kitts and Nevis",
    "Saint Vincent and the Grenadines": "St Vincent and the Grenadines",
}


def load_jhu_timeseries(file_name: str, value_name: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, file_name)
    df = pd.read_csv(path)
    df = df.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="date",
        value_name=value_name,
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["country"] = df["Country/Region"].replace(COUNTRY_NAME_MAP)
    df = df[["country", "date", value_name]]
    return df


def standardize_country_name(country: str) -> str:
    if pd.isna(country):
        return country
    s = country.strip()
    if s in COUNTRY_NAME_MAP:
        return COUNTRY_NAME_MAP[s]
    return s


def load_owid_data() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "owid-covid-data.csv")
    owid = pd.read_csv(path, parse_dates=["date"])
    owid["country"] = owid["location"].apply(standardize_country_name)
    return owid


def load_mobility_data() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "Global_Mobility_Report.csv")
    if not os.path.exists(path):
        print("WARNING: Mobility file missing. Mobility values will be NaN.")
        return pd.DataFrame(columns=["country", "date", "mobility_index"])
    mob = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    if "country_region" in mob.columns:
        mob["country"] = mob["country_region"].apply(standardize_country_name)
    elif "Country/Region" in mob.columns:
        mob["country"] = mob["Country/Region"].apply(standardize_country_name)
    else:
        mob["country"] = mob["location"].apply(standardize_country_name)
    # Use average of workplace, residential, transit stations as a mobility index fallback
    for col in [
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
        "retail_and_recreation_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
    ]:
        if col not in mob.columns:
            mob[col] = pd.NA
    mob["mobility_index"] = mob[
        [
            "workplaces_percent_change_from_baseline",
            "retail_and_recreation_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "residential_percent_change_from_baseline",
        ]
    ].mean(axis=1, skipna=True)
    mob = mob[["country", "date", "mobility_index"]]
    return mob


def build_final_dataset() -> pd.DataFrame:
    confirmed = load_jhu_timeseries(
        "time_series_covid19_confirmed_global.csv", "confirmed_cases"
    )
    deaths = load_jhu_timeseries("time_series_covid19_deaths_global.csv", "deaths")

    merged = pd.merge(
        confirmed,
        deaths,
        on=["country", "date"],
        how="outer",
    )

    merged = (
        merged.groupby(["country", "date"], as_index=False)[
            ["confirmed_cases", "deaths"]
        ]
        .sum(min_count=1)
        .sort_values(["country", "date"])
    )
    merged["confirmed_cases"] = merged["confirmed_cases"].fillna(0).astype(int)
    merged["deaths"] = merged["deaths"].fillna(0).astype(int)

    owid = load_owid_data()
    merge_cols = ["country", "date"]
    merged2 = pd.merge(
        merged,
        owid[
            [
                "country",
                "date",
                "total_vaccinations_per_hundred",
                "people_vaccinated_per_hundred",
                "people_fully_vaccinated_per_hundred",
            ]
        ],
        on=merge_cols,
        how="left",
    )

    merged2["vaccination_rate"] = merged2["people_fully_vaccinated_per_hundred"].fillna(
        0
    )

    mobility = load_mobility_data()
    merged2 = pd.merge(
        merged2,
        mobility,
        on=["country", "date"],
        how="left",
    )

    merged2 = merged2.rename(
        columns={
            "confirmed_cases": "cases",
            "mobility_index": "mobility",
        }
    )

    merged2 = merged2[
        ["date", "country", "cases", "deaths", "vaccination_rate", "mobility"]
    ]
    merged2 = merged2.sort_values(["country", "date"]).reset_index(drop=True)
    return merged2


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    feature = df.copy()
    feature = feature.sort_values(["country", "date"]).reset_index(drop=True)
    feature["daily_cases"] = (
        feature.groupby("country")["cases"].diff().fillna(feature["cases"])
    )
    feature["daily_cases"] = feature["daily_cases"].clip(lower=0)
    feature["previous_daily_cases"] = feature.groupby("country")["daily_cases"].shift(1)
    feature["case_growth_rate"] = feature["daily_cases"] / feature[
        "previous_daily_cases"
    ].replace(0, pd.NA)
    feature["case_growth_rate"] = feature["case_growth_rate"].replace(
        [pd.NA, float("inf"), -float("inf")], pd.NA
    )
    feature["7_day_moving_average"] = feature.groupby("country")[
        "daily_cases"
    ].transform(lambda x: x.rolling(7, min_periods=1).mean())
    feature["vaccination_ratio"] = feature["vaccination_rate"] / 100
    feature["mobility_change"] = feature.groupby("country")["mobility"].transform(
        lambda x: x - x.ffill().bfill()
    )
    feature = feature.drop(columns=["previous_daily_cases"])
    return feature


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    final = build_final_dataset()
    final_path = os.path.join(PROCESSED_DIR, "final_covid_dataset.csv")
    final.to_csv(final_path, index=False)
    print(f"Written final dataset to: {final_path}")

    features = feature_engineer(final)
    features_path = os.path.join(PROCESSED_DIR, "feature_engineered_dataset.csv")
    features.to_csv(features_path, index=False)
    print(f"Written feature engineered dataset to: {features_path}")

    print("Dataset sizes:")
    print("final", final.shape)
    print("features", features.shape)


if __name__ == "__main__":
    main()
