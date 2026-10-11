from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------

# Study window: 2015 through the most recent data each source publishes.
START_YEAR = 2015

# Base year for inflation-adjusted ("real") dollar figures.
REAL_DOLLAR_BASE_YEAR = 2024

SOCRATA_BASE = "https://data.cdc.gov/resource"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
WONDER_URL = "https://wonder.cdc.gov/controller/datarequest/{db}"
CMS_NHE_URLS = {
    "summary": "https://www.cms.gov/files/zip/nhe-summary-including-share-gdp-cy-1960-2024.zip",
    "detail": "https://www.cms.gov/files/zip/national-health-expenditures-type-service-source-funds-cy-1960-2024.zip",
}

# CDC data.cdc.gov dataset ids
CDC_DATASETS = {
    "flu_coverage": "vh55-3he6",  # FluVaxView: influenza coverage, all ages, by race/ethnicity
    "kindergarten_vax": "ijqb-a7ye",  # SchoolVaxView: kindergarten MMR coverage & exemptions
    "child_vax": "fhky-rtsk",  # ChildVaxView (NIS-Child): MMR by race/ethnicity & poverty
    "brfss": "dttw-5yxu",  # BRFSS prevalence: flu shot (65+) by income & education
    "leading_causes_state": "bi63-dtpu",  # NCHS leading causes of death by state (to 2017)
    "vsrr_mortality": "489q-934x",  # NCHS VSRR quarterly provisional mortality by state
}

STATES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}
STATE_ABBR = {name: abbr for abbr, name in STATES.items()}

# Harmonized cause-of-death names used across every mortality source.
MAJOR_CAUSES = [
    "All causes",
    "Heart disease",
    "Cancer",
    "Unintentional injuries",
    "Stroke",
    "Chronic lower respiratory disease",
    "Alzheimer disease",
    "Diabetes",
    "Kidney disease",
    "Chronic liver disease",
    "Influenza & pneumonia",
    "Suicide",
    "COVID-19",
]

# Display order for harmonized race/ethnicity groups.
RACE_GROUPS = [
    "Hispanic",
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Non-Hispanic Asian",
    "Non-Hispanic American Indian/Alaska Native",
    "Non-Hispanic Native Hawaiian/Pacific Islander",
    "Non-Hispanic Multiracial/Other",
]
