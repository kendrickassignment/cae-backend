"""Versioned country/area catalog; a reporting scope, not a legal classification."""

CATALOG_VERSION = "asia-2026-10"
CATALOG_SOURCE = "https://unstats.un.org/unsd/methodology/m49/"
_REGIONS = {
    "Southeast Asia": "BN:Brunei|KH:Cambodia|ID:Indonesia|LA:Laos|MY:Malaysia|MM:Myanmar|PH:Philippines|SG:Singapore|TH:Thailand|TL:Timor-Leste|VN:Vietnam",
    "East Asia": "CN:China|HK:Hong Kong|MO:Macao|JP:Japan|MN:Mongolia|KP:North Korea|KR:South Korea|TW:Taiwan",
    "South Asia": "AF:Afghanistan|BD:Bangladesh|BT:Bhutan|IN:India|IR:Iran|MV:Maldives|NP:Nepal|PK:Pakistan|LK:Sri Lanka",
    "Central Asia": "KZ:Kazakhstan|KG:Kyrgyzstan|TJ:Tajikistan|TM:Turkmenistan|UZ:Uzbekistan",
    "West Asia": "AM:Armenia|AZ:Azerbaijan|BH:Bahrain|CY:Cyprus|GE:Georgia|IQ:Iraq|IL:Israel|JO:Jordan|KW:Kuwait|LB:Lebanon|OM:Oman|PS:Palestine|QA:Qatar|SA:Saudi Arabia|SY:Syria|TR:Türkiye|AE:United Arab Emirates|YE:Yemen",
}
COUNTRIES = {}
for region, entries in _REGIONS.items():
    for entry in entries.split("|"):
        code, name = entry.split(":")
        COUNTRIES[code] = {"code": code, "name": name, "subregion": region}


def normalize_countries(values: list[str]) -> list[str]:
    if not values:
        raise ValueError("target_countries must contain at least one country code")
    result = []
    for value in values:
        code = value.strip().upper()
        if code not in COUNTRIES:
            raise ValueError(f"Unsupported country code: {code}. See GET /countries")
        if code not in result:
            result.append(code)
    return result


def catalog() -> dict:
    return {
        "version": CATALOG_VERSION,
        "source": CATALOG_SOURCE,
        "notes": "UN M49-inspired statistical regions, plus TW as a separate reporting market. No legal or sovereignty determination. Russia is outside this catalog; no inference about operations there.",
        "countries": list(COUNTRIES.values()),
    }
