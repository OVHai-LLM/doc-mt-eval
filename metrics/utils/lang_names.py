import re
import pycountry

MAPPING= {
    "bn-hi": ("Bengali", "Hindi"),
    "cs-de_DE": ("Czech", "German"),
    "cs-en": ("Czech", "English"),
    "cs-uk": ("Czech", "Ukrainian"),
    "cs-uk_UA": ("Czech", "Ukrainian"),
    "de-cs": ("German", "Czech"),
    "de-en": ("German", "English"),
    "de-fr": ("German", "French"),
    "en-ar_EG": ("English", "Arabic (Egyptian)"),
    "en-bho_IN": ("English", "Bhojpuri"),
    "en-cs": ("English", "Czech"),
    "en-cs_CZ": ("English", "Czech"),
    "en-de": ("English", "German"),
    "en-es": ("English", "Spanish"),
    "en-et_EE": ("English", "Estonian"),
    "en-fi": ("English", "Finnish"),
    "en-gu": ("English", "Gujarati"),
    "en-ha": ("English", "Hausa"),
    "en-hi": ("English", "Hindi"),
    "en-hr": ("English", "Croatian"),
    "en-is": ("English", "Icelandic"),
    "en-is_IS": ("English", "Icelandic"),
    "en-it_IT": ("English", "Italian"),
    "en-iu": ("English", "Inuktitut"),
    "en-ja": ("English", "Japanese"),
    "en-ja_JP": ("English", "Japanese"),
    "en-kk": ("English", "Kazakh"),
    "en-ko_KR": ("English", "Korean"),
    "en-lt": ("English", "Lithuanian"),
    "en-mas_KE": ("English", "Masai"),
    "en-pl": ("English", "Polish"),
    "en-ru": ("English", "Russian"),
    "en-ru_RU": ("English", "Russian"),
    "en-sr_Cyrl_RS": ("English", "Serbian (Cyrillic)"),
    "en-ta": ("English", "Tamil"),
    "en-uk": ("English", "Ukrainian"),
    "en-uk_UA": ("English", "Ukrainian"),
    "en-zh": ("English", "Chinese (Simplified)"),
    "en-zh_CN": ("English", "Chinese (Simplified)"),
    "fi-en": ("Finnish", "English"),
    "fr-de": ("French", "German"),
    "gu-en": ("Gujarati", "English"),
    "ha-en": ("Hausa", "English"),
    "he-en": ("Hebrew", "English"),
    "hi-bn": ("Hindi", "Bengali"),
    "is-en": ("Icelandic", "English"),
    "iu-en": ("Inuktitut", "English"),
    "ja-en": ("Japanese", "English"),
    "ja-zh": ("Japanese", "Chinese (Simplified)"),
    "ja-zh_CN": ("Japanese", "Chinese (Simplified)"),
    "kk-en": ("Kazakh", "English"),
    "km-en": ("Khmer", "English"),
    "lt-en": ("Lithuanian", "English"),
    "pl-en": ("Polish", "English"),
    "ps-en": ("Pushto", "English"),
    "ru-en": ("Russian", "English"),
    "sah-ru": ("Yakut", "Russian"),
    "ta-en": ("Tamil", "English"),
    "uk-cs": ("Ukrainian", "Czech"),
    "uk-en": ("Ukrainian", "English"),
    "xh-zu": ("Xhosa", "Zulu"),
    "zh-en": ("Chinese (Simplified)", "English"),
    "zu-xh": ("Zulu", "Xhosa"),
}
LP_PATTERN = re.compile(
    '|'.join(re.escape(k) for k in sorted(MAPPING, key=len, reverse=True))
)

def _pycountry_name(code):
    lang = pycountry.languages.get(alpha_2=code) or pycountry.languages.get(alpha_3=code)
    return lang.name if lang else None


def get_full_names(lp_string):
    match = LP_PATTERN.search(lp_string)
    if match:
        return MAPPING[match.group(0)]

    parts = lp_string.split("-", 1)
    if len(parts) == 2:
        src_name = _pycountry_name(parts[0])
        tgt_name = _pycountry_name(parts[1])
        if src_name and tgt_name:
            return (src_name, tgt_name)

    print(f'[Warning] No mapping found in "{lp_string}", falling back to raw codes.')
    return (parts[0], parts[1]) if len(parts) == 2 else (lp_string, lp_string)
