# SPDX-FileCopyrightText: 2024-present Lukas Strobel <mail@lstrobel.com>
#
# SPDX-License-Identifier: MIT

import re
import unicodedata
from collections.abc import Iterator

import marisa_trie  # type: ignore
from nltk.tokenize import RegexpTokenizer  # type: ignore
from nltk.tokenize.api import TokenizerI  # type: ignore


class PinyinTokenizer(TokenizerI):
    """
    Splits Hanyu Pinyin words on syllable boundaries.
    Preserves punctuation and original character spans, ignoring whitespace.
    Supports precomposed and decomposed Unicode tone marks.

    Args:
        include_nonstandard: If True, includes rare/non-standard syllables in the valid
            set. Defaults to False.

    Example:
        >>> tokenizer = PinyinTokenizer()
        >>> tokenizer.tokenize("nǐhǎo")
        ['nǐ', 'hǎo']
    """

    # fmt: off
    STANDARD_SYLLABLES = [
        'a', 'o', 'e', 'ê', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'er',
        'yi', 'ya', 'yo', 'ye', 'yao', 'you', 'yan', 'yin', 'yang', 'ying',
        'wu', 'wa', 'wo', 'wai', 'wei', 'wan', 'wen', 'wang', 'weng',
        'yu', 'yue', 'yuan', 'yun', 'yong',
        
        'ba', 'bai', 'bei', 'bao', 'ban', 'ben', 'bang', 'beng',
        'bi', 'bie', 'biao', 'bian', 'bin', 'bing',
        'bu', 'bo',
        
        'pa', 'pai', 'pei', 'pao', 'pou', 'pan', 'pen', 'pang', 'peng',
        'pi', 'pie', 'piao', 'pian', 'pin', 'ping',
        'pu', 'po',
        
        'ma', 'me', 'mai', 'mei', 'mao', 'mou', 'man', 'men', 'mang', 'meng',
        'mi', 'mie', 'miao', 'miu', 'mian', 'min', 'ming',
        'mu', 'mo',
        
        'fa', 'fei', 'fou', 'fan', 'fen', 'fang', 'feng',
        'fu', 'fo',
        
        'da', 'de', 'dai', 'dei', 'dao', 'dou', 'dan', 'den', 'dang', 'deng',
        'di', 'die', 'diao', 'diu', 'dian', 'din', 'ding',
        'du', 'duo', 'dui', 'duan', 'dun', 'dong',
        
        'ta', 'te', 'tai', 'tao', 'tou', 'tan', 'tang', 'teng',
        'ti', 'tie', 'tiao', 'tian', 'ting',
        'tu', 'tuo', 'tui', 'tuan', 'tun', 'tong',
        
        'na', 'ne', 'nai', 'nei', 'nao', 'nou', 'nan', 'nen', 'nang', 'neng',
        'ni', 'nie', 'niao', 'niu', 'nian', 'nin', 'niang', 'ning',
        'nu', 'nuo', 'nuan', 'nun', 'nong',
        'nü', 'nüe', 'nv', 'nve',
        
        'la', 'lo', 'le', 'lai', 'lei', 'lao', 'lou', 'lan', 'lang', 'leng',
        'li', 'lie', 'liao', 'liu', 'lian', 'lin', 'liang', 'ling',
        'lu', 'luo', 'luan', 'lun', 'long',
        'lü', 'lüe', 'lv', 'lve',
        
        'ga', 'ge', 'gai', 'gei', 'gao', 'gou', 'gan', 'gen', 'gang', 'geng',
        'gu', 'gua', 'guo', 'guai', 'gui', 'guan', 'gun', 'guang', 'gong',
        
        'ka', 'ke', 'kai', 'kao', 'kou', 'kan', 'ken', 'kang', 'keng',
        'ku', 'kua', 'kuo', 'kuai', 'kui', 'kuan', 'kun', 'kuang', 'kong',
        
        'ha', 'he', 'hai', 'hei', 'hao', 'hou', 'han', 'hen', 'hang', 'heng',
        'hu', 'hua', 'huo', 'huai', 'hui', 'huan', 'hun', 'huang', 'hong',
        
        'ji', 'jia', 'jie', 'jiao', 'jiu', 'jian', 'jin', 'jiang', 'jing',
        'ju', 'jue', 'juan', 'jun', 'jiong',
        
        'qi', 'qia', 'qie', 'qiao', 'qiu', 'qian', 'qin', 'qiang', 'qing',
        'qu', 'que', 'quan', 'qun', 'qiong',

        'xi', 'xia', 'xie', 'xiao', 'xiu', 'xian', 'xin', 'xiang', 'xing',
        'xu', 'xue', 'xuan', 'xun', 'xiong',
        
        'zhi', 'zha', 'zhe', 'zhai', 'zhao', 'zhou', 'zhan', 'zhen', 'zhang', 'zheng',
        'zhu', 'zhua', 'zhuo', 'zhuai', 'zhui', 'zhuan', 'zhun', 'zhuang', 'zhong',

        'chi', 'cha', 'che', 'chai', 'chao', 'chou', 'chan', 'chen', 'chang', 'cheng',
        'chu', 'chua', 'chuo', 'chuai', 'chui', 'chuan', 'chun', 'chuang', 'chong',

        'shi', 'sha', 'she', 'shai', 'shei', 'shao', 'shou', 'shan', 'shen', 'shang', 'sheng',  # noqa: E501
        'shu', 'shua', 'shuo', 'shuai', 'shui', 'shuan', 'shun', 'shuang',

        'r', # Include to cover erhua
        'ri', 're', 'rao', 'rou', 'ran', 'ren', 'rang', 'reng',
        'ru', 'ruo', 'rui', 'ruan', 'run', 'rong',

        'zi', 'za', 'ze', 'zai', 'zei', 'zao', 'zou', 'zan', 'zen', 'zang', 'zeng',
        'zu', 'zuo', 'zui', 'zuan', 'zun', 'zong',

        'ci', 'ca', 'ce', 'cai', 'cao', 'cou', 'can', 'cen', 'cang', 'ceng',
        'cu', 'cuo', 'cui', 'cuan', 'cun', 'cong',

        'si', 'sa', 'se', 'sai', 'sao', 'sou', 'san', 'sen', 'sang', 'seng',
        'su', 'suo', 'sui', 'suan', 'sun', 'song',
    ]

    NON_STANDARD_SYLLABLES = [
        'yai', 'ong', 
        'biang', 
        'pia', 'pun',
        'fai', 'fiao',
        'dia', 'diang', 'duang',
        'tei', 
        'nia', 'nui',
        'len', 'lia',
        'lüan', 'lün', 'lvan', 'lvn',
        'gin', 'ging', 
        'kei', 'kiu', 'kiang',
        'zhei',
        'rua',
        'cei',
        'sei',
    ]
    # fmt: on

    VOWEL_TONE_VARIANTS = {
        "a": ("ā", "á", "ǎ", "à"),
        "e": ("ē", "é", "ě", "è"),
        "ê": ("ê\u0304", "ế", "ê\u030c", "ề"),
        "i": ("ī", "í", "ǐ", "ì"),
        "o": ("ō", "ó", "ǒ", "ò"),
        "u": ("ū", "ú", "ǔ", "ù"),
        "ü": ("ǖ", "ǘ", "ǚ", "ǜ"),
        "v": ("v\u0304", "v\u0301", "v\u030c", "v\u0300"),  # ü alternative
    }

    # https://lingua.mtsu.edu/chinese-computing/phonology/syllable.php
    SYLLABLE_FREQUENCIES = {
        "a": "143836",
        "ai": "213586",
        "an": "418511",
        "ang": "10267",
        "ao": "60455",
        "ba": "679337",
        "bai": "349026",
        "ban": "429225",
        "bang": "87257",
        "bao": "620534",
        "bei": "670092",
        "ben": "411819",
        "beng": "12432",
        "bi": "421278",
        "bian": "597897",
        "biao": "320022",
        "bie": "188222",
        "bin": "25234",
        "bing": "533163",
        "bo": "226232",
        "bu": "2943968",
        "ca": "15954",
        "cai": "427296",
        "can": "141816",
        "cang": "28725",
        "cao": "84874",
        "ce": "128686",
        "cen": "1069",
        "ceng": "143033",
        "cha": "255265",
        "chai": "7276",
        "chan": "291680",
        "chang": "849564",
        "chao": "163164",
        "che": "166297",
        "chen": "288341",
        "cheng": "857501",
        "chi": "312261",
        "chong": "146742",
        "chou": "85191",
        "chu": "1221539",
        "chuai": "1829",
        "chuan": "280973",
        "chuang": "125052",
        "chui": "32523",
        "chun": "79748",
        "chuo": "4140",
        "ci": "657819",
        "cong": "405173",
        "cou": "5544",
        "cu": "46527",
        "cuan": "5002",
        "cui": "39761",
        "cun": "173519",
        "cuo": "94761",
        "da": "1469164",
        "dai": "497044",
        "dan": "700482",
        "dang": "564806",
        "dao": "1788900",
        "de": "8722289",
        "deng": "356833",
        "di": "1724350",
        "dian": "575699",
        "diao": "167083",
        "die": "37329",
        "ding": "521121",
        "diu": "11285",
        "dong": "723244",
        "dou": "541323",
        "du": "248897",
        "duan": "247728",
        "dui": "870671",
        "dun": "87900",
        "duo": "747728",
        "e": "109896",
        "en": "37898",
        "eng": "6707",
        "er": "1353892",
        "fa": "1088739",
        "fan": "459450",
        "fang": "874075",
        "fei": "427791",
        "fen": "566699",
        "feng": "289836",
        "fo": "47038",
        "fou": "64772",
        "fu": "1056555",
        "ga": "5050",
        "gai": "335105",
        "gan": "426276",
        "gang": "140899",
        "gao": "475204",
        "ge": "1748292",
        "gei": "217815",
        "gen": "196736",
        "geng": "202642",
        "gong": "1072260",
        "gou": "229650",
        "gu": "432908",
        "gua": "43244",
        "guai": "55710",
        "guan": "788407",
        "guang": "238613",
        "gui": "280303",
        "gun": "18003",
        "guo": "1841775",
        "ha": "52475",
        "hai": "779180",
        "han": "203571",
        "hang": "51372",
        "hao": "573206",
        "he": "1635213",
        "hei": "84324",
        "hen": "321875",
        "heng": "58378",
        "hong": "149967",
        "hou": "749405",
        "hu": "471675",
        "hua": "807991",
        "huai": "97699",
        "huan": "232948",
        "huang": "184840",
        "hui": "1181130",
        "hun": "96486",
        "huo": "712024",
        "ji": "2977951",
        "jia": "1051033",
        "jian": "1451889",
        "jiang": "511050",
        "jiao": "703885",
        "jie": "1109315",
        "jin": "1163074",
        "jing": "1186243",
        "jiong": "4019",
        "jiu": "1166024",
        "ju": "685312",
        "juan": "51759",
        "jue": "384873",
        "jun": "463498",
        "ka": "60440",
        "kai": "398586",
        "kan": "469240",
        "kang": "91959",
        "kao": "130219",
        "ke": "1254178",
        "ken": "47955",
        "keng": "6954",
        "kong": "260769",
        "kou": "208960",
        "ku": "132880",
        "kua": "19956",
        "kuai": "163050",
        "kuan": "67005",
        "kuang": "153891",
        "kui": "35822",
        "kun": "53949",
        "kuo": "81713",
        "la": "183642",
        "lai": "1118779",
        "lan": "142369",
        "lang": "73480",
        "lao": "299072",
        "le": "2226721",
        "lei": "238732",
        "leng": "62152",
        "li": "2308041",
        "lian": "389564",
        "liang": "639396",
        "liao": "130518",
        "lie": "165597",
        "lin": "192140",
        "ling": "457865",
        "liu": "351731",
        "long": "94753",
        "lou": "52042",
        "lu": "362789",
        "luan": "55675",
        "lun": "264565",
        "luo": "281150",
        "lü": "276875",
        "lüe": "61411",
        "ma": "388106",
        "mai": "137363",
        "man": "188310",
        "mang": "65817",
        "mao": "170048",
        "me": "480720",
        "mei": "933373",
        "men": "1391330",
        "meng": "132845",
        "mi": "230299",
        "mian": "503878",
        "miao": "68770",
        "mie": "37465",
        "min": "354208",
        "ming": "696618",
        "miu": "4290",
        "mo": "273029",
        "mou": "116225",
        "mu": "408250",
        "na": "831925",
        "nai": "69877",
        "nan": "356362",
        "nang": "6709",
        "nao": "93398",
        "ne": "112185",
        "nei": "231331",
        "nen": "3707",
        "neng": "665358",
        "ni": "812061",
        "nian": "695312",
        "niang": "40907",
        "niao": "29319",
        "nie": "12991",
        "nin": "51094",
        "ning": "44017",
        "niu": "52960",
        "nong": "131225",
        "nu": "67922",
        "nuan": "9762",
        "nuo": "40022",
        "nü": "185188",
        "nüe": "2979",
        "o": "4582",
        "ou": "64452",
        "pa": "96735",
        "pai": "177618",
        "pan": "108530",
        "pang": "57763",
        "pao": "99236",
        "pei": "114110",
        "pen": "16807",
        "peng": "102631",
        "pi": "376008",
        "pian": "158598",
        "piao": "63377",
        "pie": "5059",
        "pin": "184837",
        "ping": "281773",
        "po": "176842",
        "pou": "3767",
        "pu": "138110",
        "qi": "1765121",
        "qia": "21271",
        "qian": "638821",
        "qiang": "232291",
        "qiao": "105464",
        "qie": "287521",
        "qin": "199822",
        "qing": "774492",
        "qiong": "23167",
        "qiu": "244599",
        "qu": "884073",
        "quan": "503025",
        "que": "321359",
        "qun": "74521",
        "ran": "549902",
        "rang": "136101",
        "rao": "33503",
        "re": "72069",
        "ren": "2318926",
        "reng": "75535",
        "ri": "363763",
        "rong": "164398",
        "rou": "45636",
        "ru": "669787",
        "ruan": "30350",
        "rui": "28898",
        "run": "16922",
        "ruo": "89865",
        "sa": "48768",
        "sai": "55584",
        "san": "349119",
        "sang": "32783",
        "sao": "27351",
        "se": "155077",
        "sen": "29533",
        "seng": "6920",
        "sha": "147176",
        "shai": "4290",
        "shan": "284678",
        "shang": "1292610",
        "shao": "256246",
        "she": "433892",
        "shei": "61528",
        "shen": "1006355",
        "sheng": "1150523",
        "shi": "7050291",
        "shou": "803601",
        "shu": "867809",
        "shua": "13882",
        "shuai": "26859",
        "shuan": "3605",
        "shuang": "76927",
        "shui": "277968",
        "shun": "42354",
        "shuo": "884288",
        "si": "1055744",
        "song": "152392",
        "sou": "29413",
        "su": "318074",
        "suan": "121239",
        "sui": "243720",
        "sun": "69029",
        "suo": "604505",
        "ta": "2376365",
        "tai": "412339",
        "tan": "224714",
        "tang": "122309",
        "tao": "147601",
        "te": "239091",
        "teng": "27136",
        "ti": "724525",
        "tian": "489761",
        "tiao": "242251",
        "tie": "62877",
        "ting": "314552",
        "tong": "909188",
        "tou": "400461",
        "tu": "341681",
        "tuan": "104151",
        "tui": "154449",
        "tun": "14204",
        "tuo": "121320",
        "wa": "48764",
        "wai": "300896",
        "wan": "452252",
        "wang": "545917",
        "wei": "2092014",
        "wen": "695970",
        "weng": "8702",
        "wo": "1749745",
        "wu": "1372087",
        "xi": "1082565",
        "xia": "734342",
        "xian": "1167460",
        "xiang": "1575940",
        "xiao": "868443",
        "xie": "686105",
        "xin": "918778",
        "xing": "1359245",
        "xiong": "104841",
        "xiu": "129816",
        "xu": "570666",
        "xuan": "195599",
        "xue": "572292",
        "xun": "169768",
        "ya": "289491",
        "yan": "791181",
        "yang": "698164",
        "yao": "988311",
        "ye": "1240777",
        "yi": "5869055",
        "yin": "750738",
        "ying": "698618",
        "yo": "3635",
        "yong": "652185",
        "you": "2746758",
        "yu": "1713834",
        "yuan": "1024450",
        "yue": "491401",
        "yun": "219022",
        "za": "44146",
        "zai": "2252027",
        "zan": "60431",
        "zang": "55933",
        "zao": "272008",
        "ze": "293345",
        "zei": "7045",
        "zen": "112294",
        "zeng": "103042",
        "zha": "68234",
        "zhai": "50407",
        "zhan": "584585",
        "zhang": "341340",
        "zhao": "289937",
        "zhe": "2602857",
        "zhen": "418098",
        "zheng": "983292",
        "zhi": "3185308",
        "zhong": "2141356",
        "zhou": "239028",
        "zhu": "958453",
        "zhua": "36094",
        "zhuai": "1396",
        "zhuan": "215756",
        "zhuang": "220567",
        "zhui": "59572",
        "zhun": "113877",
        "zhuo": "51036",
        "zi": "1589345",
        "zong": "271651",
        "zou": "208069",
        "zu": "344981",
        "zuan": "10064",
        "zui": "392734",
        "zun": "36597",
        "zuo": "893830",
    }

    NON_ALPHABETIC_REGEX = re.compile(r"[^\w\s\u0300-\u036f]+")

    def _get_tone_variants(self, syllable: str):
        """
        Generate all valid tone variants for a syllable. Assumes syllabe is lowercase
        """
        variants = [syllable]  # Include toneless variation

        # Find the vowels in the syllable (both upper and lower case)
        vowels = [c for c in syllable if c in self.VOWEL_TONE_VARIANTS]
        if not vowels:
            return variants

        # Determine which vowel gets the tone mark
        # https://en.wikipedia.org/wiki/Pinyin#Placement_and_omission
        tone_vowel = None
        if any(v == "a" for v in vowels):
            tone_vowel = next(v for v in vowels if v == "a")
        elif any(v == "e" for v in vowels):
            tone_vowel = next(v for v in vowels if v == "e")
        elif any(v == "o" for v in vowels):
            tone_vowel = next(v for v in vowels if v == "o")
        else:
            # If there is no a e or o, the vowels are either 'iu', 'ui', or 'ê',
            # in which case the mark goes on the last vowel
            tone_vowel = vowels[-1]

        # Generate variants with each tone mark
        for toned_vowel in self.VOWEL_TONE_VARIANTS[tone_vowel]:
            variant = syllable.replace(tone_vowel, toned_vowel)
            variants.append(variant)

        return variants

    def _remove_tone(self, syllable: str):
        """Remove tone marks from a pinyin syllable, returning the base form."""
        result = unicodedata.normalize("NFC", syllable)
        for base_vowel, toned_vowels in self.VOWEL_TONE_VARIANTS.items():
            for toned in toned_vowels:
                result = result.replace(toned, base_vowel)
        return result

    def __init__(self, include_nonstandard=False):
        self.preprocess_tokenizer = RegexpTokenizer(
            r"[\w\u0300-\u036f]+|" + self.NON_ALPHABETIC_REGEX.pattern
        )

        trie_contents = []

        # Add standard syllables
        for syllable in self.STANDARD_SYLLABLES:
            for variant in self._get_tone_variants(syllable):
                trie_contents.append(unicodedata.normalize("NFD", variant))

        if include_nonstandard:
            for syllable in self.NON_STANDARD_SYLLABLES:
                for variant in self._get_tone_variants(syllable):
                    trie_contents.append(unicodedata.normalize("NFD", variant))

        self.trie = marisa_trie.Trie(trie_contents)
        self._max_syllable_length = max(len(variant) for variant in trie_contents)

    def _get_best_split(self, text: str) -> list[tuple[int, int]] | None:
        """Find the best local spans by syllable count, vowel starts, and frequency."""
        normalized_parts = []
        original_offsets = [0]
        for end, character in enumerate(text, 1):
            normalized = unicodedata.normalize("NFD", character.lower())
            normalized_parts.append(normalized)
            original_offsets.extend([end] * len(normalized))
        text = "".join(normalized_parts)

        scores: list[tuple[int, int, int] | None] = [None] * (len(text) + 1)
        next_positions = [0] * len(text)
        scores[-1] = (0, 0, 0)

        for start in range(len(text) - 1, -1, -1):
            remaining = text[start : start + self._max_syllable_length]
            best_score = None
            for match in reversed(self.trie.prefixes(remaining)):
                end = start + len(match)
                suffix_score = scores[end]
                if suffix_score is None:
                    continue

                syllable = self._remove_tone(match)
                score = (
                    suffix_score[0] + 1,
                    suffix_score[1] + int(syllable[0] in self.VOWEL_TONE_VARIANTS),
                    suffix_score[2] - int(self.SYLLABLE_FREQUENCIES.get(syllable, "0")),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    next_positions[start] = end

            scores[start] = best_score

        if scores[0] is None:
            return None

        spans = []
        start = 0
        while start < len(text):
            end = next_positions[start]
            spans.append((original_offsets[start], original_offsets[end]))
            start = end
        return spans

    def span_tokenize(self, s: str) -> Iterator[tuple[int, int]]:
        """Yield offsets into the original text, preserving Unicode composition."""
        starting_spans = self.preprocess_tokenizer.span_tokenize(s)

        final_spans = []
        for start, end in starting_spans:
            original = s[start:end]
            if self.NON_ALPHABETIC_REGEX.fullmatch(original):
                final_spans.append((start, end))
                continue

            chosen = self._get_best_split(original)
            if chosen is None:
                raise ValueError(f"Invalid pinyin at substring: {original}")

            for local_start, local_end in chosen:
                final_spans.append((start + local_start, start + local_end))

        yield from final_spans

    def tokenize(self, s: str) -> list[str]:
        return [s[start:end] for start, end in self.span_tokenize(s)]
