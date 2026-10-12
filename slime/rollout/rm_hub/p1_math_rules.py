"""Rule-only P1 scoring helpers exported from the local slime_B implementation.

Function bodies and constants are preserved from p1.py; optional network and
Math Verify paths are excluded because training uses use_xverify=False.
Numeric rounding/tolerance is intentionally retained for reward compatibility.
Source SHA256: 584ef4fb493b31afdf19fac9ba715bd2a9e670728f7ae1f7745b643bfd990fe8
"""

import re
import math
from typing import Optional
from pylatexenc import latex2text
from decimal import Decimal, localcontext
import sympy
from sympy import N, Pow, Mul
from sympy.parsing import sympy_parser

unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "p . m",
    "lb",
    "tile",
    "per",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "month",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
    "kilogram",
    "second",
    "ampere",
    "A",
    "K",
    "mol",
    "cd",
    "N",
    "J",
    "W",
    "Pa",
    "Hz",
    "C",
    "V",
    "Ω",
    "F",
    "T",
    "H",
    "eV",
    "kW·h",
    "atm",
    "bar",
    "°C"
]

def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr

def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False

def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False

def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))

def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False

def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)

def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step

def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()

def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer

def sympy_normalize_answer(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _expr = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", expr)
            if _expr != "":
                expr = _expr

    expr = re.sub(rf"\^ *\\circ", "", expr)


    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr

def judge_MC(pred, gold):
    common_answer = [chr(i) for i in range(65, 91)] # 'A'~'Z'
    if pred == gold:
        return True
    else:
        if pred.startswith("[") and pred.endswith("]"):
            pred = pred.strip("[]")
        if not pred:
            return False
        if pred[0] in common_answer and (len(pred) > 1 and (pred[1] == "." or pred[1] == ":")):
            return pred[0] == gold
        if "'{}'".format(gold) in pred:
            return True
        else:
            return False

def judge_TF(pred, gold):
    def contains_chinese(d):
        def is_chinese_char(ch):
            return '\u4e00' <= ch <= '\u9fff'

        def check(value):
            if isinstance(value, str):
                return any(is_chinese_char(ch) for ch in value)
            elif isinstance(value, dict):
                return any(check(v) for v in value.values())
            elif isinstance(value, list):
                return any(check(item) for item in value)
            return False

        return check(d)

    if contains_chinese(pred):
        if pred in ["是", "对", "正确", "能"]:
            pred = "TRUE"
        elif pred in ["否", "错", "错误", "不能"]:
            pred = "FALSE"
    else:
        pred = pred.upper()
    answers = ["TRUE", "FALSE", "T", "F", "YES", "NO", "Y", "N"]
    gold = gold.upper()
    if gold not in answers or pred not in answers:
        return False
    if gold in ["TRUE", "YES", "T", "Y"]:
        gold = "TRUE"
    if gold in ["FALSE", "NO", "F", "N"]:
        gold = "FALSE"
    if pred in ["TRUE", "YES", "T", "Y"]:
        pred = "TRUE"
    if pred in ["FALSE", "NO", "F", "N"]:
        pred = "FALSE"
    return pred == gold

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)
    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True, given_answer_normalized_mathd, ground_truth_normalized_mathd
    return False, given_answer_normalized_mathd, ground_truth_normalized_mathd

BAD_SUBSTRINGS = ["^{", "^("]

BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]

TUPLE_CHARS = "()[]"

def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems

def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
        evaluate=False,
    )

def should_allow_eval(expr: str):
    def count_unknown_letters_in_expr(expr: str):
        expr = expr.replace("sqrt", "")
        expr = expr.replace("frac", "")
        letters_in_expr = set([x for x in expr if x.isalpha()])
        return len(letters_in_expr)
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True

def handle_pi(string, pi):
    if isinstance(string, str) and "pi" in string:
        # Find the first occurrence of "\pi"
        idx = string.find("pi")

        # Iterate over the string and find all occurrences of "\pi" with a valid previous character
        while idx != -1:

            if idx > 0 and string[idx - 1].isdigit():
                # Replace "\pi" with "*math.pi" if the previous character is a digit
                string = string[:idx] + "*{}".format(pi) + string[idx + 2:]
            else:
                # Replace "\pi" with "1*math.pi" if the previous character is not a digit
                string = string[:idx] + "1*{}".format(pi) + string[idx + 2:]

            # Find the next occurrence of "\pi"
            idx = string.find("pi", idx + 1)

        # Evaluate the expression using eval() function
        try:
            string = eval(string)
        except:
            pass

    return string

def are_equal_under_sympy(gold: str, pred: str, precision: float = 2e-3):
    def is_scientific_notation(expr):
        return (
            isinstance(expr, Mul)
            and isinstance(expr.args[1], Pow)
            and expr.args[1].args[0] == 10
        )

    def to_scientific_notation_sympy(num):
        num_sci = "{:.2e}".format(num)  # e.g., "1.23e-5"
        base, exponent = num_sci.split("e")
        return "{}*10**{}".format(base, int(exponent))

    def count_decimal_places(x, tol=1e-6):
        """
        返回浮点数 x 的有效小数位数，只保留重要前几位，忽略接近 0 的浮点尾巴。
        """
        with localcontext() as ctx:
            ctx.prec = 20  # 高精度防止误差
            d = Decimal(str(x)).normalize()
            s = format(d, "f")  # 固定点格式
            if "." not in s:
                return 0
            integer_part, decimal_part = s.split(".")
            # 去掉右侧全是0或接近0的部分（人为容差）
            clean_decimal = ""
            for i, ch in enumerate(decimal_part):
                clean_decimal += ch
                if abs(x - round(x, i+1)) <= tol:
                    break

            return len(clean_decimal)

    try:
        if pred == gold:
            return True

        # 尝试转为 float 后做相对误差比较
        pred_value = float(pred)
        gold_value = float(gold)
        min_decimal_places = min(count_decimal_places(gold_value), count_decimal_places(pred_value))

        pred_value = round(pred_value, min_decimal_places)
        gold_value = round(gold_value, min_decimal_places)
        if abs((pred_value - gold_value) / gold_value) <= precision * 1.01:
            return True

        # 转为科学记数法后转 sympy 表达式
        spred = _sympy_parse(to_scientific_notation_sympy(float(pred)))
        sgold = _sympy_parse(to_scientific_notation_sympy(float(gold)))
        if is_scientific_notation(spred) and is_scientific_notation(sgold):
            base_pred, exponent_pred = N(spred.args[0]), N(spred.args[1].args[1])
            base_gold, exponent_gold = N(sgold.args[0]), N(sgold.args[1].args[1])
            min_decimal_places = min(count_decimal_places(base_gold), count_decimal_places(base_pred))
            base_pred = round(base_pred, min_decimal_places)
            base_gold = round(base_gold, min_decimal_places)
            if exponent_pred == exponent_gold and abs(base_pred - base_gold) <= precision * 1.01:
                return True
    except Exception:
        pass

    # 如果上面都失败，退回原始符号化处理（但注意保留结构）
    try:
        if should_allow_eval(gold) and should_allow_eval(pred):
            exp_gold = _sympy_parse(gold)
            exp_pred = _sympy_parse(pred)

            expr = (exp_gold - exp_pred) / (exp_gold if exp_gold != 0 else 1)
            simplified = sympy.simplify(expr)
            if abs(N(simplified)) <= precision * 1.01:
                return True
            if is_scientific_notation(exp_pred) != is_scientific_notation(exp_gold):
                if is_scientific_notation(exp_pred):
                    gold = to_scientific_notation_sympy(float(gold))
                    exp_gold = _sympy_parse(gold)
                else:
                    pred = to_scientific_notation_sympy(float(pred))
                    exp_pred = _sympy_parse(pred)

            if is_scientific_notation(exp_pred) and is_scientific_notation(exp_gold):
                base_pred, exponent_pred = N(exp_pred.args[0]), N(exp_pred.args[1].args[1])
                base_gold, exponent_gold = N(exp_gold.args[0]), N(exp_gold.args[1].args[1])
                min_decimal_places = min(count_decimal_places(base_gold), count_decimal_places(base_pred))
                base_pred = round(base_pred, min_decimal_places)
                base_gold = round(base_gold, min_decimal_places)

                if exponent_pred == exponent_gold and abs(base_pred - base_gold) <= precision * 1.01:
                    return True
            else:
                if N(exp_pred) == N(exp_gold):
                    return True
    except Exception:
        pass

    return False

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = sympy_normalize_answer(ground_truth)
    given_normalized = sympy_normalize_answer(given_answer)
    if ground_truth_normalized is None:
        return False, given_normalized, ground_truth_normalized

    if ground_truth_normalized == given_normalized:
        return True, given_normalized, ground_truth_normalized

    if len(given_normalized) == 0:
        return False, given_normalized, ground_truth_normalized

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            # elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
            #     # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
            #     is_correct = False
            else:
                is_correct = judge_MC(given_elem, ground_truth_elem) or judge_TF(given_elem, ground_truth_elem)
                if not is_correct:
                    if "pi" in given_elem or "pi" in ground_truth_elem:
                        equivs = []
                        for pi in [math.pi, 3.14, 180]:
                            given_elem_pi = handle_pi(given_elem, pi)
                            ground_truth_elem_pi = handle_pi(ground_truth_elem, pi)
                            try:
                                equivs.append(are_equal_under_sympy(ground_truth_elem_pi, given_elem_pi))
                            except TimeoutError:
                                equivs.append(False)
                        is_correct = any(equivs)
                    else:
                        try:
                            is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                        except TimeoutError:
                            is_correct = False
            if not is_correct:
                break

    return is_correct, given_normalized, ground_truth_normalized

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None

def last_n_boxed_strings(string, n):
    boxed_list = []

    work_str = string[:]
    while work_str and len(boxed_list) < n:
        idx = work_str.rfind("\\boxed")
        if idx < 0:
            idx = work_str.rfind("\\fbox")

        if idx < 0:
            break

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(work_str):
            if work_str[i] == "{":
                num_left_braces_open += 1
            elif work_str[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is not None:
            boxed_expr = work_str[idx: right_brace_idx + 1]
            boxed_list.append(boxed_expr)
            work_str = work_str[:idx]
        else:
            work_str = work_str[:idx]

    boxed_list.reverse()
    return boxed_list

def get_answer_str(s: str, return_origin=False, num_answers=1):
    boxed_list = last_n_boxed_strings(s, num_answers)
    answer_list = [remove_boxed(b) if b else "" for b in boxed_list]

    missing = num_answers - len(answer_list)
    fill_str = s if return_origin else ""
    answer_list = [fill_str] * missing + answer_list

    return answer_list

def solution2answer(solution: str, math_mode="eval_peeking", return_origin=False, num_answers=1) -> tuple[bool, list | str]:
    answer = solution
    if math_mode == "eval_peeking":
        answer = get_answer_str(solution, return_origin, num_answers)
    else:
        raise ValueError("Invalid math_mode: {}".format(math_mode))
    return answer
