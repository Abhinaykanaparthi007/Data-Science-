import gradio as gr
import tempfile
import os 

def numerology_name_calculator(name):
    numerology_chart = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
        'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8
    }
    name = name.upper()
    total = sum(numerology_chart[char] for char in name if char in numerology_chart)
    return total

def initials_extractor(name):
    initials = ''.join([part[0].upper() for part in name.split()])
    return initials

def initials_numerology_calculator(name):
    initials = initials_extractor(name)
    numerology_sum = numerology_name_calculator(initials)
    return numerology_sum

def first_letter_numerology(first_name, last_name):
    numerology_chart = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
        'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8
    }
    first_letter_first_name = first_name[0].upper()
    first_letter_last_name = last_name[0].upper()
    
    first_name_num = numerology_chart.get(first_letter_first_name, 0)
    last_name_num = numerology_chart.get(first_letter_last_name, 0)
    
    return first_name_num, last_name_num

def reduce_to_single_digit(n):
    while n > 9 and n != 11 and n != 22:  # 11 and 22 are considered master numbers
        n = sum(int(digit) for digit in str(n))
    return n

def calculate_life_path_number(day, month, year):
    life_path_number = reduce_to_single_digit(day + month + year)
    return life_path_number

def calculate_expression_number(name):
    consonants = ''.join([char for char in name.upper() if char not in 'AEIOU'])
    expression_number = reduce_to_single_digit(numerology_name_calculator(consonants))
    return expression_number

def calculate_soul_urge_number(name):
    vowels = 'AEIOU'
    soul_urge_number = reduce_to_single_digit(numerology_name_calculator(''.join([char for char in name.upper() if char in vowels])))
    return soul_urge_number

def calculate_personality_number(name):
    consonants = ''.join([char for char in name.upper() if char not in 'AEIOU'])
    personality_number = reduce_to_single_digit(numerology_name_calculator(consonants))
    return personality_number

def calculate_birthday_number(day):
    birthday_number = reduce_to_single_digit(day)
    return birthday_number

def calculate_maturity_number(life_path_number, expression_number):
    maturity_number = reduce_to_single_digit(life_path_number + expression_number)
    return maturity_number

def calculate_current_name_number(current_name):
    current_name_number = numerology_name_calculator(current_name)
    return current_name_number

def calculate_balance_number(name):
    balance_number = reduce_to_single_digit(sum(numerology_name_calculator(char) for char in name.upper()))
    return balance_number

def calculate_hidden_passion_number(full_name):
    hidden_passion_number = reduce_to_single_digit(sum(numerology_name_calculator(char) for char in full_name.upper()))
    return hidden_passion_number

def calculate_karmic_lesson_number(full_name):
    numerology_chart = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
        'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8
    }
    full_name = full_name.upper()
    present_numbers = [numerology_chart[char] for char in full_name if char in numerology_chart]
    karmic_lesson_number = reduce_to_single_digit(sum(range(1, 10)) - sum(present_numbers))
    return karmic_lesson_number

def calculate_karmic_debt_number(full_name):
    karmic_debt_number = reduce_to_single_digit(numerology_name_calculator(full_name))
    return karmic_debt_number

def calculate_personal_year_number(day, month, year):
    personal_year_number = reduce_to_single_digit(day + month + year)
    return personal_year_number

def calculate_personal_month_number(personal_year_number, current_month):
    personal_month_number = reduce_to_single_digit(personal_year_number + current_month)
    return personal_month_number

def calculate_personal_day_number(personal_month_number, current_day):
    personal_day_number = reduce_to_single_digit(personal_month_number + current_day)
    return personal_day_number

def calculate_pinnacle_numbers(life_path_number):
    pinnacle_numbers = []
    for i in range(1, 5):
        pinnacle_numbers.append(reduce_to_single_digit(life_path_number + i))
    return pinnacle_numbers

def calculate_challenge_numbers(life_path_number):
    challenge_numbers = []
    for i in range(1, 5):
        challenge_numbers.append(reduce_to_single_digit(life_path_number - i))
    return challenge_numbers

def calculate_bridge_numbers(life_path_number, expression_number):
    bridge_numbers = [reduce_to_single_digit(life_path_number + expression_number)]
    return bridge_numbers

def calculate_essence_number(name, personal_year_number):
    essence_number = reduce_to_single_digit(numerology_name_calculator(name) + personal_year_number)
    return essence_number

def calculate_transit_letters(name):
    transit_letters = [reduce_to_single_digit(numerology_name_calculator(char)) for char in name.upper()]
    return transit_letters

def calculate_cornerstone(first_name):
    cornerstone = numerology_name_calculator(first_name[0].upper())
    return cornerstone

def calculate_capstone(first_name):
    capstone = numerology_name_calculator(first_name[-1].upper())
    return capstone

def calculate_subconscious_self_number(full_name):
    subconscious_self_number = reduce_to_single_digit(sum(numerology_name_calculator(char) for char in full_name.upper()))
    return subconscious_self_number

def calculate_planes_of_expression(name):
    planes = {
        'Physical': 0,
        'Mental': 0,
        'Emotional': 0,
        'Intuitive': 0
    }
    for char in name.upper():
        number = numerology_name_calculator(char)
        if number % 4 == 1:
            planes['Physical'] += 1
        elif number % 4 == 2:
            planes['Mental'] += 1
        elif number % 4 == 3:
            planes['Emotional'] += 1
        elif number % 4 == 0:
            planes['Intuitive'] += 1
    return planes

def calculate_intensity_number(name):
    intensity_numbers = [reduce_to_single_digit(numerology_name_calculator(char)) for char in name.upper()]
    intensity_number = max(intensity_numbers)
    return intensity_number

def calculate_rational_thought_number(day, month, year, name):
    rational_thought_number = reduce_to_single_digit(day + month + year + numerology_name_calculator(name))
    return rational_thought_number

def calculate_soul_path_number(life_path_number, soul_urge_number):
    soul_path_number = reduce_to_single_digit(life_path_number + soul_urge_number)
    return soul_path_number

def calculate_universal_year_number(current_year):
    universal_year_number = reduce_to_single_digit(current_year)
    return universal_year_number

def calculate_universal_month_number(current_month):
    universal_month_number = reduce_to_single_digit(current_month)
    return universal_month_number

def calculate_universal_day_number(current_day):
    universal_day_number = reduce_to_single_digit(current_day)
    return universal_day_number

def calculate_zodiac_sign_numbers(zodiac_sign):
    zodiac_numbers = {
        'Aries': 1, 'Taurus': 2, 'Gemini': 3, 'Cancer': 4, 'Leo': 5, 'Virgo': 6, 'Libra': 7, 'Scorpio': 8,
        'Sagittarius': 9, 'Capricorn': 10, 'Aquarius': 11, 'Pisces': 12
    }
    return zodiac_numbers.get(zodiac_sign, 0)

def calculate_personal_cycle_number(birth_date):
    personal_cycle_number = reduce_to_single_digit(numerology_name_calculator(birth_date))
    return personal_cycle_number

def numerology_report(first_name, last_name, day, month, year, current_day, current_month, current_year, zodiac_sign):
    full_name = f"{first_name} {last_name}"
    
    first_name_num = numerology_name_calculator(first_name)
    last_name_num = numerology_name_calculator(last_name)
    total_name_num = first_name_num + last_name_num
    
    initials = initials_extractor(full_name)
    initials_num = initials_numerology_calculator(full_name)
    
    life_path_number = calculate_life_path_number(day, month, year)
    expression_number = calculate_expression_number(full_name)
    soul_urge_number = calculate_soul_urge_number(full_name)
    personality_number = calculate_personality_number(full_name)
    birthday_number = calculate_birthday_number(day)
    maturity_number = calculate_maturity_number(life_path_number, expression_number)
    current_name_number = calculate_current_name_number(full_name)
    balance_number = calculate_balance_number(full_name)
    hidden_passion_number = calculate_hidden_passion_number(full_name)
    karmic_lesson_number = calculate_karmic_lesson_number(full_name)
    karmic_debt_number = calculate_karmic_debt_number(full_name)
    personal_year_number = calculate_personal_year_number(day, month, current_year)
    personal_month_number = calculate_personal_month_number(personal_year_number, current_month)
    personal_day_number = calculate_personal_day_number(personal_month_number, current_day)
    pinnacle_numbers = calculate_pinnacle_numbers(life_path_number)
    challenge_numbers = calculate_challenge_numbers(life_path_number)
    bridge_numbers = calculate_bridge_numbers(life_path_number, expression_number)
    essence_number = calculate_essence_number(full_name, personal_year_number)
    transit_letters = calculate_transit_letters(full_name)
    cornerstone = calculate_cornerstone(first_name)
    capstone = calculate_capstone(first_name)
    subconscious_self_number = calculate_subconscious_self_number(full_name)
    planes_of_expression = calculate_planes_of_expression(full_name)
    intensity_number = calculate_intensity_number(full_name)
    rational_thought_number = calculate_rational_thought_number(day, month, year, full_name)
    soul_path_number = calculate_soul_path_number(life_path_number, soul_urge_number)
    universal_year_number = calculate_universal_year_number(current_year)
    universal_month_number = calculate_universal_month_number(current_month)
    universal_day_number = calculate_universal_day_number(current_day)
    zodiac_sign_number = calculate_zodiac_sign_numbers(zodiac_sign)
    personal_cycle_number = calculate_personal_cycle_number(f"{day}-{month}-{year}")

    report = {
        "Numerology sum of first name ({}):".format(first_name): first_name_num,
        "Numerology sum of last name ({}):".format(last_name): last_name_num,
        "Total numerology sum:": total_name_num,
        "First letter numerology: A = {}, K = {}".format(first_name[0], last_name[0]): (first_letter_numerology(first_name, last_name)),
        "Initials of full name:": initials,
        "Numerology sum of initials:": initials_num,
        "Basic Number (Day):": birthday_number,
        "Attitude Number (Day + Month):": reduce_to_single_digit(day + month),
        "Life Path Number (Day + Month + Year):": life_path_number,
        "Expression Number (Destiny Number):": expression_number,
        "Soul Urge Number (Heart's Desire Number):": soul_urge_number,
        "Personality Number:": personality_number,
        "Birthday Number:": birthday_number,
        "Maturity Number:": maturity_number,
        "Current Name Number:": current_name_number,
        "Balance Number:": balance_number,
        "Hidden Passion Number:": hidden_passion_number,
        "Karmic Lesson Number:": karmic_lesson_number,
        "Karmic Debt Number:": karmic_debt_number,
        "Personal Year Number:": personal_year_number,
        "Personal Month Number:": personal_month_number,
        "Personal Day Number:": personal_day_number,
        "Pinnacle Numbers:": pinnacle_numbers,
        "Challenge Numbers:": challenge_numbers,
        "Bridge Numbers:": bridge_numbers,
        "Essence Number:": essence_number,
        "Transit Letters:": transit_letters,
        "Cornerstone:": cornerstone,
        "Capstone:": capstone,
        "Subconscious Self Number:": subconscious_self_number,
        "Planes of Expression:": planes_of_expression,
        "Intensity Number:": intensity_number,
        "Rational Thought Number:": rational_thought_number,
        "Soul Path Number:": soul_path_number,
        "Universal Year Number:": universal_year_number,
        "Universal Month Number:": universal_month_number,
        "Universal Day Number:": universal_day_number,
        "Zodiac Sign Number ({}):".format(zodiac_sign): zodiac_sign_number,
        "Personal Cycle Number:": personal_cycle_number
    }
    
    return report

# Existing numerology functions would go here...

def generate_numerology_report(first_name, last_name, day, month, year, current_day, current_month, current_year, zodiac_sign):
    report = numerology_report(first_name, last_name, day, month, year, current_day, current_month, current_year, zodiac_sign)
    report_text = "\n".join([f"{key} {value}" for key, value in report.items()])
    
    # Create a file name based on the first name
    file_name = f"{first_name}_Numerology.txt"
    
    # Create text file with the dynamic name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix=file_name) as temp_file:
        temp_file.write(report_text.encode())
        text_file_path = temp_file.name

    # Rename the file to remove the full temporary path and keep the desired name
    final_path = os.path.join(os.path.dirname(text_file_path), file_name)
    os.rename(text_file_path, final_path)

    return report_text, final_path

# Create Gradio Interface
iface = gr.Interface(
    fn=generate_numerology_report,
    inputs=[
        gr.Textbox(label="First Name"),
        gr.Textbox(label="Last Name"),
        gr.Number(label="Day of Birth"),
        gr.Number(label="Month of Birth"),
        gr.Number(label="Year of Birth"),
        gr.Number(label="Current Day"),
        gr.Number(label="Current Month"),
        gr.Number(label="Current Year"),
        gr.Textbox(label="Zodiac Sign")
    ],
    outputs=[
        gr.Textbox(label="Numerology Report"),
        gr.File(label="Download Report as Text File")
    ],
    title="Comprehensive Numerology Report",
    description="Enter your details to get a comprehensive numerology report."
)

iface.launch(share=True)