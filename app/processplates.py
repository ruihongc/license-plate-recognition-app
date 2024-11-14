def reordered_plates(plates):
    # plates = {} # plate: name
    reordered_plates = {}

    for i in plates:
        if (len(i) > 1) and i[-1].isalpha() and i[-2].isalpha():
            c = -1
            while ((-c) <= len(i)) and i[c].isalpha():
                c -= 1
            front = i[c:]
            back = i[:c]
        else:
            c = 0
            while (c < len(i)) and i[c].isalpha():
                c += 1
            front = i[:c]
            back = i[c:]
        reordered_plates[front+back] = plates[i]
        reordered_plates[back+front] = plates[i]

    # replace twice by mapping dict
    return {double_replace(i): reordered_plates[i] for i in reordered_plates}

number_replacements = {
    "0": "O",
    "1": "L",
    "2": "Z",
    "3": "3",
    "4": "A",
    "5": "S",
    "6": "6",
    "7": "7",
    "8": "B",
    "9": "9",
}

character_replacements = {
    "I": "L",
    "Q": "O",
    "D": "O",
    # "F": "P",
    "W": "VV",
    "M": "LVL",
    "N": "LV",
    "U": "V",
}

def replace(origin, mapping):
    return "".join(mapping[char] if char in mapping else char for char in origin)

def double_replace(plate):
    return replace(replace(plate, number_replacements), character_replacements)
