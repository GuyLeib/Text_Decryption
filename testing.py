def how_close_to_real_dict(dict, test):
    if test == "test1":
        solution = {
            'a': 'q',
            'b': 'w',
            'c': 'e',
            'd': 'r',
            'e': 't',
            'f': 'y',
            'g': 'u',
            'h': 'i',
            'i': 'o',
            'j': 'p',
            'k': 'a',
            'l': 's',
            'm': 'd',
            'n': 'f',
            'o': 'g',
            'p': 'h',
            'q': 'k',
            'r': 'j',
            's': 'l',
            't': 'z',
            'u': 'x',
            'v': 'c',
            'w': 'v',
            'x': 'b',
            'y': 'n',
            'z': 'm'
        }

    elif test == "test2":
        solution = {
            'a': 'q',
            'b': 'w',
            'c': 'e',
            'd': 'r',
            'e': 't',
            'f': 'y',
            'g': 'u',
            'h': 'i',
            'i': 'o',
            'j': 'p',
            'k': 'a',
            'l': 's',
            'm': 'd',
            'n': 'f',
            'o': 'g',
            'p': 'h',
            'q': 'j',
            'r': 'k',
            's': 'l',
            't': 'z',
            'u': 'x',
            'v': 'c',
            'w': 'v',
            'x': 'b',
            'y': 'n',
            'z': 'm'
        }

    else:
        solution = {'a': 'y', 'b': 'x', 'c': 'i', 'd': 'n', 'e': 't', 'f': 'o', 'g': 'z', 'h': 'j', 'i': 'c', 'j': 'e',
                    'k': 'b', 'l': 'l', 'm': 'd', 'n': 'u', 'o': 'k', 'p': 'm', 'q': 's', 'r': 'v', 's': 'p', 't': 'q',
                    'u': 'r', 'v': 'h', 'w': 'w', 'x': 'g', 'y': 'a', 'z': 'f'}
    same = 0
    for key in dict.keys():
        if dict[key] == solution[key]:
            same += 1
    return same / 26
