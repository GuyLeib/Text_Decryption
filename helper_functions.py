import itertools


# This function creates all the possible permutations of the cipher:
def create_permutations():
    # create a list of dictionaries:
    permutations = []
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    for permutation in itertools.permutations(letters):
        dictionary = {}
        for i, letter in enumerate(letters):
            dictionary[letter] = permutation[i]
        permutations.append(dictionary)
    return permutations


permutations = create_permutations()
for dictionary in permutations:
    print(dictionary)