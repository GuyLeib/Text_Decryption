import itertools
import  random
population_size = 1000
# This function creates all the possible permutations of the cipher:
def create_permutations():
    # create a list of dictionaries:
    permutations = []
    global population_size
    counter=0
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    letters_for_perm=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    for i in range(population_size):
        random.shuffle(letters_for_perm)
        dictionary = {}

        for j, letter in enumerate(letters):
            dictionary[letter] = letters_for_perm[j]
        permutations.append(dictionary)
    return permutations


permutations = create_permutations()
for dictionary in permutations:
    print(dictionary)