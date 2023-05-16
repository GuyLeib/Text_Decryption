import string
import numpy as np
import itertools
import  random

population_size = 1000
###################### IMPORTANT ###########################
# check if need a more relative path. 

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

# This function import and organize the helper files.
def import_helper_files():
    common_words = open("Genetic_Algorithms_EX2/dict.txt", "r").read().split("\n")
    # Filter empty lines
    common_words = [word.lower() for word in common_words if word != ""]
    # Import common letters.
    common_letters = open("Genetic_Algorithms_EX2/Letter_Freq.txt", "r").read().split("\n")
    # store in dictionary
    common_letters_dict = {}
    common_letters = [letter.split("\t") for letter in common_letters]
    for letter in common_letters:
        try:
            common_letters_dict[letter[1].lower()] = letter[0]
        except IndexError:  
            continue
    common_bigrams = open("Genetic_Algorithms_EX2/Letter2_Freq.txt", "r").read().split("\n")
    common_bigrams_dict = {}
    common_bigrams = [bigram.split("\t") for bigram in common_bigrams]
    for bigram in common_bigrams:
        try:
            if bigram[1]!= "" and len(bigram[1]) == 2:
                common_bigrams_dict[bigram[1].lower()] = bigram[0]
        except IndexError:  
            continue
    with open('Genetic_Algorithms_EX2/enc.txt', 'r') as file:
        text = file.read()
        # split the text into words
    enc = text.split()
    return common_words, common_letters_dict, common_bigrams_dict, enc 

# Import the helper files.
common_words, common_letters_dict, common_bigrams_dict, enc = import_helper_files()



# This function calculate the KL divergence between two distributions.
def kl_divergence(p, q):
    # Adding a small constant to p and q to avoid division by zero
    epsilon = 1e-10
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    kl = np.sum(p * np.log(p / q))
    return kl

# This function calculate a score based on how many common words are in the solution.
def common_words_score(decrypted_text):
    # Create the new text using the solution.
    total_words = len(decrypted_text)
    score = 0
    for word in decrypted_text:
        if word in common_words:
            score += 1
    score = score / total_words
    return score

# This function calculate a score based on how many common letters are in the solution.
def common_letters_score(decrypted_text):
    # Calculate each letter frequency in the text.
    letter_freq = {}
    total_letters = 0
    for word in decrypted_text:
        for letter in word:
            if letter in letter_freq:
                letter_freq[letter] += 1
                total_letters +=1
            else:
                letter_freq[letter] = 1
                total_letters += 1

    # Normalize to achieve frequency.
    for letter in letter_freq:
        letter_freq[letter] = letter_freq[letter] / total_letters
    
    # Calculate the score.
    score = 0
    for letter, freq in letter_freq.items():
        if letter in common_letters_dict:
            score += (letter_freq[letter] - float(common_letters_dict[letter]))**2
    score = score ** 0.5
    # normalize the score to be between 0 and 1
    #score = 1 - (score / len(common_letters_dict))
    return score

# This function calculate a score based on how many common bigrams are in the solution.
def common_bigrams_score(decrypted_text):
    # Calculate each bigram frequency in the text.
    bigram_freq = {}
    total_bigrams = 0
    for word in decrypted_text:
        for i in range(len(word)-1):
            bigram = word[i:i+2]
            if bigram in bigram_freq:
                bigram_freq[bigram] += 1
                total_bigrams += 1
            else:
                bigram_freq[bigram] = 1
                total_bigrams += 1

    known_bigrams_freq = {}
    final_bigram_freq = {}
    # Normalize to achieve frequency.
    for bigram in bigram_freq:
        try:
            known_bigrams_freq[bigram] = common_bigrams_dict[bigram]
        except KeyError:
            continue
        final_bigram_freq[bigram] = bigram_freq[bigram] / total_bigrams
    
    # Calculate the score.
    score = 0
    for bigram, freq in final_bigram_freq.items():
        if bigram in known_bigrams_freq:
            score += (final_bigram_freq[bigram] - float(known_bigrams_freq[bigram]))**2

    score = score ** 0.5

    return score


# # This function calculate a score based on how many common bigrams are in the solution.
# def common_bigrams_score(decrypted_text):
#     # Calculate each bigram frequency in the decrypted text.
#     bigram_freq = {}
#     known_bigrams_freq = {}
#     total_bigrams = 0
#     for word in decrypted_text:
#         for i in range(len(word) - 1):
#             bigram = word[i] + word[i + 1]
#             if bigram in bigram_freq:
#                 bigram_freq[bigram] += 1
#             else:
#                 bigram_freq[bigram] = 1
#             total_bigrams += 1

#     final_bigram_freq = {}
#     # Normalize to achieve frequency.
#     for bigram in bigram_freq:
#         try:
#             known_bigrams_freq[bigram] = common_bigrams_dict[bigram]
#         except KeyError:
#             continue
#         final_bigram_freq[bigram] = bigram_freq[bigram] / total_bigrams

#     # Calculate the score.
#     p = np.array([final_bigram_freq[bigram] for bigram in final_bigram_freq.keys()])
#     q = np.array([float(known_bigrams_freq[bigram]) for bigram in known_bigrams_freq.keys()])

#     # Adding a small constant to p and q to avoid division by zero
#     epsilon = 1e-10
#     p = np.maximum(p, epsilon)
#     q = np.maximum(q, epsilon)

#     # Calculate the score.
#     score = kl_divergence(p, q)

#     normalized_score = 1 - np.exp(-score)
#     # Normalize the score between 0 and 1
#     max_score = kl_divergence(p, p)
#     score = 1 - (score / max_score)

#     return score


# This function calculate the fitness score of a given individual = solution. 
def fitness(individual):    
    # Create the new text using the solution.
    decrypted_text = []
    for word in enc:
        dec_word =""
        for letter in word:
            try:
                dec_word += individual[letter]
            except KeyError:
                continue
        decrypted_text.append(dec_word)


    # Calculate score based of the frequency of the most common words in the english language.
    score1 = common_words_score(decrypted_text)
    # Calculate score based of the frequency of the most common letters in the english language.
    score2 = common_letters_score(decrypted_text)
    # Calculate the score based of the frequency of the most common bigrams in the english language.
    score3 = common_bigrams_score(decrypted_text)
    # Calculate the total fitness score 
    total_score = 0.5*score1 + 0.3*score2 + 0.2*score3
    return total_score
    


# This function handles the flow of the genetic algorithm.
def decryption_flow():
    generations = 500
    population = create_permutations()
    for i in range(generations):
        # Store the calculated fitness score for each individual and the individual.
        fitness_scores = []
        for individual in population:
            fitness_scores.append((individual,fitness(individual)))
        # Sort the population by descending fitness score.
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        # Create a list of the top 40-70% individuals of the population - for crossover.
        crossover_list = [individual[0] for individual in fitness_scores[int(len(fitness_scores)*0.4):int(len(fitness_scores)*0.7)]]
        # Create a list of the top 70-90% of the population - for replication
        replication_list = [individual[0] for individual in fitness_scores[int(len(fitness_scores)*0.7):int(len(fitness_scores)*0.9)]]
        # Create a list of the top 90-100% of the population - elitism.
        elitism_list = [individual[0] for individual in fitness_scores[int(len(fitness_scores)*0.9):]]
        

decryption_flow()