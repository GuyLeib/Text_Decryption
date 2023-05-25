import string
import numpy as np
import itertools
import  random
import copy
from bisect import bisect_left
random.seed(147)
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
population_size = 1000
score1_weight = 0.1
score2_weight = 0.7
score3_weight = 0.2
fitness_scores = []
###################### IMPORTANT ###########################
# check if need a more relative path. 

# This function creates all the possible permutations of the cipher:
def create_permutations():
    # create a list of dictionaries:
    permutations = []
    global population_size
    global letters
    letters_for_perm = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    for i in range(population_size):
        random.shuffle(letters_for_perm)
        dictionary = {}
        for j, letter in enumerate(letters):
            dictionary[letter] = letters_for_perm[j]
        permutations.append(dictionary)
    return permutations


def inversion(dict_list, rate):
    global letters
    
    for dictionary in dict_list:
         if random.random() <=rate:
            # choose 2 random letters:
            letter_1 = random.choice(letters)
            while letter_1=='x' or letter_1=='y'or letter_1=='z':
                letter_1 = random.choice(letters)
            index = letters.index(letter_1)
            # swap values between 2 keys:
            temp=dictionary[letters[index]]
            dictionary[letters[index]] = dictionary[letters[index+3]]
            dictionary[letters[index+3]] = temp
            temp=dictionary[letters[index+1]]
            dictionary[letters[index+1]] = dictionary[letters[index + 2]]
            dictionary[letters[index + 2]] = temp

    return dict_list


def mutation(dict_list, rate):
    global letters
    for dictionary in dict_list:
        # do the mutation over only 5 percent in the population:
        for m in range(3):
            if random.random() <= rate:
                # choose 2 random letters:
                letter_1 = random.choice(letters)
                letter_2 = random.choice(letters)
                while letter_2 == letter_1:
                    letter_2 = random.choice(letters)
                # swap values between 2 keys:
                temp = dictionary[letter_1]
                dictionary[letter_1] = dictionary[letter_2]
                dictionary[letter_2] = temp
    return dict_list

def local_optimization(n,algo_type):
    global letters
    global population
    global fitness_scores
    new_population=[]
    new_fitness_scores=[]
    for individual in fitness_scores:
        for i in range(n):
            new_individual = copy.deepcopy(individual)
            # choose 2 random letters:
            letter_1 = random.choice(letters)
            letter_2 = random.choice(letters)
            while letter_2 == letter_1:
                letter_2 = random.choice(letters)
            # swap values between 2 keys:
            temp = individual[0][letter_1]
            new_individual[0][letter_1] = new_individual[0][letter_2]
            new_individual[0][letter_2] = temp
            old_fitness=individual[0][2]
            new_fitness=fitness(new_individual)
        if new_fitness>=old_fitness:
            if algo_type=="lamark":
                new_population.append(new_individual)
            else:
                individual[2]=new_individual[2]
                new_population.append(individual)
            new_fitness_scores.append(individual)
        fitness_scores=copy.deepcopy(new_fitness_scores)
        population=copy.deepcopy(new_population)


def replace_dup(dictionary):
    updated_dict = {}
    encountered_values = set()
    for key, value in dictionary.items():
        if value in encountered_values:
            available_letters = string.ascii_lowercase
            used_letters = set(value.lower() for value in updated_dict.values())
            unused_letters = [letter for letter in available_letters if letter not in used_letters]
            if unused_letters:
                new_letter = random.choice(unused_letters)
                new_value = new_letter
                updated_dict[key] = new_value
                encountered_values.add(new_value)
        else:
            updated_dict[key] = value
            encountered_values.add(value)
    return updated_dict



def crossover(dict_list, next_gen_size):
    # create a new list of dictionaries
    crossover_results_list = []
    # create indexes dict in order to ease the crossover:
    global letters
    indexes_dict = {}
    for i in range(26):
        indexes_dict[i] = letters[i]
    for i in range(int(next_gen_size/2)):
        # choose 2 random parents:
        parent_1 = random.choice(dict_list)
        parent_2 = random.choice(dict_list)
        # make sure the parents are not the same:
        while parent_1 is parent_2:
            parent_2=random.choice(dict_list)
        # create new dict that will be the child:
        child_1 = {}
        child_2 = {}
        # choose a random index in the dict (but not the edges):
        r = random.randint(1, 24)
        # insert to the child the first r values from the first parent:
        for j in range(r):
            key_for_child=indexes_dict[j]
            child_1[key_for_child] = parent_1[key_for_child]
            child_2[key_for_child] = parent_2[key_for_child]
        for k in range(r, 26, 1):
            key_for_child = indexes_dict[k]
            child_1[key_for_child] = parent_2[key_for_child]
            child_2[key_for_child] = parent_1[key_for_child]
        # make sure the value doesn't already exist in the child's dict:
        child_1 = replace_dup(child_1)
        child_2 = replace_dup(child_2)
        # add the new dicts to the crossover list:
        crossover_results_list.append(child_1)
        crossover_results_list.append(child_2)

    return crossover_results_list

    # choose random index (bigger than 1)
    # take the first part of the first dictionary until the index and than take the rest from the second dictionary


# This function import and organize the helper files.
def import_helper_files():
    common_words = open("dict.txt", "r").read().split("\n")
    # Filter empty lines
    common_words = [word.lower() for word in common_words if word != ""]
    # Import common letters.
    common_letters = open("Letter_Freq.txt", "r").read().split("\n")
    # store in dictionary
    common_letters_dict = {}
    common_letters = [letter.split("\t") for letter in common_letters]
    for letter in common_letters:
        try:
            common_letters_dict[letter[1].lower()] = letter[0]
        except IndexError:  
            continue
    common_bigrams = open("Letter2_Freq.txt", "r").read().split("\n")
    common_bigrams_dict = {}
    common_bigrams = [bigram.split("\t") for bigram in common_bigrams]
    for bigram in common_bigrams:
        try:
            if bigram[1]!= "" and len(bigram[1]) == 2:
                common_bigrams_dict[bigram[1].lower()] = bigram[0]
        except IndexError:
            continue
    with open('enc.txt', 'r') as file:
        text = file.read()
        # split the text into words
    enc = text.split()
    with open('test1enc.txt', 'r') as file:
        text = file.read()
        # split the text into words
    test1 = text.split()
    with open('test2enc.txt', 'r') as file:
        text = file.read()
        # split the text into words
    test2 = text.split()
    return common_words, common_letters_dict, common_bigrams_dict, enc, test1, test2
# Import the helper files.
common_words, common_letters_dict, common_bigrams_dict, enc, test1,test2 = import_helper_files()



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
        # Use binary search to find the word in the common words list.
        index = bisect_left(common_words, word)
        if index != len(common_words) and common_words[index] == word:
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
    score = 1 - score
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

    score = 1 - score

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
def fitness(individual,enc=enc):
    global score1_weight, score2_weight, score3_weight   
    # Create the new text using the solution.
    decrypted_text = []
    for word in enc:
        dec_word =""
        for letter in word:
            try:
                dec_word += individual[letter]
            except KeyError:
                # Special letter case.
                dec_word += letter
        decrypted_text.append(dec_word)


    # Calculate score based of the frequency of the most common words in the english language.
    score1 = common_words_score(decrypted_text)
    # Calculate score based of the frequency of the most common letters in the english language.
    score2 = common_letters_score(decrypted_text)
    # Calculate the score based of the frequency of the most common bigrams in the english language.
    score3 = common_bigrams_score(decrypted_text)
    # Calculate the total fitness score 
    total_score = score1_weight*score1 + score2_weight*score2 + score3_weight*score3
    return total_score
    
def how_close_to_real_dict(dict,test):
    if test=="test1":
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
        solution  = {
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
        solution={'a': 'y', 'b': 'x', 'c': 'i', 'd': 'n', 'e': 't', 'f': 'o', 'g': 'z', 'h': 'j', 'i': 'c', 'j': 'e', 'k': 'b', 'l': 'l', 'm': 'd', 'n': 'u', 'o': 'k', 'p': 'm', 'q': 's', 'r': 'v', 's': 'p', 't': 'q', 'u': 'r', 'v': 'h', 'w': 'w', 'x': 'g', 'y': 'a', 'z': 'f'}
    same=0
    for key in dict.keys():
        if dict[key] == solution[key]:
            same+=1
    return same/26


def biased_crossover_list(fitness_scores):
    total_score_sum = sum(score[2] for score in fitness_scores)
    new_list=[]
    for individual in fitness_scores:
        amount=int(individual[2]*total_score_sum)
        for i in range(amount):
            new_list.append(individual)

    return new_list

# This function checks if the algorithm should stop.
def check_stop(generation, fitness):
    if generation>70 and fitness>0.86:
        return True
    if fitness > 0.91:
        return True
    return False

# This function writes the solution and the decrypted file to the files.
def write_solution(individual,enc):
    # Write the dictionary to a file called perm.txt.
    with open("Genetic_Algorithms_EX2/perm.txt", "w") as file:  
        for letter, value in individual.items():
            file.write(letter + " " + value + "\n")
    # Decrypt the text using the solution.
    decrypted_text = []
    for word in enc:
        dec_word =""
        for letter in word:
            try:
                dec_word += individual[letter]
            except KeyError:
                # Special letter case.
                dec_word += letter
        decrypted_text.append(dec_word)
    # Write the decrypted_text to plain.txt.
    with open("Genetic_Algorithms_EX2/plain.txt", "w") as file:
        for word in decrypted_text:
            file.write(word + " ")
    

# This function handles the flow of the # algorithm.
def decryption_flow(algo_type="classic"):
    fitness_history=0
    count_same_fitness=0
    rate = 0.2
    global score1_weight, score2_weight, score3_weight
    generations = 1000
    population = create_permutations()
    for i in range(generations):
        # Adapt the weights of the fitness scores.
        if i==0:
            score1_weight = 0.7
            score2_weight = 0.2
            score3_weight = 0.1
        # Store the calculated fitness score for each individual and the individual.
        global fitness_scores
        if algo_type=="classic":
            fitness_scores=[]
            for individual in population:
                words_perc, total_score=fitness(individual)
                fitness_scores.append((individual,words_perc, total_score))
        # Sort the population by descending fitness score.
        fitness_scores.sort(key=lambda x: x[2], reverse=True)
        best_solution=fitness_scores[0]
        print("Generation: " + str(i) + " Best solution: " + str(fitness_scores[0][0]) + " Fitness score: " + str(
            fitness_scores[0][2]) + " success percent: " + str(how_close_to_real_dict((fitness_scores[0][0]))))
        # Print the best solution in the current generation.
        if (fitness_history==fitness_scores[0][2]):
            count_same_fitness+=1
        else:
            count_same_fitness=0
        fitness_history=fitness_scores[0][2]
        # if i%2==0 and i<30:
        #     fitness_scores = [individual for individual in fitness_scores[0:int(len(fitness_scores) * 0.1)]]
        if count_same_fitness>=1:
            if rate<=0.8:
                rate=rate*1.2
            # if count_same_fitness>4:
            #     global population_size
            #     population_size+=100
            fitness_scores = [individual for individual in fitness_scores[0:int(len(fitness_scores) * 0.2)]]
        elif count_same_fitness==0:
            rate=rate*0.8
            # population_size -= 100

        # Create a list of the top 40-70% individuals of the population - for crossover.
        crossover_list = [individual[0] for individual in fitness_scores[0:int(len(fitness_scores) * 0.3)]]
        #new_crossover_list = biased_crossover_list(crossover_list)
        # Create a list of the top 70-90% of the population - for replication
        #replication_list = [individual[0] for individual in fitness_scores[int(len(fitness_scores)*0.1):int(len(fitness_scores)*0.2)]]
        # Create a list of the top 90-100% of the population - elitism.
        elitism_list = [individual[0] for individual in fitness_scores[0:int(len(fitness_scores)*0.05)]]

        elitism_to_mutate = copy.deepcopy(elitism_list)
        # Reset the population.
        population = []
        new_population = []
        # Create a new population using crossover.
        new_population = crossover(crossover_list, population_size)
        # Add the replication list to the new population.
        #new_population.extend(replication_list)
        # Mutate the new population.
        new_population = inversion(new_population, rate)
        new_population = mutation(new_population, rate)
        elitism_to_mutate = mutation(elitism_to_mutate, 1)
        new_population.extend(elitism_to_mutate)
        # Add the elitism list to the new population.
        new_population.extend(elitism_list)
        # Replace the old population with the new population.
        population = copy.deepcopy(new_population)
    if algo_type!="classic":
        # perform local optimization:
        n=5
        fitness_scores=local_optimization(fitness_scores,n,algo_type)



def testing():
    fitness_history1=0
    fitness_history2=0
    fitness_history3=0
    count_same_fitness1=0
    count_same_fitness2=0
    count_same_fitness3=0
    rate1 = 0.2
    rate2 = 0.2
    rate3 = 0.2
    global score1_weight, score2_weight, score3_weight
    generations = 1000
    population1 = create_permutations()
    population2 = create_permutations()
    population3 = create_permutations()
    for i in range(generations):
        # Adapt the weights of the fitness scores.
        if i==0:
            score1_weight = 0.7
            score2_weight = 0.2
            score3_weight = 0.1
        # Store the calculated fitness score for each individual and the individual.
        fitness_scores1 = []
        fitness_scores2 = []
        fitness_scores3 = []
        for individual1, individual2, individual3 in zip(population1, population2, population3):
            total_score1=fitness(individual1,test1)
            success_percent1=how_close_to_real_dict(individual1,"test1")
            fitness_scores1.append((individual1,success_percent1, total_score1))
            total_score2=fitness(individual2,test2)
            success_percent2=how_close_to_real_dict(individual2,"test2")
            fitness_scores2.append((individual2,success_percent2, total_score2))
            total_score3=fitness(individual3,enc)
            success_percent3=how_close_to_real_dict(individual3,"enc")
            fitness_scores3.append((individual3,success_percent3, total_score3))
        # Sort the population by descending fitness score.
        fitness_scores1.sort(key=lambda x: x[2], reverse=True)
        fitness_scores2.sort(key=lambda x: x[2], reverse=True)
        fitness_scores3.sort(key=lambda x: x[2], reverse=True)
        print("TEST1 "+"Generation: " + str(i) + " Fitness score: " + str(fitness_scores1[0][2]) + " success percent: " + str(fitness_scores1[0][1]))
        print("TEST2 " + "Generation: " + str(i) + " Fitness score: " + str(fitness_scores2[0][2]) + " success percent: " + str(fitness_scores2[0][1]))
        print("TEST3 "+" Generation: " + str(i) +  " Fitness score: " + str(fitness_scores3[0][2]) + " success percent: " + str(fitness_scores3[0][1]) +"\n\n")

        if (fitness_history1==fitness_scores1[0][2]):
            count_same_fitness1+=1
        else:
            count_same_fitness1=0
        fitness_history1=fitness_scores1[0][2]

        if count_same_fitness1>=1:
            if rate1<=0.8:
                rate1=rate1*1.2
            # Genetic drift
            fitness_scores1 = [individual for individual in fitness_scores1[0:int(len(fitness_scores1) * 0.1)]]
        elif count_same_fitness1==0:
            rate1=rate1*0.8

        if (fitness_history2==fitness_scores2[0][2]):
            count_same_fitness2+=1
        else:
            count_same_fitness2=0
        fitness_history2=fitness_scores2[0][2]

        if count_same_fitness2>=1:
            if rate2<=0.8:
                rate2=rate2*1.2
            # Genetic drift
            fitness_scores2 = [individual for individual in fitness_scores2[0:int(len(fitness_scores2) * 0.1)]]
        elif count_same_fitness2==0:
            rate2=rate2*0.8

        if (fitness_history3==fitness_scores3[0][2]):
            count_same_fitness3+=1
        else:
            count_same_fitness3=0
        fitness_history3=fitness_scores3[0][2]

        if count_same_fitness3>=1:
            if rate3<=0.8:
                rate3=rate3*1.2
            # Genetic drift
            fitness_scores3 = [individual for individual in fitness_scores3[0:int(len(fitness_scores3) * 0.1)]]
        elif count_same_fitness3==0:
            rate3=rate3*0.8
        

        # Create a list of the top 40-70% individuals of the population - for crossover.
        test1_crossover_list = [individual[0] for individual in fitness_scores1[0:int(len(fitness_scores1) * 0.3)]]
        test1_elitism_list = [individual[0] for individual in fitness_scores1[0:int(len(fitness_scores1)*0.1)]]
        test1_elitism_to_mutate = copy.deepcopy(test1_elitism_list)
        test2_crossover_list = [individual[0] for individual in fitness_scores2[0:int(len(fitness_scores2) * 0.3)]]
        test2_elitism_list = [individual[0] for individual in fitness_scores2[0:int(len(fitness_scores2)*0.1)]]
        test2_elitism_to_mutate = copy.deepcopy(test2_elitism_list)
        test3_crossover_list = [individual[0] for individual in fitness_scores3[0:int(len(fitness_scores3) * 0.3)]]
        test3_elitism_list = [individual[0] for individual in fitness_scores3[0:int(len(fitness_scores3)*0.1)]]
        test3_elitism_to_mutate = copy.deepcopy(test3_elitism_list)

        # Reset the population.
        population1 = []
        population2 = []
        population3 = []
        new_population1 = []
        new_population2 = []
        new_population3 = []
        # Create a new population using crossover.
        new_population1 = crossover(test1_crossover_list, population_size)
        new_population2 = crossover(test2_crossover_list, population_size)
        new_population3 = crossover(test3_crossover_list, population_size)
        # Add the replication list to the new population.
        #new_population.extend(replication_list)
        # Mutate the new population.
        new_population1 = inversion(new_population1, rate1)
        new_population1 = mutation(new_population1, rate1)   
        test1_elitism_to_mutate = mutation(test1_elitism_to_mutate, 1)
        new_population1.extend(test1_elitism_to_mutate)
        # Add the elitism list to the new population.
        new_population1.extend(test1_elitism_list)
        # Replace the old population with the new population.
        population1 = copy.deepcopy(new_population1)

        new_population2 = inversion(new_population2, rate2)
        new_population2 = mutation(new_population2, rate2)
        test2_elitism_to_mutate = mutation(test2_elitism_to_mutate, 1)
        new_population2.extend(test2_elitism_to_mutate)
        # Add the elitism list to the new population.
        new_population2.extend(test2_elitism_list)
        # Replace the old population with the new population.
        population2 = copy.deepcopy(new_population2)

        new_population3 = inversion(new_population3, rate3)
        new_population3 = mutation(new_population3, rate3)
        test3_elitism_to_mutate = mutation(test3_elitism_to_mutate, 1)
        new_population3.extend(test3_elitism_to_mutate)
        # Add the elitism list to the new population.
        new_population3.extend(test3_elitism_list)
        # Replace the old population with the new population.
        population3 = copy.deepcopy(new_population3)




#decryption_flow()
testing()
