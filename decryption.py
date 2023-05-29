import string
import itertools
import random
import copy
import math
from bisect import bisect_left
import tkinter as tk
from tkinter import ttk
import threading
import csv

random.seed(147)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
           'w', 'x', 'y', 'z']
population_size = 100
score1_weight = 0.5
score2_weight = 0.3
score3_weight = 0.2
num_of_mut = 1
population = []
statistics_per_generation = []
fitness_scores = []
n = 1
fitness_calling = 0
algo_type="classic"


'''This function creates all the possible permutations of the cipher.'''
def create_permutations():
    # create a list of dictionaries:
    permutations = []
    global population_size
    global letters
    letters_for_perm = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                        't', 'u', 'v', 'w', 'x', 'y', 'z']
    for i in range(population_size):
        random.shuffle(letters_for_perm)
        dictionary = {}
        for j, letter in enumerate(letters):
            dictionary[letter] = letters_for_perm[j]
        permutations.append(dictionary)
    return permutations



''' This function gets a list of dictionary and rate and performs an
 inversion operation on the dictionaries within the list based on the given rate.'''
def inversion(dict_list, rate):
    global letters

    for dictionary in dict_list:
        if random.random() <= rate:
            # choose 2 random letters (excluding 'x', 'y', and 'z'):
            letter_1 = random.choice(letters)
            while letter_1 == 'x' or letter_1 == 'y' or letter_1 == 'z':
                letter_1 = random.choice(letters)
            # Find the index of the first letter in the 'letters' sequence:
            index = letters.index(letter_1)
            # swap values between 2 keys:
            # Swap values of letters[index] and letters[index + 3]:
            temp = dictionary[letters[index]]
            dictionary[letters[index]] = dictionary[letters[index + 3]]
            dictionary[letters[index + 3]] = temp
            # Swap values of letters[index + 1] and letters[index + 2]
            temp = dictionary[letters[index + 1]]
            dictionary[letters[index + 1]] = dictionary[letters[index + 2]]
            dictionary[letters[index + 2]] = temp

    return dict_list


''' This function takes in a list of dictionaries and a rate and performs a mutation 
operation on a portion of the dictionaries within the list based on the given rate. '''
def mutation(dict_list, rate):
    global letters
    global num_of_mut
    for dictionary in dict_list:
        # do the mutation over only 5 percent in the population:
        for m in range(num_of_mut):
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

'''This function gets n (number) and algo_type and perform n local optiomaztions.
  if the algo is "lamark" the function will keep both improved fitness and improved individual
  if algo is "Darwin" the function will keep only improved fitness. '''
def local_optimization(n, algo_type):
    global letters
    global population
    global fitness_scores
    new_population = []
    new_fitness_scores = []
    # Iterate over each individual in the population
    for individual in population:
        new_individual = copy.deepcopy(individual)
        new_fitness = 0
        # Perform  n local optimization on the individual:
        for i in range(n):
            # choose 2 random letters:
            letter_1 = random.choice(letters)
            letter_2 = random.choice(letters)
            while letter_2 == letter_1:
                letter_2 = random.choice(letters)
            # swap values between 2 keys:
            temp = individual[letter_1]
            new_individual[letter_1] = new_individual[letter_2]
            new_individual[letter_2] = temp
            # Compute fitness scores for old and new individuals
            old_fitness = fitness(individual)
            new_fitness = fitness(new_individual)
            # Check if the new individual's fitness is better than the old individual:
            if new_fitness < old_fitness:
                new_individual = copy.deepcopy(individual)
                new_fitness = old_fitness
        if algo_type == "lamark":
            # in "Lamark", append the new individual and its fitness score to the new population:
            new_population.append(new_individual)
            new_fitness_scores.append((new_individual, new_fitness))
        else:
            # in "Darwin", append the only the new fitness score:
            new_population.append(individual)
            new_fitness_scores.append((individual, new_fitness))
    fitness_scores = copy.deepcopy(new_fitness_scores)
    population = copy.deepcopy(new_population)


'''
This function checks if there are duplicated letters in the dict and if so it randomly replace them .
'''
def replace_dup(dictionary):
    updated_dict = {}
    encountered_values = set()
    # Iterate over each key-value pair in the dictionary
    for key, value in dictionary.items():
        # Value encountered previously, need to replace it:
        if value in encountered_values:
            available_letters = string.ascii_lowercase
            used_letters = set(value.lower() for value in updated_dict.values())
            unused_letters = [letter for letter in available_letters if letter not in used_letters]
            # Check if there are unused letters available:
            if unused_letters:
                new_letter = random.choice(unused_letters)
                new_value = new_letter
                updated_dict[key] = new_value
                encountered_values.add(new_value)
        else:
            # Unique value encountered, add it to the updated dictionary:
            updated_dict[key] = value
            encountered_values.add(value)
    return updated_dict



'''
This function performs crossover between random parents individuals from a
 given dictionary list to generate a desired number of offspring.
'''
def crossover(dict_list, next_gen_size):
    # create a new list of dictionaries
    crossover_results_list = []
    # create indexes dict in order to ease the crossover:
    global letters
    indexes_dict = {}
    for i in range(26):
        indexes_dict[i] = letters[i]
    for i in range(int(next_gen_size)):

        # choose 2 random parents:
        parent_1 = random.choice(dict_list)
        parent_2 = random.choice(dict_list)
        # make sure the parents are not the same:
        while parent_1 is parent_2:
            parent_2 = random.choice(dict_list)
        # create new dict that will be the child:
        child_1 = {}
        child_2 = {}
        # choose a random index in the dict (but not the edges):
        r = random.randint(1, 24)
        # insert to the child the first r values from the first parent:
        for j in range(r):
            key_for_child = indexes_dict[j]
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
        # crossover_results_list.append(child_2)

    return crossover_results_list

''' 
This function imports and organizes the helper files.
'''
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
            if bigram[1] != "" and len(bigram[1]) == 2:
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
common_words, common_letters_dict, common_bigrams_dict, enc, test1, test2 = import_helper_files()


'''
 This function calculate a score based on how many common words are in the solution.
 '''
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


'''
This function calculate a score based on how many common letters are in the solution.
'''
def common_letters_score(decrypted_text):
    # Calculate each letter frequency in the text.
    letter_freq = {}
    total_letters = 0
    for word in decrypted_text:
        for letter in word:
            if letter in letter_freq:
                letter_freq[letter] += 1
                total_letters += 1
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
            score += (letter_freq[letter] - float(common_letters_dict[letter])) ** 2
    score = score ** 0.5
    # normalize the score to be between 0 and 1
    # score = 1 - (score / len(common_letters_dict))
    score = 1 - score
    return score


'''
 This function calculate a score based on how many common bigrams are in the solution.
'''
def common_bigrams_score(decrypted_text):
    # Calculate each bigram frequency in the text.
    bigram_freq = {}
    total_bigrams = 0
    for word in decrypted_text:
        for i in range(len(word) - 1):
            bigram = word[i:i + 2]
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
            score += (final_bigram_freq[bigram] - float(known_bigrams_freq[bigram])) ** 2

    score = score ** 0.5

    score = 1 - score

    return score



'''
 This function calculate the fitness score of a given individual = solution.
 '''
def fitness(individual, enc=enc):
    global score1_weight, score2_weight, score3_weight, fitness_calling
    fitness_calling += 1
    # Create the new text using the solution.
    decrypted_text = []
    for word in enc:
        dec_word = ""
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
    total_score = score1_weight * score1 + score2_weight * score2 + score3_weight * score3
    return total_score


'''
This function performs biased selection of individuals from a crossover list based on their scores.
'''
def biased_crossover_list(crossover_list):
    new_list = []
    for individual in fitness_scores:
        relative_score = (math.ceil(individual[1] * 10)) + 1
        for i in range(int(relative_score)):
            new_list.append(individual[0])
    return new_list


'''
This function writes the solution and the decrypted file to the files.
'''
def write_solution(individual):
    # Write the dictionary to a file called perm.txt.
    with open("perm.txt", "w") as file:
        for letter, value in individual.items():
            file.write(letter + " " + value + "\n")

    # Read the encrypted text from the file
    with open("enc.txt", "r") as file:
        enc = [line.strip().split() for line in file]

    # Decrypt the text using the solution.
    decrypted_text = []
    for line in enc:
        dec_line = []
        for word in line:
            dec_word = ""
            for letter in word:
                try:
                    dec_word += individual[letter]
                except KeyError:
                    # Special letter case.
                    dec_word += letter
            dec_line.append(dec_word)
        decrypted_text.append(' '.join(dec_line))

    # Write the decrypted_text to plain.txt.
    with open("plain.txt", "w") as file:
        for line in decrypted_text:
            file.write(line + "\n")


'''
This function handles the flow of the  algorithm.
'''
def decryption_flow():
    global population
    global algo_type
    algo_type = ToggleButton.selected
    fitness_history = 0
    count_same_fitness = 0
    rate = 0.1
    global score1_weight, score2_weight, score3_weight, num_of_mut,fitness_calling
    generations = 200
    population = create_permutations()
    best_solution = {}
    progress_bar.start()
    for i in range(generations):
        # Store the calculated fitness score for each individual and the individual.
        global fitness_scores
        if algo_type == "classic" or i == 0:
            fitness_scores = []
            for individual in population:
                total_score = fitness(individual)
                fitness_scores.append((individual, total_score))

        # Sort the population by descending fitness score.
        avg_score = sum(item[1] for item in fitness_scores) / len(fitness_scores)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        statistics_per_generation.append((i, fitness_scores[0][1], avg_score, fitness_scores[-1][1]))

        if i > 0:
            # for lamarak or darwin:
            if algo_type != "classic":
                # perform local optimization:
                global n
                local_optimization(n, algo_type)
            # Keep only the 100 highest individuals to next generation:
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            fitness_scores = [individual for individual in fitness_scores[0:100]]
            population = [individual[0] for individual in fitness_scores]
        # check if there is local  convergence and handle it:
        if fitness_history == fitness_scores[0][1]:
            count_same_fitness += 1
        else:
            count_same_fitness = 0
        fitness_history = fitness_scores[0][1]
        if count_same_fitness > 1:
            num_of_mut += 1
        # danger for convergence is over:
        elif count_same_fitness == 0 and num_of_mut > 1:
            num_of_mut -= 1
        # found a solution:
        if count_same_fitness > 9:
            progress_bar.stop()
            progress_label.config(text="Decryption completed.")
            progress_bar.pack_forget()
            progress_label.pack_forget()
            write_solution(fitness_scores[0][0])
            display_solution(individual,i,fitness_calling)
            num_of_mut=1
            fitness_calling=0
            return

        # create a biased list for crossover:
        crossover_list = [individual for individual in fitness_scores[0:int(len(fitness_scores) * 0.8)]]
        new_crossover_list = biased_crossover_list(crossover_list)
        # Reset the  new population.
        new_population = []
        # add the old population to the new pop:
        new_population.extend(population)
        # Create  new offsprings using crossover.
        crossover_to_append = crossover(new_crossover_list, len(new_crossover_list))
        # add the offsprings to the new pop:
        new_population.extend(crossover_to_append)
        # perform mutations on the new population:
        new_population = mutation(new_population, rate)
        # add the old population to the new pop in order to save solutions:
        new_population.extend(population)
        # Replace the old population with the new population.
        population = copy.deepcopy(new_population)


def create_data(algo_type):
    n_list = [1, 5, 7, 10]
    filename = "{}.csv".format(algo_type)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        table_headers = ['generation', 'best', 'avg', 'bad']
        if algo_type != "classic":

            # Run lamark or darwin algo and save results:
            global n, statistics_per_generation, fitness_calling, population, fitness_scores, num_of_mut
            for num in n_list:
                n = num
                decryption_flow(algo_type)
                title = [algo_type + str(n), ' ']
                writer.writerow(title)
                writer.writerow(table_headers)
                for row in statistics_per_generation:
                    writer.writerow(row)
                writer.writerow([])
                statistics_per_generation = []
                title = [algo_type + str(n) + ' ' + 'fitness_calling: ' + str(fitness_calling)]
                writer.writerow(title)
                writer.writerow([])
                fitness_calling = 0
                population = []
                fitness_scores = []
                num_of_mut = 1
        else:
            # Run classic algo:
            decryption_flow()
            title = ['classic:', ' ']
            writer.writerow(title)
            writer.writerow(table_headers)
            for row in statistics_per_generation:
                writer.writerow(row)
            writer.writerow([])
            statistics_per_generation = []
            title = ['classic:' + 'fitness_calling: ' + str(fitness_calling)]
            writer.writerow(title)
            writer.writerow([])
            fitness_calling = 0

def start_decryption():
    progress_frame.place(x=50, y=275)  # place frame under the start button
    progress_bar.start()
    progress_label.config(text="Decrypting the text...")
    progress_bar.pack()
    progress_label.pack()
    thread = threading.Thread(target=decryption_flow)
    thread.daemon = True
    thread.start()

# This class implement a tk button and manage color changing and keeping track of which buttin is pressed.
class ToggleButton(tk.Button):
    selected = 'classic'
    def __init__(self, *args, **kwargs):
        self.algorithm = kwargs.pop('algorithm', None)
        super().__init__(*args, **kwargs)
        self.bind("<Button-1>", self.toggle)

    def toggle(self, event):
        for button in self.master.winfo_children():
            if isinstance(button, ToggleButton):
                button.config(bg='black', fg='lightgrey')
        self.config(bg='green', fg='white')
        ToggleButton.selected = self.algorithm

# GUI implementation
def display_solution(solution_dict, num_generations, num_fitness_calls):
    # Convert the solution_dict to a string representation for display
    add_to_solution_text = "Solution:\n"
    solution_text = ', '.join(f'{k}->{v}' for k, v in solution_dict.items())

    solution_text = add_to_solution_text + solution_text
    # Read decrypted text from plain.txt
    with open("plain.txt", "r") as file:
        decrypted_text = file.read()

    # Create new window
    new_window = tk.Toplevel(root)
    new_window.title("Decryption Results")
    new_window.configure(bg='black')

    # Create labels for the solution and statistics
    solution_label = tk.Label(new_window, text=f'{solution_text}', font=('Courier New', 10), fg='lightgrey',
                              bg='black', wraplength=500)
    generations_label = tk.Label(new_window, text=f'Number of Generations: {num_generations}', font=('Courier New', 10),
                                 fg='lightgrey', bg='black')
    fitness_calls_label = tk.Label(new_window, text=f'Number of Fitness Calls: {num_fitness_calls}',
                                   font=('Courier New', 10), fg='lightgrey', bg='black')

    # Create scrollable text box for the decrypted text
    decrypted_text_box = tk.Text(new_window, font=('Courier New', 10), fg='lightgrey', bg='black', wrap="word")
    decrypted_text_box.insert(tk.END, f'Decrypted text:\n{decrypted_text}')
    decrypted_text_box.configure(state='disabled')  # Make it read-only

    # Create a scrollbar and attach it to the text box
    scrollbar = tk.Scrollbar(new_window, command=decrypted_text_box.yview)
    decrypted_text_box['yscrollcommand'] = scrollbar.set

    # Place the labels and text box on the screen
    solution_label.grid(row=2, column=0, sticky='w', pady=10)
    generations_label.grid(row=0, column=0, sticky='w', pady=10)
    fitness_calls_label.grid(row=1, column=0, sticky='w', pady=10)
    decrypted_text_box.grid(row=3, column=0, sticky='nsew')
    scrollbar.grid(row=3, column=1, sticky='ns')

    # Configure grid to allow text box to expand
    new_window.grid_columnconfigure(0, weight=1)
    new_window.grid_rowconfigure(3, weight=1)


# Create the main window
root = tk.Tk()
root.geometry("301x320")
root.title('Text Decryptions System')
root.configure(bg='black')

# A variable that will be assigned by the algo buttons.
algorithm = tk.StringVar(value='classic')

# Create the titles on the screen
title_label = tk.Label(root, text="Text Decryption System", font=('Courier New', 16, 'bold'), fg='lightgrey', bg='black')
sub_title_label = tk.Label(root, text="Genetic Algorithm Powered", font=('Courier New', 12), fg='lightgrey', bg='black')
choose_algorithm_label = tk.Label(root, text="Choose the algorithm:", font=('Courier New', 14), fg='lightgrey', bg='black')

algorithm_frame = tk.Frame(root, bg='black')

# Creates the algorigm buttons
classic_button = ToggleButton(algorithm_frame, text="Classic", font=('Courier New', 10), bg='black', fg='lightgrey', algorithm='classic')
darwin_button = ToggleButton(algorithm_frame, text="Darwin", font=('Courier New', 10), bg='black', fg='lightgrey', algorithm='darwin')
lamark_button = ToggleButton(algorithm_frame, text="Lamark", font=('Courier New', 10), bg='black', fg='lightgrey', algorithm='lamark')

# Align the buttons on the screen.
classic_button.grid(row=0, column=0)
darwin_button.grid(row=0, column=1)
lamark_button.grid(row=0, column=2)
start_button = tk.Button(root, text="Start", command=start_decryption, font=('Courier New', 12), bg='blue', fg='white')

# Create a Frame to hold the progress bar and label
progress_frame = tk.Frame(root, bg='black')

# Create a progress bar and label initially hidden and will appear only when the algorithm starts.
progress_bar = ttk.Progressbar(progress_frame, length=200, mode='indeterminate')
progress_label = tk.Label(progress_frame, text="", font=('Courier New', 10), fg='lightgrey', bg='black')

progress_bar.pack()
progress_label.pack()

# Place all the labels on the screen
title_label.grid(row=0, column=0, pady=20)
sub_title_label.grid(row=1, column=0, pady=10)
choose_algorithm_label.grid(row=2, column=0, pady=20)
algorithm_frame.grid(row=3, column=0, pady=5)
start_button.grid(row=6, column=0, pady=20)

root.mainloop()
