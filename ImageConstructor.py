import numpy as np
from keras.models import model_from_json
from PIL import Image
import random
import time
import pandas as pd

class GeneticImageAlghorithm:
    shape = (32,32,3)
    selection = 0.2
    crossover = 0.2
    mutation = 0.2
    class_index = 0
    population_size = 1000
    number_of_generations = 1000
    elitism = 0.1

    model = None

    image_progression = []
    fitness_progression = []

    def __init__(self, selection = 0.2, crossover = 0.4, mutation=0.05, class_index=2, shape = (32,32,3), population_size = 10000, number_of_generations = 100,elitism=0.1):
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.class_index = class_index
        self.shape = shape
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.elitism = elitism
        self.model = self.load_model('models/second_version_32x32/model.json','models/second_version_32x32/model.h5')

    def load_model(self, model_path, model_weights):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(model_weights)

        return model

    def save_progression(self):
        df = pd.DataFrame(columns=['fitness','image_loc'])
        for idx, (image, fitness) in enumerate(zip(self.image_progression, self.fitness_progression)):
            im_array = np.asarray(image)
            im = Image.fromarray(im_array.astype('uint8'))
            path = f"progression{idx}.jpg"
            im.save(f'progression/{path}')
            df = df.append({'fitness':fitness,'image_loc':path},ignore_index=True)
        df.to_csv('progression/progression.csv')

    def run(self):
        now = time.time()
        # Create population amount of arrays (images) so If population is 1000, there will be a thousand arrays of the provided shape.
        def random_population():
            population_shape = (self.population_size,) + self.shape
            return np.random.randint(0,255,population_shape)

        #We use our ANN model to predict the probability of our class (class_index) being the instance.
        def calculate_fitness(single_instance):
            def preprocess_image(single_instance):
                single_instance = np.expand_dims(single_instance,axis=0)
                single_instance = single_instance.astype('float32')
                single_instance /= 255
                return single_instance

            prediction = self.model.predict(preprocess_image(single_instance))
            return prediction[0][self.class_index]

        #We calculate fitness scores for the whole population
        def population_fitness(population):
            population_fitness_score = np.asarray([])
            population_temp = np.asarray(population)
            population_temp = population_temp / 255
            population_fitness_score = self.model.predict(population_temp)
            print('shape', population_fitness_score)
            population_fitness_score_temp = np.asarray([])
            for instance in population_fitness_score:
                population_fitness_score_temp = np.append(population_fitness_score_temp,instance[self.class_index])

            return population_fitness_score_temp

        population = random_population()

        #Start of evolution
        for c in range(self.number_of_generations):
            #The new generation which will be created in this loop out of the current population
            descendents = np.empty(1)
            population_fitness_score = population_fitness(population)

            #Sorting the population based on it's fitness score
            if not isinstance(population, list):
                population = population.tolist()

            if not isinstance(population_fitness_score, list):
                population_fitness_score = population_fitness_score.tolist()

            population_fitness_score,population  = (list(t) for t in zip(*sorted(zip(population_fitness_score,population), key=lambda x: x[0],reverse=True)))

            if population_fitness_score[0]>= 0.995 :
                im_array = np.asarray(population[0])
                im = Image.fromarray(im_array.astype('uint8'))
                #print(self.model.predict(np.expand_dims(im_array,axis=0)/255))
                im.save("result2.png")
                im.show()
                self.save_progression()
                return

            self.image_progression.append(population[0])
            self.fitness_progression.append(population_fitness_score[0])

            print(population_fitness_score[0])
            print(self.model.predict(np.expand_dims(population[0],axis=0)/255))
            #Elitism - Add a few top scoring individuals
            descendents = np.asarray(population[:(int(round(self.population_size*self.elitism)))])

            print('After elitism population size: ',descendents.shape)

            for d in range( (self.population_size-int(round((self.elitism**self.population_size))//2)) ):

                first_parent = None
                second_parent = None

                #Choose parents as long as they're equal.. The function random.choices should already prevent that tho
                while np.array_equal(first_parent,second_parent):
                    choices = random.choices(descendents,k=2)
                    first_parent = choices[0].copy()
                    second_parent = choices[1].copy()

                #The chosen parents will now create children or descendents
                #First: Crossover
                #Each pixel of a parent will have a chance to cross
                for i in range(len(first_parent)):
                    for j in range(len(first_parent[i])):
                        #Crossover
                        if random.random() <= self.crossover:
                            temp_pixel = first_parent[i][j].copy()
                            first_parent[i][j] = second_parent[i][j]
                            second_parent[i][j] = temp_pixel
                            #logging.warning('Crossover happened')

                # Mutation
                if (random.random() <= self.mutation):
                    first_parent[random.randint(0,self.shape[0]-1)][random.randint(0,self.shape[1]-1)] = np.random.randint(0,255,self.shape[2])
                    second_parent[random.randint(0, self.shape[0]-1)][random.randint(0, self.shape[1]-1)] = np.random.randint(0, 255, self.shape[2])
                if not isinstance(descendents,list):
                    descendents = descendents.tolist()
                descendents.append(first_parent)
                descendents.append(second_parent)
                #print('Population after ',d,' ', descendents.shape)
            population = descendents






GeneticImageAlghorithm().run()