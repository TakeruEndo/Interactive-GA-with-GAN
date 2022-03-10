import numpy as np
import math
import random
#from easygui import *

def crossover_matrix(variables, level, noise, indices):

    mutation = 0.5

    selected = np.array(indices)
    pop_size, latent_size = variables.shape
    #pop_size, latent_size = variables.shape
    select_size = selected.shape[0]
    crossover_size = pop_size
    #foreign_size = int(min(pop_size - select_size, foreign))

    #Crossover
    A = variables[randomChoice(selected, crossover_size)]
    B = variables[randomChoice(selected, crossover_size)]
    T = np.random.uniform(low=0, high=1, size=crossover_size)
    crossover = _slerp(T, A, B)

    #New random members
    #new = np.random.normal(loc=0, scale=1, size=(foreign_size, latent_size))
    #rnd = np.random.RandomState(None)
    #new = rnd.randn(foreign_size, input_shape)
    #new_dlatents = Gs.components.mapping.run(new, None)

    #Mutate
    next_pop = crossover
    next_pop = _mutate(next_pop, noise, mutation)
    #next_pop = np.concatenate((next_pop, new))

    return next_pop

def crossover_nm(variables, indices):

    #mutation = 0.5

    selected = np.array(indices)
    pop_size, latent_size = variables.shape
    #pop_size, latent_size = variables.shape
    select_size = selected.shape[0]
    crossover_size = pop_size
    #foreign_size = int(min(pop_size - select_size, foreign))

    #Crossover
    A = variables[randomChoice(selected, crossover_size)]
    B = variables[randomChoice(selected, crossover_size)]
    T = np.random.uniform(low=0, high=1, size=crossover_size)
    crossover = _slerp(T, A, B)

    #New random members
    #new = np.random.normal(loc=0, scale=1, size=(foreign_size, latent_size))
    #rnd = np.random.RandomState(None)
    #new = rnd.randn(foreign_size, input_shape)
    #new_dlatents = Gs.components.mapping.run(new, None)

    #Mutate
    next_pop = crossover
    #next_pop = _mutate(next_pop, noise, mutation)
    #next_pop = np.concatenate((next_pop, new))

    return next_pop

def crossover3(variables, noise, indices):

    mutation = 0.5

    selected = np.array(indices)
    pop_size, latent_size = variables.shape
    #pop_size, latent_size = variables.shape
    select_size = selected.shape[0]
    crossover_size = int(max(pop_size - select_size, 0))
    #foreign_size = int(min(pop_size - select_size, foreign))

    #Crossover
    A = variables[randomChoice(selected, crossover_size)]
    B = variables[randomChoice(selected, crossover_size)]
    T = np.random.uniform(low=0, high=1, size=crossover_size)
    crossover = _slerp(T, A, B)

    #New random members
    #new = np.random.normal(loc=0, scale=1, size=(foreign_size, latent_size))
    #rnd = np.random.RandomState(None)
    #new = rnd.randn(foreign_size, input_shape)
    #new_dlatents = Gs.components.mapping.run(new, None)

    #Mutate
    next_pop = np.concatenate((variables[selected], crossover))
    next_pop = _mutate(next_pop, noise, mutation)
    #next_pop = np.concatenate((next_pop, new))

    return next_pop

def crossover2(variables, foreign, noise, indices):

    mutation = 0.5

    selected = np.array(indices)
    pop_size, level, latent_size = variables.shape
    select_size = selected.shape[0]
    crossover_size = int(max(pop_size - select_size - foreign, 0))
    foreign_size = int(min(pop_size - select_size, foreign))

    #Crossover
    A = variables[randomChoice(selected, crossover_size)]
    B = variables[randomChoice(selected, crossover_size)]
    T = np.random.uniform(low=0, high=1, size=crossover_size)
    crossover = _slerp(T, A, B)

    #New random members
    new = np.random.normal(loc=0, scale=1, size=(foreign_size, latent_size))

    #Mutate
    next_pop = np.concatenate((variables[selected], crossover))
    next_pop = _mutate(next_pop, noise, mutation)
    next_pop = np.concatenate((next_pop, new))

    return next_pop

def crossover(variables, foreign, noise, indices):

    mutation = 0.5

    selected = np.array(indices)
    pop_size, latent_size = variables.shape
    select_size = selected.shape[0]
    crossover_size = int(max(pop_size - select_size - foreign, 0))
    foreign_size = int(min(pop_size - select_size, foreign))

    #Crossover
    A = variables[randomChoice(selected, crossover_size)]
    B = variables[randomChoice(selected, crossover_size)]
    T = np.random.uniform(low=0, high=1, size=crossover_size)
    crossover = _slerp(T, A, B)

    #New random members
    new = np.random.normal(loc=0, scale=1, size=(foreign_size, latent_size))

    #Mutate
    next_pop = np.concatenate((variables[selected], crossover))
    next_pop = _mutate(next_pop, noise, mutation)
    next_pop = np.concatenate((next_pop, new))

    return next_pop

def interpol(variables, steps, a, b):

    pt_a = variables[a]
    pt_b = variables[b]
    z = np.empty((steps, 512))
    #for i, alpha in enumerate(np.linspace(start=0.0, stop=1.0, num=steps)):
    #    z[i] = (1 - alpha) * pt_a + alpha * pt_b

    for i, alpha in enumerate(np.linspace(start=pt_a, stop=pt_b, num=steps)):
        z[i] =alpha

    return z

def interpol2(variables, steps, a, b):

    pt_a = variables[a]
    pt_b = variables[b]
    z = np.empty((steps, 18, 512))
    #for i, alpha in enumerate(np.linspace(start=0.0, stop=1.0, num=steps)):
    #    z[i] = (1 - alpha) * pt_a + alpha * pt_b

    for i, alpha in enumerate(np.linspace(start=pt_a, stop=pt_b, num=steps)):
        z[i] =alpha

    return z

def arit(variables):

    cont = True
    count = 0
    sum = np.zeros((variables.shape[0],2))
    k = np.zeros(variables.shape[1], dtype=float, order='C')
    while cont:

        op = indexbox(msg='Do you want to add or subtract this latent?', title=' ', choices=('Add', 'Subtract'), image=None, default_choice='Add', cancel_choice='Subtract')

        if op == 0:

            a = integerbox(msg = "Enter an index to add")
            k = k + variables[a]
            sum[count][0] = a
            sum[count][1] = 1

        if op == 1:
            a = integerbox(msg="Enter an index to subtract")
            k = k - variables[a]
            sum[count][0] = a
            sum[count][1] = -1

        count = count + 1
        cont = ynbox(msg='Add/subtract another image?', title=' ', choices=('[<F1>]Yes', '[<F2>]No'), image=None,
              default_choice='[<F1>]Yes', cancel_choice='[<F2>]No')

    z = np.empty((variables.shape[0], variables.shape[1]))

    for i in range(z.shape[0]):
        z[i] = k

    noise = random.random()
    mutation = 0.7

    z = _mutate(z, noise, mutation)
    z[0] = k

    return z, sum

def _slerp(t, a, b):
    unit_a = a / np.transpose([np.linalg.norm(a, 2, 1)])
    unit_b = b / np.transpose([np.linalg.norm(b, 2, 1)])

    omega = np.sum(unit_a*unit_b, 1)
    omega = np.arccos(np.clip(omega, -1, 1)) + 1e-8

    weighted_a = np.transpose(np.sin(omega*(1-t))/np.sin(omega)*np.transpose(a))
    weighted_b = np.transpose(np.sin(omega*t)/np.sin(omega)*np.transpose(b))
    return weighted_a + weighted_b

def _mutate(a, noise, prob):
    size = a.shape[0]
    mask = np.random.binomial(1, prob, size)
    t = -math.log(1-.9*noise, 10)*mask
    b = np.random.uniform(low=0, high=1, size=a.shape)
    return _slerp(t, a, b)

def randomChoice(tensor, n):
    return np.random.choice(tensor, n, replace=True)
