import numpy as np
import operators as op

import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

import os
import shutil

import dnnlib
import dnnlib.tflib as tflib


def init_deepie(url, test, rand=True, seeds=None):
    # def init_deepie(generator, test, rand = True, seeds = None):

    tf.InteractiveSession()

    with dnnlib.util.open_url(url) as f:
        _G, _D, Gs = pickle.load(f)

    if (rand == False):
        seeds = [639, 701, 687, 615, 2268, 444, 555, 666, 789, 1092,
                 6745, 5653, 4543, 435, 4345, 45322, 563, 785, 543, 6785]
        latents = np.stack(np.random.RandomState(
            seed).randn(Gs.input_shape[1]) for seed in seeds)

    else:
        # initialice random latents
        rnd = np.random.RandomState(None)
        latents = rnd.randn(20, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    # Run the generator to produce a set of images.
    l1 = latents[:len(latents)//2]
    l2 = latents[len(latents)//2:]
    images1 = Gs.run(l1, None, truncation_psi=0.7,
                     randomize_noise=True, output_transform=fmt)
    images2 = Gs.run(l2, None, truncation_psi=0.7,
                     randomize_noise=True, output_transform=fmt)
    #images1 = generator.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    #images2 = generator.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    images = np.concatenate([images1, images2])

    if not os.path.exists("Deepie"):
        os.mkdir("Deepie")

    if not os.path.exists("Deepie/Test%d" % test):
        os.mkdir("Deepie/Test%d" % test)

    if not os.path.exists("Deepie/Test%d/selected" % test):
        os.mkdir("Deepie/Test%d/selected" % test)

    if not os.path.exists("Deepie/Test%d/Generation0" % test):
        os.mkdir("Deepie/Test%d/Generation0" % test)

    for idx in range(images.shape[0]):
        # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
        PIL.Image.fromarray(images[idx], 'RGB').save(
            'Deepie/Test%d/Generation0/img%d.png' % (test, idx))

    return latents


def cross_deepie(url, test, indices, variables, foreign, noise, count):
    # def cross_deepie(generator, test, indices, variables, foreign, noise, count):

    tf.InteractiveSession()

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    with dnnlib.util.open_url(url) as f:
        _G, _D, Gs = pickle.load(f)

    next_pop = op.crossover(variables, foreign, noise, indices)

    l1 = next_pop[:len(next_pop) // 2]
    l2 = next_pop[len(next_pop) // 2:]
    images1 = Gs.run(l1, None, truncation_psi=0.7,
                     randomize_noise=True, output_transform=fmt)
    images2 = Gs.run(l2, None, truncation_psi=0.7,
                     randomize_noise=True, output_transform=fmt)
    #images1 = generator.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    #images2 = generator.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    images = np.concatenate([images1, images2])

    # save images
    if not os.path.exists("Deepie/Test%d/Generation%d" % (test, count)):
        os.mkdir("Deepie/Test%d/Generation%d" % (test, count))

    for idx in range(images.shape[0]):
        # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
        PIL.Image.fromarray(images[idx], 'RGB').save(
            'Deepie/Test%d/Generation%d/img%d.png' % (test, count, idx))

    return next_pop


def show_images(n, gen, test):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np
    from PIL import Image

    im_arr = [None] * n

    for idx in range(n):
        im = Image.open(r"Deepie/Test%d/Generation%d/img%d.png" %
                        (test, gen, idx))
        #im_arr[idx] = im
        im_arr[idx] = im.resize((256, 256))

    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(4, 5),  #
                     axes_pad=0.5,  # pad between axes in inch.
                     )

    for idx, (ax, im) in enumerate(zip(grid, im_arr)):
     # Iterating over the grid returns the Axes.
        ax.set_title("img%d.png" % idx)
        ax.imshow(im)

    plt.show()


def show_images2(n, gen, test):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np
    from PIL import Image

    im_arr = [None] * n

    for idx in range(n):
        im = Image.open(r"DeepSIE/Test%d/Generation%d/img%d.png" %
                        (test, gen, idx))
        #im_arr[idx] = im
        im_arr[idx] = im.resize((256, 256))

    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(4, 5),  #
                     axes_pad=0.5,  # pad between axes in inch.
                     )

    for idx, (ax, im) in enumerate(zip(grid, im_arr)):
     # Iterating over the grid returns the Axes.
        ax.set_title("img%d.png" % idx)
        ax.imshow(im)

    plt.show()


def deepie2(url, test, gen, latents, indices, foreign, noise):

    tf.InteractiveSession()

    with dnnlib.util.open_url(url) as f:
        _G, _D, Gs = pickle.load(f)

     # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    if gen == 0:

        #next_pop = None
        #variables = latents.copy()

        # Run the generator to produce a set of images.
        #l1 = variables[:len(variables)//2]
        #l2 = variables[len(variables)//2:]
        l1 = latents[:len(latents)//2]
        l2 = latents[len(latents)//2:]
        images1 = Gs.run(l1, None, truncation_psi=0.7,
                         randomize_noise=True, output_transform=fmt)
        images2 = Gs.run(l2, None, truncation_psi=0.7,
                         randomize_noise=True, output_transform=fmt)
        images = np.concatenate([images1, images2])

        if not os.path.exists("Deepie"):
            os.mkdir("Deepie")

        if not os.path.exists("Deepie/Test%d" % test):
            os.mkdir("Deepie/Test%d" % test)

        if not os.path.exists("Deepie/Test%d/selected" % test):
            os.mkdir("Deepie/Test%d/selected" % test)

        if not os.path.exists("Deepie/Test%d/Generation%d" % (test, gen)):
            os.mkdir("Deepie/Test%d/Generation%d" % (test, gen))

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save(
                'Deepie/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        show_images(images.shape[0], gen, test)

        gen = gen + 1

        return latents, gen

    else:

        next_pop = op.crossover(latents, foreign, noise, indices)

        # save selected images
        for idx in indices:
            shutil.copy("Deepie/Test%d/Generation%d/img%d.png" %
                        (test, gen - 1, idx), "Deepie/Test%d/selected" % test)
            old_file = os.path.join("Deepie/Test%d/selected" %
                                    (test), "img%d.png" % (idx))
            new_file = os.path.join("Deepie/Test%d/selected" %
                                    (test), "img%d_gen%d.png" % (idx, gen - 1))
            os.rename(old_file, new_file)

        l1 = next_pop[:len(next_pop) // 2]
        l2 = next_pop[len(next_pop) // 2:]
        images1 = Gs.run(l1, None, truncation_psi=0.7,
                         randomize_noise=True, output_transform=fmt)
        images2 = Gs.run(l2, None, truncation_psi=0.7,
                         randomize_noise=True, output_transform=fmt)
        images = np.concatenate([images1, images2])

        # save images
        if not os.path.exists("Deepie/Test%d/Generation%d" % (test, gen)):
            os.mkdir("Deepie/Test%d/Generation%d" % (test, gen))

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save(
                'Deepie/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        show_images(images.shape[0], gen, test)

        gen = gen + 1

        return next_pop, gen


def deepie(generator, test, gen, latents, indices, foreign, noise):

     # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    if gen == 0:

        #next_pop = None
        #variables = latents.copy()

        # Run the generator to produce a set of images.
        #l1 = variables[:len(variables)//2]
        #l2 = variables[len(variables)//2:]
        l1 = latents[:len(latents)//2]
        l2 = latents[len(latents)//2:]
        images1 = generator.run(
            l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images2 = generator.run(
            l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = np.concatenate([images1, images2])

        if not os.path.exists("Deepie"):
            os.mkdir("Deepie")

        if not os.path.exists("Deepie/Test%d" % test):
            os.mkdir("Deepie/Test%d" % test)

        if not os.path.exists("Deepie/Test%d/selected_A" % test):
            os.mkdir("Deepie/Test%d/selected_A" % test)

        if not os.path.exists("Deepie/Test%d/Generation%d" % (test, gen)):
            os.mkdir("Deepie/Test%d/Generation%d" % (test, gen))

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save(
                'Deepie/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        show_images(images.shape[0], gen, test)

        gen = gen + 1

        return latents, gen

    else:

        next_pop = op.crossover(latents, foreign, noise, indices)

        # save selected images
        for idx in indices:
            shutil.copy("Deepie/Test%d/Generation%d/img%d.png" %
                        (test, gen - 1, idx), "Deepie/Test%d/selected_A" % test)
            old_file = os.path.join("Deepie/Test%d/selected_A" %
                                    (test), "img%d.png" % (idx))
            new_file = os.path.join("Deepie/Test%d/selected_A" %
                                    (test), "gen%d_img%d.png" % (gen - 1, idx))
            os.rename(old_file, new_file)

        l1 = next_pop[:len(next_pop) // 2]
        l2 = next_pop[len(next_pop) // 2:]
        images1 = generator.run(
            l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images2 = generator.run(
            l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = np.concatenate([images1, images2])

        # save images
        if not os.path.exists("Deepie/Test%d/Generation%d" % (test, gen)):
            os.mkdir("Deepie/Test%d/Generation%d" % (test, gen))

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save(
                'Deepie/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        if gen != 0:

            names = np.array(
                ['Test', 'Generation', 'Choosen', "foreigns", "mutation"])
            values = np.array([test, gen, len(indices), foreign, noise])

            ab = np.zeros(names.size, dtype=[('var1', 'U6'), ('var2', float)])
            ab['var1'] = names
            ab['var2'] = values

            file_name = 'Deepie/Test%d/selected_A/log%d.csv' % (test, gen)

            #np.savetxt(file_name, ab, fmt="%10s %10.3f")
            np.savetxt(file_name, ab, fmt='%s', delimiter=',')

        show_images(images.shape[0], gen, test)

        gen = gen + 1

        return next_pop, gen


def deepsie(generator, test, gen, latents, indices, foreign, noise):

     # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    synthesis_kwargs = dict(output_transform=dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

    if gen == 0:

        images = generator.components.synthesis.run(
            latents, randomize_noise=False, **synthesis_kwargs)

        if not os.path.exists("DeepSIE"):
            os.mkdir("DeepSIE")

        if not os.path.exists("DeepSIE/Test%d" % test):
            os.mkdir("DeepSIE/Test%d" % test)

        if not os.path.exists("DeepSIE/Test%d/selected_B" % test):
            os.mkdir("DeepSIE/Test%d/selected_B" % test)

        if not os.path.exists("DeepSIE/Test%d/Generation%d" % (test, gen)):
            os.mkdir("DeepSIE/Test%d/Generation%d" % (test, gen))

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save(
                'DeepSIE/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        show_images2(images.shape[0], gen, test)

        gen = gen + 1

        return latents, gen

    else:

        style_levels = [range(0, 4), range(4, 8), range(8, 18), range(0, 18)]

        variables = latents

        leveled_cross = [None] * len(style_levels)

        new_variables = np.zeros_like(variables)

        for k, levels in enumerate(style_levels):

            new_variables = np.zeros_like(variables)

            tmp_variables = variables.copy()

            new_variables[range(0, 20)] = tmp_variables[np.random.choice(
                np.array(indices), 20, replace=True)]

            temp = np.zeros((tmp_variables.shape[0], tmp_variables.shape[2]))

            for j in levels:

                for i in range(tmp_variables.shape[0]):
                    temp[i] = tmp_variables[i][j]

                crosst = op.crossover3(temp, noise, indices)

                for i in range(tmp_variables.shape[0]):
                    new_variables[i][j] = crosst[i]

            leveled_cross[k] = new_variables

        next_pop = np.zeros_like(variables)
        next_pop2 = np.zeros_like(variables)

        # print("leveled_cross.shape")
        # print(leveled_cross.shape)
        #cross1 = np.random.choice(leveled_cross[0], 6, replace=False)
        #cross2 = np.random.choice(leveled_cross[1], 6, replace=False)
        #cross3 = np.random.choice(leveled_cross[2], 8, replace=False)

        #next_pop[range(0,5)] = cross1[range(0,5)]
        #next_pop[range(6,11)] = cross1[range(0,5)]
        #next_pop[range(12,19)] = cross1[range(0,5)]

        cross1 = leveled_cross[0]

        cross2 = leveled_cross[1]
        cross3 = leveled_cross[2]
        cross4 = leveled_cross[3]

        #next_pop[range(0,19)] = cross1[range(0,19)]

        next_pop2[range(0, 5)] = cross1[np.random.choice(20, 5, replace=True)]
        next_pop2[range(5, 10)] = cross2[np.random.choice(20, 5, replace=True)]
        next_pop2[range(10, 15)] = cross3[np.random.choice(
            20, 5, replace=True)]
        next_pop2[range(15, 20)] = cross4[np.random.choice(
            20, 5, replace=True)]

        #next_pop[range(0,5)] = leveled_cross[0][np.random.choice(np.arange(20), 6, replace=False)]
        #next_pop[range(6,11)] = leveled_cross[1][np.random.choice(np.arange(20),6, replace=False)]
        #next_pop[range(12,19)] = leveled_cross[2][np.random.choice(np.arange(20), 8, replace=False)]

        np.random.shuffle(next_pop2)

        next_pop = next_pop2

        for i, idx in enumerate(indices):
            next_pop[i] = variables[idx]

        if(foreign != 0):
            rnd = np.random.RandomState(None)
            foreing_latents = rnd.randn(foreign, generator.input_shape[1])
            foreing_dlatents = generator.components.mapping.run(
                foreing_latents, None)

            ccc = 0
            for i in range(next_pop.shape[0] - 1, next_pop.shape[0] - 1 - foreign, -1):
                next_pop[i] = foreing_dlatents[ccc]
                ccc = ccc + 1

        # save selected images
        for idx in indices:
            shutil.copy("DeepSIE/Test%d/Generation%d/img%d.png" %
                        (test, gen - 1, idx), "DeepSIE/Test%d/selected_B" % test)
            old_file = os.path.join(
                "DeepSIE/Test%d/selected_B" % (test), "img%d.png" % (idx))
            new_file = os.path.join(
                "DeepSIE/Test%d/selected_B" % (test), "gen%d_img%d.png" % (gen - 1, idx))
            os.rename(old_file, new_file)

        images = generator.components.synthesis.run(
            next_pop, randomize_noise=False, **synthesis_kwargs)

        # save images
        if not os.path.exists("DeepSIE/Test%d/Generation%d" % (test, gen)):
            os.mkdir("DeepSIE/Test%d/Generation%d" % (test, gen))

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save(
                'DeepSIE/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        if (gen != 0):

            names = np.array(
                ['Test', 'Generation', 'Choosen', "foreigns", "mutation"])
            values = np.array([test, gen, len(indices), foreign, noise])

            ab = np.zeros(names.size, dtype=[('var1', 'U6'), ('var2', float)])
            ab['var1'] = names
            ab['var2'] = values

            file_name = 'DeepSIE/Test%d/selected_B/log%d.csv' % (test, gen)

            #np.savetxt(file_name, ab, fmt="%10s %10.3f")
            np.savetxt(file_name, ab, fmt='%s', delimiter=',')

            #NAMES  = np.array(['Test', 'Generation', 'Choosen', "foreigns", "mutation" ])
            #FLOATS = np.array([ test    , gen     , len(indices), foreigns, noise     ])

            #DAT =  np.column_stack((NAMES, FLOATS))

            #np.savetxt('test.txt', DAT, delimiter=" ")

            # write log
            #file_name = 'DeepSIE/Test%d/Generation%d/log%d.txt' % (test, gen, gen)
            # with open(file_name, 'w+') as file:

            #   file.write('Test' + str(test) + "\n")
            #   file.write(('Generation%d' % gen) + "\n")

            #   file.write("Choosen images:" + str(indices) + "\n")
            #   file.write("Number of foreigns:" + str(foreign) + "\n")
            #   file.write("Mutation std:" + str(noise) + "\n")

            #   file.flush()
            #   file.close()
            # file.flush()

        show_images2(images.shape[0], gen, test)

        gen = gen + 1

        return next_pop, gen
