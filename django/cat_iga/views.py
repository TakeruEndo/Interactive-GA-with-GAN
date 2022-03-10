import sys

sys.path.append(
    "/Users/endotakeru/Documents/Interactive-GA-with-GAN/stylegan2-ada-pytorch")
sys.path.append(
    "/Users/endotakeru/Documents/Interactive-GA-with-GAN/Interactive-StyleGAN")

import itertools
import operators as op
import generate_fn
import sys
from glob import glob
import random

import torch
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.urls import reverse



def str_to_list(char):
    l = char.split(',')
    l = [int(i) for i in l]
    l = list(set(l))
    return l


def crossfusion(latent_vector, select_number):
    """遺伝子の交配
    REFERENCE: https://github.com/cto300/Interactive-StyleGAN
    """
    latent_vector = latent_vector.to('cpu').detach().numpy().copy()
    new_gen = op.crossover(latent_vector, foreign=0,
                           noise=0, indices=select_number)
    return torch.from_numpy(new_gen)


def crossfusion_simple(latent_vector, select_number):
    """シンプルな遺伝子の交配
    """
    new_vectors = []
    latent_vector = latent_vector.to('cpu').detach().numpy().copy()
    # 1. 全体の平均を取る
    new_vectors.append(np.mean(latent_vector[select_number], axis=0))
    # 2. ペアを交配
    all_pair = itertools.combinations(select_number, 2)
    for pair in all_pair:
        a = 0.7
        b = 0.3
        alpha = (b - a) * np.random.rand() + a
        new_vector = alpha * \
            latent_vector[pair[0]] * (1 - alpha) * latent_vector[pair[1]]
        new_vectors.append(np.mean(latent_vector[list(pair)], axis=0))
    print(len(new_vectors))
    # 3. 突然変異
    """
    1. 何桁を変更するかをランダムに選択
    2. 何桁目をランダムに変更するかを選択
    3. 実行
    """
    temp_list = []
    for new_vector in new_vectors:
        # 20以内で変異させる個数を選択
        num = np.random.randint(0, 20)
        # 変異させる場所を選択
        indexes = np.random.randint(0, 512, num)
        for index in indexes:
            new_vector[index] = np.random.randn()
        temp_list.append(new_vector)
    random.shuffle(temp_list)
    new_vectors = temp_list
    # 4. 足りなかったらランダム生成
    if len(new_vectors) <= 20:
        for i in range(20 - len(new_vectors)):
            new_vectors.append(np.random.randn(512))
        new_vectors = np.array(new_vectors)
        return torch.from_numpy(new_vectors)
    else:
        new_vectors = np.array(new_vectors)
        return torch.from_numpy(new_vectors[:20])


def multiplication(latent_vactor):
    new_vectors = []
    for i in range(20):
        new_vector = latent_vactor.copy()
        # 20以内で変異させる個数を選択
        num = np.random.randint(0, 50)
        # 変異させる場所を選択
        indexes = np.random.randint(0, 512, num)
        for index in indexes:
            new_vector[index] = np.random.randn()
        new_vectors.append(new_vector)
    return torch.from_numpy(new_vectors)


def get_image_paths(generation: int):
    img_paths = glob(
        f"/Users/endotakeru/Documents/Interactive-GA-with-GAN/django/cat_iga/static/images/gen{generation}/*.png")
    img_paths = [i.replace(
        '/Users/endotakeru/Documents/Interactive-GA-with-GAN/django/cat_iga', '') for i in img_paths]
    img_paths = sorted(img_paths)
    return img_paths


def next_generation(request, new_generation):
    pre_generation = new_generation
    new_generation = new_generation + 1
    select_number = str_to_list(request.POST.get('name'))
    latent_vector = torch.load(
        f"/Users/endotakeru/Documents/Interactive-GA-with-GAN/django/cat_iga/static/images/gen{pre_generation}/gen.pt")
    if len(latent_vector) == 1:
        latent_vector = multiplication(latent_vector)
    latent_vector = crossfusion(latent_vector, select_number)
    generate_fn.main(
        f"/Users/endotakeru/Documents/Interactive-GA-with-GAN/django/cat_iga/static/images/gen{new_generation}", latent_vector=latent_vector)
    img_paths = get_image_paths(new_generation)
    context = {'img_path_list': img_paths, 'generation': new_generation}
    return render(request, 'cat_iga/index.html', context)


def finish(request):
    return HttpResponse('終了')


def re_create(request):
    """作り直し
    第一世代にしか適用できない
    """
    generate_fn.main(
        "/Users/endotakeru/Documents/Interactive-GA-with-GAN/django/cat_iga/static/images/gen1")
    img_paths = get_image_paths(1)
    context = {'img_path_list': img_paths, 'generation': 1}
    return render(request, 'cat_iga/index.html', context)


def index(request):
    img_paths = get_image_paths(1)
    context = {'img_path_list': img_paths, 'generation': 1}
    return render(request, 'cat_iga/index.html', context)
