import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import Image

import clip
from tensorboardX import SummaryWriter

import torch
import argparse
import yaml
from tqdm import tqdm
from algorithm.CMA_ES import shallow_cma
from algorithm.LM_CMA_ES import Shallow_LMCMAES
from algorithm.MMES import Shallow_MMES
from algorithm.LMMAES import Shallow_LMMAES
from model.Shallow_Prompt_CLIP_Testtime import PromptCLIP_Shallow
import numpy as np
import time
import logging
from datetime import datetime
from dataset.general import load_train, load_test, load_fix

__classification__ = ["dtd", "caltech101", "eurosat", "ucf101", "flower102", "pets", "aircraft", "food101", "SUN397",
                      "cars",
                      "imagenet", "imagenet-a", "imagenet-r", "imagenet-sketch", "imagenet-v2"]
__pypop__ = ["shallow_lmcmaes", "shallow_mmes", "shallow_dcem", "shallow_maes"]
__dataset__ = "./data"
__output__ = "./result"
__backbone__ = "ViT-B/16"

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="dtd", type=str)
parser.add_argument("--opt", default="shallow_cma", type=str)
# parser.add_argument("--gpu", default='1', help='gpu id')
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')

args = parser.parse_args()

assert "shallow" in args.opt, "Only shallow prompt tuning is supported in this file."
cfg = yaml.load(open("./configs/shallow_prompt.yaml"), Loader=yaml.FullLoader)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f'log/{args.task_name}_{current_time}.txt'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

writer = SummaryWriter()

cfg["opt_name"] = args.opt
cfg["data_dir"] = __dataset__
cfg["output_dir"] = __output__
cfg["opt_name"] = args.opt
cfg["backbone"] = __backbone__
n_popsize = cfg["popsize"]

for k, v in cfg[args.task_name].items():
    cfg[k] = v
cfg["parallel"] = args.parallel

device = "cuda" if torch.cuda.is_available() else "cpu"
intrinsic_dim_L = cfg["intrinsic_dim_L"]
intrinsic_dim_V = cfg["intrinsic_dim_V"]



# Eval function and Settings(if needed)+
def fitness_eval(prompt_zip):
    prompt_text_list = prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in [prompt_zip]])
    prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in [prompt_zip]])
    fitnesses = [prompt_clip.eval(x).item() for x in zip(prompt_text_list, prompt_image_list)]

    if prompt_clip.num_call % (prompt_clip.test_every) == 0:
        print("-------------------------Epoch {}---------------------------".format(
            prompt_clip.num_call / prompt_clip.test_every))
    if prompt_clip.num_call % (prompt_clip.popsize) == 0:
        print("Evaluation of Individual: {}, Generation: {}".format(prompt_clip.num_call % prompt_clip.popsize,
                                                                    int(prompt_clip.num_call / prompt_clip.popsize)))
    if prompt_clip.num_call % prompt_clip.test_every == 0:
        print("current loss: {}".format(prompt_clip.min_loss))
        print("Best Prompt Embedding - Acc : " + str(prompt_clip.best_accuracy))
        print("fitness_eval")
        # logging.info("current loss: {}".format(prompt_clip.min_loss))
        # logging.info("Best Prompt Embedding - Acc : " + str(prompt_clip.best_accuracy))
        # logging.info("fitness_eval")
    return fitnesses[0]


ndim_problem = intrinsic_dim_L + intrinsic_dim_V
pro = {'fitness_function': fitness_eval,
       'ndim_problem': ndim_problem}
opt_cfg = {'fitness_threshold': 1e-10,
           'seed_rng': 0,
           'max_runtime': 20800,
           'x': 0 * np.ones((ndim_problem,)),  # mean
           'sigma': cfg['sigma'],
           'verbose_frequency': 5,
           'n_individuals': cfg["popsize"],
           'is_restart': False}

# Load algorithm
opt = None
if args.opt == "shallow_cma":
    opt = shallow_cma(cfg)
elif args.opt == "shallow_lmcmaes":
    opt = Shallow_LMCMAES(pro, opt_cfg)
elif args.opt == "shallow_mmes":
    opt = Shallow_MMES(pro, opt_cfg)
elif args.opt == "shallow_lmmaes":
    opt = Shallow_LMMAES(pro, opt_cfg)

# Build CLIP model
# if args.task_name in __classification__:
#     prompt_clip = PromptCLIP_Shallow(args.task_name, cfg)
print('Population Size: {}'.format(cfg["popsize"]))

# Black-box prompt tuning

if args.opt in __pypop__:
    if args.task_name in __classification__:
        prompt_clip = PromptCLIP_Shallow(args.task_name, cfg)
        text_context = prompt_clip.get_text_information()
        image_context = prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)
        res = opt.optimize()
else:
    if args.task_name in __classification__:

        # 加载数据集
        clip_model, preprocess = clip.load(cfg["backbone"], device="cuda")
        # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
        #                                                 root=cfg["data_dir"], dataset_dir="caltech101_Gen")
        if args.task_name == 'eurosat':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="eurosat_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="eurosat_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'caltech101':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="caltech101_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="caltech101_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'dtd':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="dtd_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="dtd_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'ucf101':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="ucf101_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="ucf101_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'flower102':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="flower102_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="flower102_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'pets':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="pets_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="pets_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'food101':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="food101_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="food101_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'aircraft':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="aircraft_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="aircraft_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'SUN397':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="SUN397_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="SUN397_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'cars':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="cars_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="cars_Gen")
            # classes = train_data.classes
            classes = test_data.classes
            # print(classes)
            # print("*********")
            # print(cla)
            n_cls = len(classes)
        elif args.task_name == 'imagenet':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="imagenet-r_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="imagenet_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'imagenet-a':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="imagenet-a_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="imagenet-a_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'imagenet-r':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="imagenet-r_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="imagenet-r_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'imagenet-sketch':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="imagenet-r_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="imagenet-sketch_Gen")
            classes = test_data.classes
            n_cls = len(classes)
        elif args.task_name == 'imagenet-v2':
            # train_data, train_loader = load_train(batch_size=cfg["batch_size"], preprocess=preprocess,
            #                                       root=cfg["data_dir"], dataset_dir="imagenet-r_Gen")
            test_data, test_loader = load_test(batch_size=cfg["batch_size"], preprocess=preprocess,
                                               root=cfg["data_dir"], dataset_dir="imagenet-v2_Gen")
            classes = test_data.classes
            n_cls = len(classes)

        # aaaaa = len(test_loader)
        logging.info('The number of batch: %d', len(test_loader))
        # print(len(test_loader))

        # 计算最大熵值
        # if n_cls <= 1:
        #     max_entropy = 0
        # p = 1 / n_cls
        # max_entropy = -n_cls * p * math.log(p)
        # max_entropy = round(max_entropy, 4)

        prompt_clip = PromptCLIP_Shallow(args.task_name, cfg, classes, n_cls)
        text_context = prompt_clip.get_text_information()
        image_context = prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)



        i = 0
        step = 0
        # best_prompt_text_batch = []
        accuracy = []
        loss = []
        # best_prompt_text_all = torch.tensor([]).to("cuda")
        # json_path = "/home/fameng/project/BPT-VLM/data/caltech101_Gen/split.json"
        # folder_path = "/home/fameng/project/BPT-VLM/data/caltech101_Gen/image_data/"
        # output_folder_path = "/home/fameng/project/BPT-VLM/heatmaps/"
        # batch_size = 32  # 根据需求调整批次大小
        # 从 JSON 文件中加载测试图像路径
        # test_image_paths = draw.load_test_images_from_json(json_path, folder_path)

        for batch in test_loader:
            r = 4
            i += 1
            step += 1
            acc_batch = []
            fit_batch = []
            best_accuracy = 0

            for ii in range(r):

                acc_iter = []
                fitnesses = []

                solutions = opt.ask()
                prompt_text_list = prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions])
                prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in solutions])

                result = [prompt_clip.eval(x, batch, ii, r) for x in tqdm(zip(prompt_text_list, prompt_image_list))]

                for f in range(int(n_popsize)):
                    fitnesses.append(result[f][0])
                    acc_iter.append(result[f][1])

                    # 更新最佳准确率及其对应的prompt
                    # if result[f][1] > best_accuracy:
                    #     best_accuracy = result[f][1]
                    #     best_prompt_text = prompt_text_list[f]
                        # best_prompt_image = prompt_image_list[f]
                    # print()
                acc_batch.append(max(acc_iter))
                fit_batch.append(fitnesses[acc_iter.index(max(acc_iter))])

                opt.tell(solutions, fitnesses)
                print(i)
                # print(acc_iter)
            # 绘图


            # print(acc_batch)
            # best_prompt_text_all = best_prompt_text.clone()
            # best_prompt_text_all = torch.cat((best_prompt_text_all, best_prompt_text), 0)
            accuracy.append(max(acc_batch))
            loss.append(fit_batch[acc_batch.index(max(acc_batch))])
            writer.add_scalar('CE_loss', loss[step-1], global_step=step)

            # best_prompt_text_batch.append(best_prompt_text)
            # print("Best Prompt Text:", best_prompt_text_all.size())
            # 重新初始化
            # opt = shallow_cma(cfg)
            # writer.add_scalar('CE_loss', fix_loss, global_step=step)


        # torch.save(best_prompt_text_all, "/home/fameng/project/BPT-VLM/best_prompt.pt")
        sum = 0
        for acc_all in accuracy:
            sum += acc_all

        Ava = sum / len(test_loader)
        # print(accuracy) # list
        # print(Ava)         # float
        print("*****************")
        logging.info('Best acc of every batch: %s', accuracy)
        logging.info("Avaerage acc:  %.4f", Ava)




    # else:
    # image_context = prompt_clip.get_image_information()
    # prompt_clip.image_encoder.set_context(image_context)
    # while not opt.stop():
    #     solutions = opt.ask()
    #     prompt_text_list = prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions])
    #     prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in solutions])
    #     if cfg["parallel"]:
    #         fitnesses = prompt_clip.eval([prompt_text_list, prompt_image_list])
    #         fitnesses = [x.item() for x in tqdm(fitnesses, ncols=50)]
    #     else:
    #         fitnesses = [prompt_clip.eval(x).item() for x in tqdm(zip(prompt_text_list, prompt_image_list))]
    #     # output current loss and acc
    #     if prompt_clip.num_call % prompt_clip.test_every == 0:
    #         # print("current loss: {}".format(prompt_clip.min_loss))
    #         # print("Best Prompt Embedding - Acc : " + str(prompt_clip.best_accuracy))
    #         logging.info("current loss: {}".format(prompt_clip.min_loss))
    #         logging.info("Best Prompt Embedding - Acc : " + str(prompt_clip.best_accuracy))
    #     opt.tell(solutions, fitnesses)

# acc = prompt_clip.test()
pass
