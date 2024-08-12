import os
import torch
from PIL.Image import Image
from torch.nn import functional as F
import numpy as np
import clip
import matplotlib
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import CIFAR100, ImageFolder
from dataset.cifar100 import load_train_cifar100, load_test_cifar100
from model.shallow_encoder import TextEncoder, VisionEncoder, VisionEncoder_Clip
from model.analysis_utils import Analysis_Util
from dataset.general import load_train, load_test, load_fix
from torchvision import transforms
from randaugment import RandAugmentMC
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
import time
from sklearn import manifold
from tensorboardX import SummaryWriter
# import draw


class PromptCLIP_Shallow:
    def __init__(self, task_name, cfg, classes, n_cls):
        self.task_name = task_name
        self.opt_name = cfg["opt_name"]
        self.data_dir = cfg["data_dir"]
        self.output_dir = cfg["output_dir"]
        self.backbone = cfg["backbone"]
        self.popsize = cfg["popsize"]
        self.parallel = cfg["parallel"]
        self.batch_size = cfg["batch_size"]
        self.classes = classes
        self.n_cls = n_cls
        # self.k_shot = cfg["k_shot"]
        # self.writer = SummaryWriter()
        self.seed = cfg["seed"]
        self.num_call = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ###########
        # self.model =
        # self.preprocess =
        ###########
        self.model, self.preprocess = clip.load(self.backbone, device=self.device)
        # self.load_dataset()
        self.loss = []
        self.acc = []
        # Text Encoder
        self.n_prompt_tokens_L = cfg["n_prompt_tokens_L"]
        self.intrinsic_dim_L = cfg["intrinsic_dim_L"]
        self.ctx_dim_L = self.model.ln_final.weight.shape[0]
        self.text_encoder = TextEncoder(self.model)

        # Image Encoder
        self.n_prompt_tokens_V = cfg["n_prompt_tokens_V"]
        self.ctx_dim_V = self.model.visual.width
        self.intrinsic_dim_V = cfg["intrinsic_dim_V"]
        self.image_encoder = VisionEncoder(self.model)
        self.image_encoder_clip = VisionEncoder_Clip(self.model)
        self.image_encoder.n_prompt_tokens_V = self.n_prompt_tokens_V

        self.loss_type = cfg["loss_type"]
        self.init_prompt = None
        self.imsize = self.image_encoder.input_resolution
        self.logit_scale = self.model.logit_scale
        self.dtype = self.model.dtype
        self.best_prompt_text = None
        self.best_prompt_image = None
        self.best_accuracy = 0
        self.min_loss = None
        self.pl_acc = None
        self.loss = []
        self.test_every = cfg["test_every"] if self.parallel else cfg["test_every"] * self.popsize
        self.sigma = cfg["sigma"]
        # Lauguage Linear Layer
        self.linear_L = torch.nn.Linear(self.intrinsic_dim_L, self.n_prompt_tokens_L * self.ctx_dim_L,
                                        bias=False, device=self.device, dtype=self.dtype)
        embedding = self.model.token_embedding.weight.cpu()
        mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
        mu = 0.0
        std = std_hat / (np.sqrt(self.intrinsic_dim_L) * self.sigma)
        print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_L.parameters():
            torch.nn.init.normal_(p, mu, std)
        # Vision Linear Layer
        self.linear_V = torch.nn.Linear(self.intrinsic_dim_V, self.n_prompt_tokens_V * self.ctx_dim_V,
                                        bias=False, device=self.device, dtype=self.dtype)
        conv = self.model.visual.conv1.weight.cpu()
        mu_hat = np.mean(conv.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(conv.reshape(-1).detach().cpu().numpy())
        # mu = 0.0
        mu = mu_hat * 3072 / self.intrinsic_dim_V
        std = std_hat * np.sqrt(3072 / self.intrinsic_dim_V) * self.sigma
        print('[Conv] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_V.parameters():
            torch.nn.init.normal_(p, mu, std)

    def get_text_information(self, caption=None):
        # classification task - caption - None
        # refcoco ask - caption - str
        prompt_prefix = " ".join(["X"] * self.n_prompt_tokens_L)
        if caption is None:
            classnames = [name.replace("_", " ").replace("-", " ") for name in self.classes]
            pattern_prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_pattern_prompts = torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls": self.n_cls, "n_prompt_tokens_L": self.n_prompt_tokens_L,
                       "init_pattern_embedding": init_pattern_embedding,
                       "tokenized_pattern_prompts": tokenized_pattern_prompts,
                       "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        else:
            pattern_prompt = prompt_prefix + caption + "."
            tokenized_pattern_prompts = torch.cat([clip.tokenize(pattern_prompt)]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls": 1, "n_prompt_tokens_L": self.n_prompt_tokens_L,
                       "init_pattern_embedding": init_pattern_embedding,
                       "tokenized_pattern_prompts": tokenized_pattern_prompts, "batch_size": self.batch_size,
                       "pop_size": self.popsize, "parallel": self.parallel}
        # print(context)
        return context

    def get_image_information(self):
        context = {"n_prompt_tokens_V": self.n_prompt_tokens_V,
                   "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        return context

    def generate_text_prompts(self, intrinsic_vectors):
        prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector, device=self.device, dtype=self.dtype)
            # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
            z = self.linear_L(z).reshape(self.n_prompt_tokens_L, -1)
            if self.init_prompt is not None:
                z = z + self.init_prompt  # Az + p_0

            prompt_list.append(z)
        return prompt_list

    def generate_visual_prompts(self, intrinsic_vectors):
        ## intrinsic_vetors list 30: [500]
        visual_prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector, device=self.device, dtype=self.dtype)  ## z 500
            # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
            z = self.linear_V(z).reshape(self.n_prompt_tokens_V, -1)  ## z: 5,768
            # z = z + self.position_V
            visual_prompt_list.append(z)

        return visual_prompt_list

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size_m = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size_m).cuda()
        else:
            index = torch.randperm(batch_size_m)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def Entropy(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def select_predictions(self, out1, out2):
        # 计算每个样本的熵值
        entropy1 = self.Entropy(out1)
        entropy2 = self.Entropy(out2)
        # print("e1:",entropy1)
        # print("e2", entropy2)
        # 预测类别
        out1 = torch.argmax(out1, dim=1)
        out2 = torch.argmax(out2, dim=1)

        pseudolabels = torch.where(out1 == out2, out1,
                                   torch.where(entropy1 <= entropy2, out1, out2))
        # if (entropy1 == entropy2):
        #     print("*********")

        return pseudolabels, entropy1, entropy2

    def select_predictions_score(self, out1, out2):
        # 计算每个样本的熵值
        weight1, _ = torch.max(out1, 1)
        weight2, __ = torch.max(out2, 1)

        # print("e1:",entropy1)
        # print("e2", entropy2)
        # 预测类别
        label1 = torch.argmax(out1, dim=1)
        label2 = torch.argmax(out2, dim=1)

        pseudolabels = torch.where(weight1 == weight2, label1,
                                   torch.where(weight1 >= weight2, label1, label2))
        # if (entropy1 == entropy2):
        #     print("*********")

        return pseudolabels, weight1, weight2

    def metric(self, logits, label, w=None):
        if w is None:
            ce_loss = F.cross_entropy(logits, label, reduction='none')
        else:
            ce_loss = torch.mul(F.cross_entropy(logits, label, reduction='none'), w)

        final_loss = 0
        if self.loss_type == "ce":
            final_loss = torch.sum(ce_loss)
        elif self.loss_type == "focal":
            gamma = 2
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** gamma * ce_loss
            final_loss = torch.sum(focal_loss)
        return final_loss

    @torch.no_grad()
    def eval(self, prompt_zip, batch, ii, r):
        prompt_text, prompt_image = prompt_zip[0], prompt_zip[1]
        self.num_call += 1
        loss = 0
        fix_loss = 0
        # self.test_list()
        # self.test_pseudo()

        ## zero-shot clip
        temp_p = "A photo of a {}."
        prompts_p = [temp_p.format(c.replace("_", " ")) for c in self.classes]
        prompts_p = torch.cat([clip.tokenize(p) for p in prompts_p])
        prompts_p = prompts_p.to(self.device)

        with torch.no_grad():
            text_features_p = self.model.encode_text(prompts_p)
            text_features_p = text_features_p / text_features_p.norm(dim=-1, keepdim=True)

        text_features = self.text_encoder(prompt_text)  # if parallel, text_features.shape = [n_cls * popsize, *, *]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print(prompts_p.dtype)
        # print(image.dtype)

        # start = time.time()
        image, label = self.parse_batch(batch)
        ### zero-shot pseudo label ###

        # image.type(self.dtype)
        # print(image.dtype)
        image_features_p = self.image_encoder_clip(image)
        # print(image_features_p.dtype)

        image_features_p = image_features_p / image_features_p.norm(dim=-1, keepdim=True)
        logit_scale_p = self.logit_scale.exp()
        logits_p = logit_scale_p * image_features_p @ text_features_p.t()

        clip_output = F.softmax(logits_p, dim=1)
        # print(weight, weight.size()) # [32,47]
        # weight, _ = torch.max(weight, 1)
        # # print(weight,weight.size()) # [32]
        _, pseudoClip = torch.max(logits_p, 1)


        #不使用mix up
        image_features = self.image_encoder(image, prompt_image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if (ii  == (r-1)):
           model_output = torch.nn.Softmax(dim=1)(logits)
           pseudoLabel, weight1, weight2 = self.select_predictions_score(clip_output, model_output)
        else:
           pseudoLabel = pseudoClip
        # pseudoLabel = pseudoClip
        # pseudoLabel, entropy_clip, entropy_model = self.select_predictions(clip_output, model_output)
        # pseudoLabel, weight1, weight2 = self.select_predictions_score(clip_output, model_output)

        # print(weight1, weight2)
        # 计算交叉熵损失
        # loss = self.metric(logits, pseudoLabel)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        loss = torch.mean(loss_fn(logits, pseudoLabel))

        # print(softmax_log,softmax_log.size()) # [32,47]
        # loss_emin = torch.mean(self.Entropy(softmax_log))

        # loss = self.metric(logits, pseudoLabel, weight)
        # loss = self.metric(logits, pseudoLabel)
        # self.writer.add_scalar('fix_loss', fix_loss, global_step=self.num_call)

        correct = 0.
        prediction = logits.argmax(dim=-1)
        correct += (prediction == label).float().sum()
        acc = correct / int(label.shape[0])

        self.acc.append(acc)
        self.best_accuracy = max(acc, self.best_accuracy)
        # if acc >= self.best_accuracy:
        #     self.best_prompt_text = prompt_text
        #     self.best_prompt_image = prompt_image
        # ---------------save_results-----------------------------------
        # output_dir = os.path.join(self.output_dir, self.task_name)
        #
        # fname = "{}_{}_{}.pth".format(self.task_name, self.opt_name, self.backbone.replace("/", "-"))
        # # fname = "{}_intrinsic_{}.pth".format(self.task_name, self.intrinsic_dim_L)
        #
        # content = {"task_name": self.task_name, "opt_name": self.opt_name, "backbone": self.backbone,
        #            "best_accuracy": self.best_accuracy, "acc": self.acc,
        #            "best_prompt_text": self.best_prompt_text, "best_prompt_image": self.best_prompt_image,
        #            "loss": self.loss, "num_call": self.num_call,
        #            "Linear_L": self.linear_L.state_dict(), "Linear_V": self.linear_V.state_dict()}
        # Analysis_Util.save_results(content, output_dir, fname)
        # ---------------save_results-----------------------------------
        # print("current loss: {}".format(self.min_loss))
        return loss.item(), acc.item()

    @torch.no_grad()
    def test(self):
        correct = 0.
        # parallel = self.parallel
        # self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = False
        for batch in self.test_loader:
            image, label = self.parse_batch(batch)
            text_features = self.text_encoder(self.best_prompt_text)
            image_features = self.image_encoder(image, self.best_prompt_image)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()
        # self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = parallel
        acc = correct / len(self.test_data)
        # print("Best Prompt Embedding - Acc : "+str(acc))
        return acc

    @torch.no_grad()
    def test_list(self):
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}
        start_test = True

        state_dict = torch.load("/home/fameng/project/BPT-VLM/result/Product/Product_shallow_cma_ViT-B-16.pth")
        correct = 0.

        # temp_p = "A photo of a {}."
        # prompts_p = [temp_p.format(c.replace("_", " ")) for c in self.classes]
        # prompts_p = torch.cat([clip.tokenize(p) for p in prompts_p])
        # prompts_p = prompts_p.to(self.device)
        # text_features = self.model.encode_text(prompts_p)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # parallel = self.parallel
        # self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = False
        for batch in self.test_loader:
            image, label = self.parse_batch(batch)
            text_features = self.text_encoder(state_dict['best_prompt_text'])
            image_features = self.image_encoder(image, state_dict['best_prompt_image'])
            # image_features = self.model.encode_image(image)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()

            if start_test:
                all_feas = image_features.cpu()
                all_output = prediction.float().cpu()
                all_label = label.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, prediction.float().cpu()), 0)
                all_label = torch.cat((all_label, label.float()), 0)
                all_feas = torch.cat((all_feas, image_features.cpu()), 0)

            for label, prediction in zip(label, prediction):
                if label == prediction:
                    correct_pred[self.classes[label]] += 1
                total_pred[self.classes[label]] += 1

        tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=300, n_iter=1500, perplexity=35)
        # tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=900, perplexity=60)
        result = tsne.fit_transform(all_feas)
        # ys = matplotlib.colors.ListedColormap(
        #     ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
        # ys = matplotlib.colors.ListedColormap('b')
        cm1 = plt.get_cmap('tab20')
        # cm2 = plt.get_cmap('Pastel1')
        # cm3 = plt.get_cmap('Pastel2')
        # ys = ListedColormap(cm1.colors + cm2.colors + cm3.colors)
        # ys = ListedColormap(cm1.colors)
        plt.scatter(result[:, 0], result[:, 1], s=40, c=all_output, marker='.', cmap=cm1, )
        plt.savefig('bpt.png')

        acc = correct / len(self.test_data)

        # for classname, correct_count in correct_pred.items():
        #     accuracy = 100 * float(correct_count) / total_pred[classname]
        #     print("Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy))
        #     print("***************************************")
        #     print(acc)
        # print("Best Prompt Embedding - Acc : "+str(acc))
        return acc

    @torch.no_grad()
    def test_pseudo(self):
        correct = 0.
        parallel = self.parallel
        self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = False
        start_test = True

        temp_p = "A photo of a {}."
        prompts_p = [temp_p.format(c.replace("_", " ")) for c in self.classes]
        prompts_p = torch.cat([clip.tokenize(p) for p in prompts_p])
        prompts_p = prompts_p.to(self.device)

        with torch.no_grad():
            text_features_p = self.model.encode_text(prompts_p)
            text_features_p = text_features_p / text_features_p.norm(dim=-1, keepdim=True)

        for batch in self.test_loader:
            image, label = self.parse_batch(batch)
            # text_features = self.text_encoder(self.best_prompt_text)
            image_features = self.model.encode_image(image)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_p = text_features_p / text_features_p.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features_p.t()
            # _,pl = torch.max(logits, 1)
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()

            if start_test:
                all_feas = image_features.cpu()
                all_output = prediction.float().cpu()
                all_label = label.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, prediction.float().cpu()), 0)
                all_label = torch.cat((all_label, label.float()), 0)
                all_feas = torch.cat((all_feas, image_features.cpu()), 0)

        self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = parallel
        tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=300, n_iter=1500, perplexity=35)
        # tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=900, perplexity=60)
        result = tsne.fit_transform(all_feas)
        # ys = matplotlib.colors.ListedColormap(
        #     ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
        # ys = matplotlib.colors.ListedColormap('b')
        cm1 = plt.get_cmap('tab20')
        # cm2 = plt.get_cmap('Pastel1')
        # cm3 = plt.get_cmap('Pastel2')
        # ys = ListedColormap(cm1.colors + cm2.colors + cm3.colors)
        # ys = ListedColormap(cm1.colors)
        plt.scatter(result[:, 0], result[:, 1], s=40, c=all_output, marker='.', cmap=cm1, )
        plt.savefig('clip.png')

        acc = correct / len(self.test_data)
        # print("Best Prompt Embedding - Acc : "+str(acc))
        return acc



    def parse_batch(self, batch):
        image = batch["image"]
        label = batch["label"]
        # print(image.dtype)
        # print("%%%")
        image = image.to(device=self.device, dtype=self.dtype)
        # print(image.dtype)
        # image = image.to(device=self.device)
        label = label.to(device=self.device)

        return image, label
