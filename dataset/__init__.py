import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image   # 引入python imaging library（pil）中的image模块，用于操作和处理图像。

from dataset.medvqa_dataset import rad_dataset, pathvqa_dataset, slake_dataset
from dataset.icm_caption_dataset import pretrain_dataset as med_pretrain_dataset

from dataset.randaugment import RandomAugment  # 导入 rand augment 模块中的 random augment 类，做数据增强。


def create_dataset(dataset, config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # 用于对图像张量进行标准化，使输入数据更一致
    pretrain_transform = transforms.Compose([     # 对图像数据进行预处理。
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),  # 随机裁剪
        transforms.RandomHorizontalFlip(),                                                    # 随机水平翻转
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        # 随机数据增强
        transforms.ToTensor(),       # 使用ToTensor将图像转化为张量
        normalize,                   # 对张量进行归一化
    ])
    train_transform = transforms.Compose([   # 数据预处理操作组合器
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # Medical image caption   dataset
    if dataset == 'med_pretrain':          # 如果数据集名称为‘’，则创建数据集，并以给定的配置文件来初始化数据集。
        dataset = med_pretrain_dataset(config['train_file'], pretrain_transform, image_root=config['image_root'])
        return dataset

    # vqa_rad

    #  path vqa
    elif dataset == 'pathvqa':
        train_dataset = pathvqa_dataset(config['train_file'], train_transform, config['vqa_root'], split='train')
        test_dataset = pathvqa_dataset(config['test_file'], test_transform, config['vqa_root'], split='test',
                                   answer_list=config['answer_list'])
        return train_dataset, test_dataset
    # slake
    elif dataset == 'slake':
        train_dataset = pathvqa_dataset(config['train_file'], train_transform, config['vqa_root'], split='train')
        test_dataset = pathvqa_dataset(config['test_file'], test_transform, config['vqa_root'], split='test',
                                       answer_list=config['answer_list'])
        return train_dataset, test_dataset


def vqa_collate_fn(batch):                              # 将一个batch的数据进行预处理
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))             # 将数据添加到列表
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n   # 堆叠操作，返回一个四元组


def create_sampler(datasets, shuffles, num_tasks, global_rank):              # 数据采样器
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):           # 打包元组，再分别赋值给两个变量。
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)      # 对数据集进行分布式采样操作
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):     # 数据加载器
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):      # 打包元组，然后赋值
        if is_train:    # 训练集
            shuffle = (sampler is None)
            drop_last = True              # 最后一批数据的大小小于 batch_size 时将其丢弃。
        else:          # 测试集
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,                 # 是否将数据存储在CUDA固定内存中
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
