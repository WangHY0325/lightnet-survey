import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import math
import time
import matplotlib

# 设置后端为 'Agg' 以支持无头服务器绘图
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. 全局配置 & 论文参数表
# ==========================================
DATA_ROOT = '/gpool/home/wanghongyang/WangHY/Aatucker/data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# [Fig 5/6 Reproduction Config]
# 使用 VGG19 配合 Lambda=0.0019 (论文配置)
# 如果仍然崩塌，可微调为 1e-4
MODEL_ARCH = 'vgg19'

if MODEL_ARCH == 'vgg13':
    LAMBDA_REG = 1e-6
elif MODEL_ARCH == 'vgg16':
    LAMBDA_REG = 2e-6
elif MODEL_ARCH == 'vgg19':
    LAMBDA_REG = 3e-6  # 论文原值，若崩塌请改为 1e-4
else:
    raise ValueError(f"Unknown model {MODEL_ARCH}")

PRETRAIN_LR = 0.01
COMPRESS_LR = 0.005
MU_START = 5e-4
MU_STEP = 1.15
MU_MAX = 10000.0
PRETRAIN_EPOCHS = 10  # 演示用，建议设为 50+
COMPRESS_EPOCHS = 60


# ==========================================
# 2. 数据加载
# ==========================================
def get_dataloader(batch_size=128):
    print(f"Loading CIFAR-10 from {DATA_ROOT}/cifar ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_root = os.path.join(DATA_ROOT, 'cifar')
    trainset = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=False, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    return trainloader, testloader


# ==========================================
# 3. VGG 模型定义
# ==========================================
cfg = {
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_CIFAR(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_CIFAR, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        print("Initializing weights (He Init)...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ==========================================
# 4. 秩选择算法
# ==========================================
def get_best_tucker_rank_conv(weight_tensor, lambda_reg, mu):
    O, I, Kh, Kw = weight_tensor.shape
    W_np = weight_tensor.detach().cpu().numpy()
    if not np.isfinite(W_np).all(): W_np = np.nan_to_num(W_np)

    Mat_In = np.transpose(W_np, (1, 0, 2, 3)).reshape(I, -1)
    try:
        U_in, _, _ = np.linalg.svd(Mat_In, full_matrices=False)
    except:
        U_in = np.zeros((I, I), dtype=W_np.dtype)

    Mat_Out = W_np.reshape(O, -1)
    try:
        U_out, _, _ = np.linalg.svd(Mat_Out, full_matrices=False)
    except:
        U_out = np.zeros((O, O), dtype=W_np.dtype)

    W_t = torch.from_numpy(W_np).to(DEVICE)
    U_in_t = torch.from_numpy(U_in).to(DEVICE)
    U_out_t = torch.from_numpy(U_out).to(DEVICE)

    Core_full = torch.einsum('oihw,is,ot->tshw', W_t, U_in_t, U_out_t)
    Core_sq = (Core_full ** 2).cpu().numpy()

    max_r_out = Core_full.shape[0]
    max_r_in = Core_full.shape[1]
    Energy_map = np.sum(Core_sq, axis=(2, 3))
    Total_Energy = np.sum(Energy_map)
    Cum_Energy = np.cumsum(np.cumsum(Energy_map, axis=0), axis=1)

    min_obj = float('inf')
    best_ranks = (max_r_out, max_r_in)

    for r_out in range(1, max_r_out + 1):
        for r_in in range(1, max_r_in + 1):
            params = (r_out * r_in * Kh * Kw) + (I * r_in) + (O * r_out)
            kept_energy = Cum_Energy[r_out - 1, r_in - 1]
            error = Total_Energy - kept_energy
            obj = lambda_reg * params + (mu / 2) * error
            if obj < min_obj:
                min_obj = obj
                best_ranks = (r_out, r_in)

    r_out_best, r_in_best = best_ranks
    U_in_trunc = U_in_t[:, :r_in_best]
    U_out_trunc = U_out_t[:, :r_out_best]
    Core_trunc = Core_full[:r_out_best, :r_in_best, :, :]
    Beta = torch.einsum('tshw,is,ot->oihw', Core_trunc, U_in_trunc, U_out_trunc)
    orig_params = O * I * Kh * Kw
    comp_params = (r_out_best * r_in_best * Kh * Kw) + (I * r_in_best) + (O * r_out_best)
    return Beta, (r_out_best, r_in_best), orig_params, comp_params


def get_best_svd_rank_fc(weight_matrix, lambda_reg, mu):
    O, I = weight_matrix.shape
    W_np = weight_matrix.detach().cpu().numpy()
    if not np.isfinite(W_np).all(): W_np = np.nan_to_num(W_np)
    try:
        U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    except:
        return weight_matrix, min(O, I), O * I, O * I

    S_sq = S ** 2
    Total_Energy = np.sum(S_sq)
    Cum_Energy = np.cumsum(S_sq)
    min_obj = float('inf')
    best_rank = len(S)

    for r in range(1, len(S) + 1):
        params = r * (r + I + O)
        kept_energy = Cum_Energy[r - 1]
        error = Total_Energy - kept_energy
        obj = lambda_reg * params + (mu / 2) * error
        if obj < min_obj:
            min_obj = obj
            best_rank = r

    U_t = torch.from_numpy(U[:, :best_rank]).to(DEVICE)
    S_t = torch.from_numpy(np.diag(S[:best_rank])).to(DEVICE)
    Vt_t = torch.from_numpy(Vt[:best_rank, :]).to(DEVICE)
    Beta = torch.mm(torch.mm(U_t, S_t), Vt_t)
    return Beta, best_rank, O * I, best_rank * (best_rank + I + O)


# ==========================================
# 5. 绘图函数 (新增)
# ==========================================
def plot_fig5(history):
    epochs = range(1, len(history['loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, history['loss'], color='tab:blue', marker='o', markersize=3, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Acc (%)', color='tab:red')
    ax2.plot(epochs, history['acc'], color='tab:red', marker='s', markersize=3, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'Fig 5. Training Loss and Test Accuracy ({MODEL_ARCH})')
    fig.tight_layout()
    plt.savefig('Fig5_Evolution.png')
    print("Saved Fig5_Evolution.png")


def plot_fig6(final_ranks):
    layer_names = []
    r1_list = []
    r2_list = []
    r_fc_list = []

    conv_idx = 1
    fc_idx = 1
    # 过滤掉非 weight 参数
    keys = [k for k in final_ranks.keys() if 'weight' in k]

    for name in keys:
        ranks = final_ranks[name]
        if isinstance(ranks, list):  # Conv [R1, R2]
            layer_names.append(f'Conv{conv_idx}')
            r1_list.append(ranks[0])
            r2_list.append(ranks[1])
            r_fc_list.append(0)
            conv_idx += 1
        else:  # FC scalar
            layer_names.append(f'FC{fc_idx}')
            r1_list.append(0)
            r2_list.append(0)
            r_fc_list.append(ranks)
            fc_idx += 1

    x = np.arange(len(layer_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width / 2, r1_list, width, label='R1 (Conv)', color='orange')
    rects2 = ax.bar(x + width / 2, r2_list, width, label='R2 (Conv)', color='green')
    rects3 = ax.bar(x, r_fc_list, width, label='R (FC)', color='red', alpha=0.5)

    ax.set_ylabel('Rank')
    ax.set_title(f'Fig 6. Final Rank Distribution ({MODEL_ARCH})')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45)
    ax.legend()

    fig.tight_layout()
    plt.savefig('Fig6_RankDistribution.png')
    print("Saved Fig6_RankDistribution.png")


# ==========================================
# 6. 主流程
# ==========================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def pretrain_model(model, train_loader, test_loader, epochs=5):
    print(f"\n[Step 0] Pre-training Network for {epochs} epochs...")
    optimizer = optim.SGD(model.parameters(), lr=PRETRAIN_LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)],
                                                     gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader)
        print(
            f"  Pretrain Epoch {epoch + 1}/{epochs} | Loss: {train_loss / len(train_loader):.4f} | Test Acc: {acc:.2f}%")
    print("Pre-training Finished.\n")


def run_aatucker():
    trainloader, testloader = get_dataloader()
    print(f"Building {MODEL_ARCH} ...")
    model = VGG_CIFAR(MODEL_ARCH).to(DEVICE)

    # 1. 预训练
    pretrain_model(model, trainloader, testloader, epochs=PRETRAIN_EPOCHS)

    # 2. 初始化
    betas = {}
    thetas = {}
    param_names = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() in [2, 4]:
            betas[name] = param.data.clone()
            thetas[name] = torch.zeros_like(param.data)
            param_names.append(name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=COMPRESS_LR, momentum=0.9, weight_decay=5e-4)
    mu = MU_START

    print(f"Start AATucker Compression Loop for {MODEL_ARCH} (lambda={LAMBDA_REG})...")

    # Data for plots
    history = {'loss': [], 'acc': []}
    final_ranks_snapshot = {}

    final_comp_p = 0
    final_orig_p = 0

    for epoch in range(COMPRESS_EPOCHS):
        start_time = time.time()

        # Step 1: W
        model.train()
        train_loss = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_cls = criterion(outputs, targets)
            loss_admm = 0.0
            for name in param_names:
                param = dict(model.named_parameters())[name]
                z = betas[name] - thetas[name] / mu
                loss_admm += (mu / 2) * torch.sum((param - z) ** 2)
            loss = loss_cls + loss_admm
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Step 2: Beta
        total_orig_p = 0
        total_comp_p = 0
        with torch.no_grad():
            for name in param_names:
                param = dict(model.named_parameters())[name]
                target = param.data + thetas[name] / mu
                if param.dim() == 4:
                    new_beta, ranks, op, cp = get_best_tucker_rank_conv(target, LAMBDA_REG, mu)
                    final_ranks_snapshot[name] = list(ranks)  # Save for Fig 6
                else:
                    new_beta, rank, op, cp = get_best_svd_rank_fc(target, LAMBDA_REG, mu)
                    final_ranks_snapshot[name] = rank  # Save for Fig 6

                betas[name] = new_beta
                total_orig_p += op
                total_comp_p += cp
                thetas[name] = thetas[name] + mu * (param.data - betas[name])

        final_comp_p = total_comp_p
        final_orig_p = total_orig_p
        mu = min(mu * MU_STEP, MU_MAX)

        # Logging for Plot (Use Beta Acc)
        backup_params = {}
        with torch.no_grad():
            for name in param_names:
                p = dict(model.named_parameters())[name]
                backup_params[name] = p.data.clone()
                p.data = betas[name]

        acc_beta = evaluate(model, testloader)

        with torch.no_grad():
            for name in param_names:
                dict(model.named_parameters())[name].data = backup_params[name]

        # Record history
        avg_loss = train_loss / len(trainloader)
        history['loss'].append(avg_loss)
        history['acc'].append(acc_beta)

        cr = total_orig_p / total_comp_p if total_comp_p > 0 else 1.0
        print(
            f"Epoch {epoch + 1}/{COMPRESS_EPOCHS} | Loss: {avg_loss:.4f} | Acc(Beta): {acc_beta:.2f}% | CR: {cr:.2f}x")

    print("\nTraining Finished. Generating Plots...")

    # 3. 绘图
    plot_fig5(history)
    plot_fig6(final_ranks_snapshot)

    # Final Result
    final_cr = final_orig_p / final_comp_p if final_comp_p > 0 else 1.0
    print("\n" + "=" * 80)
    print(f" RESULT REPORT ({MODEL_ARCH.upper()}, lambda={LAMBDA_REG})")
    print("=" * 80)
    print(f"| #params (Uncompressed): {int(final_orig_p)}")
    print(f"| #params (Compressed):   {int(final_comp_p)}")
    print(f"| Compression Ratio (CR): {final_cr:.2f}x")
    print(f"| Final Accuracy:         {history['acc'][-1]:.2f}%")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.makedirs('logs')
    run_aatucker()
