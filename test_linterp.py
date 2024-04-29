import torch
from algorithms.fscrl.linterp import LinearInterpolationModule
from ds_manager import DSManager
import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = DSManager(name="lucas_downsampled_min")
    x, _ = d.get_all_X_y()
    ax = x[0].tolist()

    xs = torch.linspace(0,1,len(ax)).tolist()
    plt.plot(ax)
    plt.show()
    plt.scatter(xs, ax)

    y = torch.tensor(x, dtype=torch.float32, device='cuda').to(device)
    interp = LinearInterpolationModule(y, device)

    x_new = torch.linspace(-0.1,1.1,10).to(device)
    x_new = x_new[torch.randperm(x_new.size(0))].to(device)
    print(x_new)
    y_new = interp(x_new)

    plt.scatter(x_new.cpu().numpy(), y_new[0].cpu().numpy(), marker='.', c="red")
    plt.show()

    criterion = torch.nn.MSELoss(reduction='mean')
    x_new = x_new[torch.randperm(x_new.size(0))].to(device)
    x_new = torch.nn.Parameter(x_new, requires_grad=True)
    optim = torch.optim.Adam([x_new], lr=0.01)
    for i in range(10):
        print(x_new)
        y = torch.tensor(x, dtype=torch.float32, device='cuda').to(device)
        ones = torch.ones(y.shape[0],10).to(device)
        y_new = interp(x_new)
        interp = LinearInterpolationModule(y, device)
        y_new = interp(x_new)
        loss = criterion(ones, y_new)
        loss.backward()
        print(loss)
        optim.step()
        optim.zero_grad()


