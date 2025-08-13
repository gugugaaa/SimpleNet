import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.style as style
import os  # 新增

def parse_data(file_path):
    # Read data from file
    with open(file_path, 'r') as f:
        data = f.read()
    
    # Split the data into lines
    lines = data.strip().split('\n')

    # Initialize lists to store the parsed data
    metrics = {
        'step': [],
        'loss': [],
        'lr': [],
        'p_true': [],
        'p_fake': [],
        'i_auroc': []
    }

    # Regex patterns
    epoch_pattern = re.compile(r'epoch:(\d+) loss:(\d+\.\d+) lr:(\d+\.\d+) p_true:(\d+\.\d+) p_fake:(\d+\.\d+)')
    auroc_pattern = re.compile(r'----- (\d+) I-AUROC:(\d+\.\d+)\(MAX:(\d+\.\d+)\) -----')

    # Iterate through the lines and extract data
    for i in range(0, len(lines), 2):
        epoch_line = lines[i]
        auroc_line = lines[i+1]

        epoch_match = epoch_pattern.search(epoch_line)
        auroc_match = auroc_pattern.search(auroc_line)

        if epoch_match and auroc_match:
            # Extract epoch line data
            # epoch = int(epoch_match.group(1)) # epoch is always 1
            loss = float(epoch_match.group(2))
            lr = float(epoch_match.group(3))
            p_true = float(epoch_match.group(4))
            p_fake = float(epoch_match.group(5))

            # Extract auroc line data
            step = int(auroc_match.group(1))
            i_auroc = float(auroc_match.group(2))

            # Append to lists
            metrics['step'].append(step)
            metrics['loss'].append(loss)
            metrics['lr'].append(lr)
            metrics['p_true'].append(p_true)
            metrics['p_fake'].append(p_fake)
            metrics['i_auroc'].append(i_auroc)
    
    return pd.DataFrame(metrics)

def plot_metrics(df, save_dir=None):
    # W&B Style Plotting
    # Use a dark theme
    style.use('dark_background')

    # Set up the figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Define W&B-like colors
    colors = ['#FF4E50', '#FC9A03', '#39A7D9', '#66D9EF']

    # --- Loss Plot ---
    axs[0].plot(df['step'], df['loss'], label='Loss', color=colors[0], marker='o', linestyle='-')
    axs[0].set_title('Loss over Steps', fontsize=16, color='white')
    axs[0].set_ylabel('Loss', fontsize=12, color='white')
    axs[0].grid(True, linestyle='--', alpha=0.6, color='gray')
    axs[0].legend()
    axs[0].tick_params(axis='both', colors='white')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_color('white')
    axs[0].spines['left'].set_color('white')

    # --- I-AUROC Plot ---
    axs[1].plot(df['step'], df['i_auroc'], label='I-AUROC', color=colors[1], marker='o', linestyle='-')
    axs[1].set_title('I-AUROC over Steps', fontsize=16, color='white')
    axs[1].set_ylabel('I-AUROC', fontsize=12, color='white')
    axs[1].grid(True, linestyle='--', alpha=0.6, color='gray')
    axs[1].legend()
    axs[1].tick_params(axis='both', colors='white')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_color('white')
    axs[1].spines['left'].set_color('white')

    # --- p_true and p_fake Plot ---
    axs[2].plot(df['step'], df['p_true'], label='p_true', color=colors[2], marker='o', linestyle='-')
    axs[2].plot(df['step'], df['p_fake'], label='p_fake', color=colors[3], marker='o', linestyle='--')
    axs[2].set_title('p_true and p_fake over Steps', fontsize=16, color='white')
    axs[2].set_xlabel('Step', fontsize=12, color='white')
    axs[2].set_ylabel('Probability', fontsize=12, color='white')
    axs[2].grid(True, linestyle='--', alpha=0.6, color='gray')
    axs[2].legend()
    axs[2].tick_params(axis='both', colors='white')
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['bottom'].set_color('white')
    axs[2].spines['left'].set_color('white')

    plt.tight_layout(pad=3.0)
    if save_dir is not None:
        save_path = os.path.join(save_dir, "wandb.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 设置文件路径变量
    file_path = r"results\01\metric_timeline.txt"  # 修改为你的日志文件路径
    
    df = parse_data(file_path)
    save_dir = os.path.dirname(os.path.abspath(file_path))
    plot_metrics(df, save_dir=save_dir)