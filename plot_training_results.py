import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==============================================================================
#   PUBLICATION CONFIGURATION (IEEE 3.5" Column Width)
# ==============================================================================

# --- Dimensions ---
FIG_WIDTH = 3.5  
FIG_HEIGHT = 6.0 

# --- Typography ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']

# --- Font Sizes (Requested) ---
FS_SUPER_TITLE = 10  # Main Algorithm Name
FS_TITLE       = 9   # Plot Title (Reward, Fuel...)
FS_LABEL       = 7   # Axis Labels (Steps, Units)
FS_TICK        = 6   # Numbers on axis
FS_LEGEND      = 6   # Legend text

# --- Colors (Apple Human Interface Guidelines) ---
C_BLUE   = "#007AFF" 
C_ORANGE = "#FF9500" 
C_GREEN  = "#248A3D" 
C_GRID   = "#E5E5EA" 
C_TEXT   = "#1C1C1E" 

# ==============================================================================
#   HELPER FUNCTIONS
# ==============================================================================

def get_valid_result_folder(algo, reward, base_path="./results"):
    """Finds the latest valid result folder."""
    if not os.path.exists(base_path):
        print(f"❌ Base path '{base_path}' does not exist.")
        return None, None

    all_folders = glob.glob(os.path.join(base_path, "*"))
    matching_folders = []
    
    for f in all_folders:
        name = os.path.basename(f).lower()
        if algo.lower() in name and reward.lower() in name:
            matching_folders.append(f)
            
    if not matching_folders:
        print(f"❌ No folders found for {algo} + {reward}")
        return None, None

    matching_folders.sort(key=os.path.getmtime, reverse=True)
    
    for folder in matching_folders:
        csv_path = os.path.join(folder, "training_log.csv")
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 10: 
            continue
        try:
            df = pd.read_csv(csv_path)
            if len(df) < 5: continue 
            print(f"✅ Using run: {os.path.basename(folder)}")
            return folder, df
        except:
            continue

    return None, None

def style_axis(ax):
    """Applies the clean, paper-ready 'despined' look."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines['left'].set_color(C_GRID)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_color(C_TEXT)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Minimal Grid
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color=C_GRID, alpha=1.0)
    ax.grid(axis='x', visible=False)
    
    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=FS_TICK, 
                   color=C_GRID, labelcolor=C_TEXT, length=2)

def plot_paper_metric(df, x_col, y_col, ax, span=1000, color=C_BLUE, title="", units=""):
    """Plots a metric with EWMA smoothing and a Legend."""
    
    # 1. Calculate Stats
    mean = df[y_col].ewm(span=span).mean()
    std  = df[y_col].ewm(span=span).std()
    x = df[x_col]
    
    # 2. Plot Main Line (Label for Legend)
    ax.plot(x, mean, color=color, linewidth=1.2, zorder=3, label='Mean')
    
    # 3. Plot Shadow (Label for Legend)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0, zorder=2, label='Std Dev')
    
    # 4. Styling
    style_axis(ax)
    
    # 5. Typography
    # Title: Bold, Size 9, Left Aligned
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', color=C_TEXT, loc='left', pad=10)
    
    # Y-Axis Unit: Top of axis, horizontal
    ax.text(0, 1.02, units, transform=ax.transAxes, 
            fontsize=FS_LABEL, fontweight='normal', color=C_TEXT, 
            ha='left', va='bottom')

    # 6. LEGEND (New)
    # frameon=False removes the box for a cleaner look
    ax.legend(loc='upper right', fontsize=FS_LEGEND, frameon=False, labelcolor=C_TEXT)

    # 7. Formatter
    if df[y_col].max() > 1000:
        ax.yaxis.set_major_formatter(ticker.EngFormatter())

# ==============================================================================
#   MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Paper-Ready RL Plots")
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--smoothness", type=int, default=500, help="EWMA Span") 
    args = parser.parse_args()

    # 1. Load Data
    folder_path, df = get_valid_result_folder(args.algo, args.reward)
    if df is None: return

    # 2. Setup Figure
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
    
    # 3. Plotting
    if 'episode_reward' in df.columns:
        plot_paper_metric(df, 'global_step', 'episode_reward', axes[0], 
                          span=args.smoothness, color=C_BLUE, 
                          title="Episode Reward", units="Total Return")
        
    if 'episode_length' in df.columns:
        plot_paper_metric(df, 'global_step', 'episode_length', axes[1], 
                          span=args.smoothness, color=C_ORANGE, 
                          title="Episode Length", units="Steps")

    if 'fuel_remaining' in df.columns:
        plot_paper_metric(df, 'global_step', 'fuel_remaining', axes[2], 
                          span=args.smoothness, color=C_GREEN, 
                          title="Fuel Remaining", units="Mass Unit")

    # 4. Final Layout
    
    # Bottom Label
    axes[2].set_xlabel("Training Steps", fontsize=FS_LABEL, fontweight='bold', color=C_TEXT, labelpad=5)
    axes[2].xaxis.set_major_formatter(ticker.EngFormatter())
    
    # Main Title - Tighter to the graphs
    # y=0.99 pushes it to the very top edge
    fig.suptitle(f"Algorithm: {args.algo.upper()}", 
                 x=0.02, y=0.99, ha='left', 
                 fontsize=FS_SUPER_TITLE, fontweight='bold', color=C_TEXT)

    # Tight Layout - rect[3]=0.96 leaves just enough room for the suptitle
    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    
    # 5. Save
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    safe_name = os.path.basename(folder_path)
    path_png = os.path.join(plots_dir, f"{safe_name}_paper.png")
    path_pdf = os.path.join(plots_dir, f"{safe_name}_paper.pdf")
    
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.savefig(path_pdf, format='pdf', bbox_inches='tight')
    
    print(f"✨ Paper-Ready Plot Saved: {path_png}")

if __name__ == "__main__":
    main()