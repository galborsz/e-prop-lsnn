import matplotlib.pyplot as plt
import numpy as np

def heatmaps(network, outputs, y_batch):
    output_value = np.argmax(outputs, axis=1)[0]
    target_value = y_batch[0]

    # Create one-hot encodings
    output_one_hot = np.zeros((network.n_out,))
    target_one_hot = np.zeros((network.n_out,))
    output_one_hot[output_value] = 1
    target_one_hot[target_value] = 1

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))  # Keeping the 1x4 grid

    # Heatmap for one-hot encoded output
    axs[0].imshow(output_one_hot[:, None], cmap='viridis', aspect=0.1)  # Set aspect ratio to make it thinner
    axs[0].set_title("One-Hot Encoded Output")
    axs[0].set_ylabel("Output Neuron Index")
    axs[0].set_yticks(range(0, network.n_out, 2))  # Show every other tick on the y-axis
    axs[0].set_xticks([])

    # Heatmap for one-hot encoded target
    axs[1].imshow(target_one_hot[:, None], cmap='viridis', aspect=0.1)  # Set aspect ratio to make it thinner
    axs[1].set_title("One-Hot Encoded Target")
    axs[1].set_ylabel("Output Neuron Index")
    axs[1].set_yticks(range(0, network.n_out, 2))  # Show every other tick on the y-axis
    axs[1].set_xticks([])

    # Heatmap for output weights
    im_w_out = axs[2].imshow(np.transpose(network.w_out), cmap='viridis')  # Set aspect ratio to make it thinner
    axs[2].set_title("Output Weights Heatmap")
    axs[2].set_xlabel("Recurrent Neuron Index")
    axs[2].set_xticks(range(network.n_rec))
    axs[2].set_ylabel("Output Neuron Index")
    axs[2].set_yticks(range(0, network.n_out, 2))  # Show every other tick on the y-axis

    # Heatmap for output bias
    im_b_out = axs[3].imshow(network.b_out[:, None], cmap='viridis', aspect=0.1)  # Set aspect ratio to make it thinner
    axs[3].set_title("Output Bias Heatmap")
    axs[3].set_xticks([])
    axs[3].set_ylabel("Output Neuron Index")
    axs[3].set_yticks(range(0, network.n_out, 2))  # Show every other tick on the y-axis

    # Add a shared color bar for the last two heatmaps
    cbar = fig.colorbar(im_w_out, ax=axs[3], orientation='vertical', fraction=0.05)
    cbar.set_label('Magnitude')

    # Adjust layout and show the plots
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.007)
    plt.show()